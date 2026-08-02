//! Transport + multi-track sequencer. Owns the musical clock (a beat-domain
//! playhead), a flat sorted note-event schedule, and the sample-accurate
//! dispatch that splits each render block at event boundaries. Each event
//! carries a track index routing it to that track's own `PolySynth`; all
//! instruments are summed and master soft-clipped.
//!
//! Why beats, not samples or seconds: beats are the natural domain for MIDI
//! import and the piano-roll, survive tempo changes, and keep f32 precision
//! healthy (a 10-minute song at 120 BPM is only ~1200 beats). The playhead is
//! converted to samples per block via `beats_per_sample = bpm / 60 / sr`.

use wasm_bindgen::prelude::*;

use crate::drums::DrumKit;
use crate::synth::PolySynth;

/// One note on/off event at a musical position. `vel == 0` is a note-off;
/// `track` selects the instrument it routes to.
#[derive(Clone, Copy)]
struct NoteEvent {
    beat: f32,
    midi: f32,
    vel: f32,
    track: usize,
}

/// A per-note pitch-bend curve, keyed to its note-on by `(beat, midi, track)`.
/// `pts` are `(beat_offset_from_head, absolute_midi)` breakpoints; they are
/// converted to seconds (using the current tempo) and handed to the voice when
/// the matching note-on dispatches. Kept *beside* the flat schedule rather than
/// folded into it so the stride-4 `[beat, midi, vel, track]` event stream — and
/// its well-tested reconcile/loop logic — stays untouched. Only the schedule
/// dispatch path applies bends; reconcile-triggered notes (seek/loop into the
/// middle of a held note) sound flat at their head pitch.
struct Bend {
    beat: f32,
    midi: f32,
    track: usize,
    pts: Vec<(f32, f32)>,
}

/// A track's instrument: either a polyphonic synth (melodic / piano-roll
/// content) or a drum kit (percussion / step grid). The sequencer holds a
/// homogeneous `Vec<TrackInstrument>` and routes every per-track call through
/// this enum, so the transport and scheduler stay instrument-agnostic — a drum
/// pattern is just notes routed to a `Drum` track. Engine/wavetable selection
/// is synth-only; drums ignore it.
enum TrackInstrument {
    Poly(PolySynth),
    Drum(DrumKit),
}

impl TrackInstrument {
    fn note_on(&mut self, midi: f32, velocity: f32) {
        match self {
            TrackInstrument::Poly(s) => s.note_on(midi, velocity),
            TrackInstrument::Drum(d) => d.note_on(midi, velocity),
        }
    }
    /// Trigger with a pitch-bend curve (stride-2 `[sec, midi, ...]` after the
    /// head). `initial_secs` seeds the bend clock for a note started mid-glide.
    /// Drums are unpitched (their midi is a lane index), so a bend is meaningless
    /// there — they just hit.
    fn note_on_bend(&mut self, midi: f32, velocity: f32, bend_pts: &[f32], initial_secs: f32) {
        match self {
            TrackInstrument::Poly(s) => s.note_on_bend(midi, velocity, bend_pts, initial_secs),
            TrackInstrument::Drum(d) => d.note_on(midi, velocity),
        }
    }
    fn note_off(&mut self, midi: f32) {
        match self {
            TrackInstrument::Poly(s) => s.note_off(midi),
            TrackInstrument::Drum(d) => d.note_off(midi),
        }
    }
    fn release_all(&mut self) {
        match self {
            TrackInstrument::Poly(s) => s.release_all(),
            TrackInstrument::Drum(d) => d.release_all(),
        }
    }
    fn set_param(&mut self, key: &str, value: f32) {
        match self {
            TrackInstrument::Poly(s) => s.set_param(key, value),
            TrackInstrument::Drum(d) => d.set_param(key, value),
        }
    }
    fn set_engine(&mut self, index: u32) {
        if let TrackInstrument::Poly(s) = self {
            s.set_engine(index);
        }
    }
    fn set_wavetable(&mut self, data: &[f32]) {
        if let TrackInstrument::Poly(s) = self {
            s.set_wavetable(data);
        }
    }
    fn render_add(&mut self, out: &mut [f32]) {
        match self {
            TrackInstrument::Poly(s) => s.render_add(out),
            TrackInstrument::Drum(d) => d.render_add(out),
        }
    }
}

#[wasm_bindgen]
pub struct Sequencer {
    sample_rate: f32,
    /// One instrument per track (synth or drum kit). Sized via `set_track_count`.
    synths: Vec<TrackInstrument>,
    /// Per-track MIDI notes currently sounding *from the schedule* (not live
    /// keys), so reconciliation can release/trigger scheduled voices at
    /// discontinuities without disturbing live-played ones. Plain Vec (not a
    /// HashSet) to avoid pulling in wasm RNG seeding for the default hasher.
    active: Vec<Vec<i32>>,

    schedule: Vec<NoteEvent>,
    /// Index of the next event to dispatch. Advances monotonically while
    /// playing; recomputed on seek / loop-wrap / schedule change.
    event_cursor: usize,

    playing: bool,
    /// Musical playhead in beats.
    playhead: f32,
    bpm: f32,

    loop_enabled: bool,
    loop_start: f32,
    loop_end: f32,

    /// Per-note pitch-bend curves, looked up by note-on identity at dispatch.
    /// Replaced wholesale by `set_bends` alongside every `set_schedule`.
    bends: Vec<Bend>,
}

#[wasm_bindgen]
impl Sequencer {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Sequencer {
        Sequencer {
            sample_rate,
            synths: vec![TrackInstrument::Poly(PolySynth::new(sample_rate))],
            active: vec![Vec::new()],
            schedule: Vec::new(),
            event_cursor: 0,
            playing: false,
            playhead: 0.0,
            bpm: 120.0,
            loop_enabled: false,
            loop_start: 0.0,
            loop_end: 0.0,
            bends: Vec::new(),
        }
    }

    /// Ensure there is one instrument per track. Grows/shrinks the pool;
    /// minimum of 1. New instruments default to the subtractive engine.
    pub fn set_track_count(&mut self, count: usize) {
        let count = count.max(1);
        let sr = self.sample_rate;
        while self.synths.len() < count {
            self.synths.push(TrackInstrument::Poly(PolySynth::new(sr)));
        }
        self.synths.truncate(count);
        while self.active.len() < count {
            self.active.push(Vec::new());
        }
        self.active.truncate(count);
    }

    // --- Per-track control (live keys + panel) ---------------------------
    // All guard the track index so a stale message can't panic.

    pub fn note_on(&mut self, track: usize, midi: f32, velocity: f32) {
        if let Some(s) = self.synths.get_mut(track) {
            s.note_on(midi, velocity);
        }
    }
    pub fn note_off(&mut self, track: usize, midi: f32) {
        if let Some(s) = self.synths.get_mut(track) {
            s.note_off(midi);
        }
    }
    pub fn set_param(&mut self, track: usize, key: &str, value: f32) {
        if let Some(s) = self.synths.get_mut(track) {
            s.set_param(key, value);
        }
    }
    pub fn set_engine(&mut self, track: usize, index: u32) {
        if let Some(s) = self.synths.get_mut(track) {
            s.set_engine(index);
        }
    }
    pub fn set_wavetable(&mut self, track: usize, data: &[f32]) {
        if let Some(s) = self.synths.get_mut(track) {
            s.set_wavetable(data);
        }
    }

    /// Switch a track's instrument kind (0 = poly synth, 1 = drum kit),
    /// rebuilding that track's instrument when the kind actually changes.
    /// Guarded so a redundant resend doesn't silence a sounding instrument, and
    /// so out-of-range tracks are ignored.
    pub fn set_instrument_kind(&mut self, track: usize, kind: u32) {
        let sr = self.sample_rate;
        if let Some(slot) = self.synths.get_mut(track) {
            let want_drum = kind == 1;
            let is_drum = matches!(slot, TrackInstrument::Drum(_));
            if want_drum == is_drum {
                return;
            }
            *slot = if want_drum {
                TrackInstrument::Drum(DrumKit::new(sr))
            } else {
                TrackInstrument::Poly(PolySynth::new(sr))
            };
        }
    }

    // --- Transport -------------------------------------------------------

    pub fn set_tempo(&mut self, bpm: f32) {
        self.bpm = bpm.clamp(1.0, 1000.0);
    }
    pub fn play(&mut self) {
        self.playing = true;
        // Start any note that already spans the playhead (e.g. pressing play in
        // the middle of a held note).
        self.reconcile();
    }
    pub fn pause(&mut self) {
        self.playing = false;
        // Release every track's voices so held scheduled notes don't drone, and
        // forget what was sounding.
        for s in &mut self.synths {
            s.release_all();
        }
        for set in &mut self.active {
            set.clear();
        }
    }
    pub fn seek(&mut self, beat: f32) {
        self.playhead = beat.max(0.0);
        self.resync_cursor();
        // A jump changes which notes should be sounding — drop the orphaned ones
        // and start the ones the new position lands inside.
        if self.playing {
            self.reconcile();
        }
    }
    pub fn set_loop(&mut self, start: f32, end: f32, enabled: bool) {
        self.loop_start = start.max(0.0);
        self.loop_end = end.max(0.0);
        self.loop_enabled = enabled;
    }

    /// Replace the schedule. `events` is stride-4 `[beat, midi, velocity, track]`;
    /// velocity 0 = note-off. Sorted here so callers needn't pre-sort.
    pub fn set_schedule(&mut self, events: &[f32]) {
        let mut sched = Vec::with_capacity(events.len() / 4);
        for ev in events.chunks_exact(4) {
            sched.push(NoteEvent {
                beat: ev[0],
                midi: ev[1],
                vel: ev[2],
                track: ev[3].max(0.0) as usize,
            });
        }
        // Stable sort by beat keeps same-beat off-before-on ordering as authored.
        sched.sort_by(|a, b| a.beat.partial_cmp(&b.beat).unwrap_or(std::cmp::Ordering::Equal));
        self.schedule = sched;
        self.resync_cursor();
        // An edit can delete/shorten/move a sounding note out from under its
        // pending note-off (→ stuck note) or extend one across the playhead
        // (→ should now sound). Reconcile against the new schedule.
        if self.playing {
            self.reconcile();
        }
    }

    /// Replace the pitch-bend curves. Self-describing flat layout: a sequence of
    /// records `[head_beat, head_midi, track, n, (t0, m0), (t1, m1), ...]` where
    /// `n` is the breakpoint count and each `(t, m)` is a `(beat_offset_from_head,
    /// absolute_midi)` pair. Sent alongside `set_schedule`; an empty array clears
    /// all bends. Note-ons whose `(beat, midi, track)` matches a record get the
    /// curve; everything else stays flat.
    pub fn set_bends(&mut self, data: &[f32]) {
        let mut bends = Vec::new();
        let mut i = 0;
        while i + 4 <= data.len() {
            let beat = data[i];
            let midi = data[i + 1];
            let track = data[i + 2].max(0.0) as usize;
            let n = data[i + 3].max(0.0) as usize;
            i += 4;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                if i + 2 > data.len() {
                    break; // truncated record — take what's there, don't panic
                }
                pts.push((data[i], data[i + 1]));
                i += 2;
            }
            bends.push(Bend { beat, midi, track, pts });
        }
        self.bends = bends;
    }

    pub fn playhead_beats(&self) -> f32 {
        self.playhead
    }
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// The bend curve for a note-on, as stride-2 `[sec, midi, ...]` breakpoints
    /// (beat offsets converted to seconds at the current tempo). `None` when no
    /// bend is keyed to this `(beat, midi, track)` — the note then sounds flat.
    fn bend_secs_for(&self, beat: f32, midi: f32, track: usize) -> Option<Vec<f32>> {
        let b = self.bends.iter().find(|b| {
            b.track == track && (b.beat - beat).abs() < 1e-3 && (b.midi - midi).abs() < 1e-3
        })?;
        if b.pts.is_empty() {
            return None;
        }
        let secs_per_beat = 60.0 / self.bpm;
        let mut out = Vec::with_capacity(b.pts.len() * 2);
        for &(beat_off, m) in &b.pts {
            out.push(beat_off * secs_per_beat);
            out.push(m);
        }
        Some(out)
    }

    /// Mix every track's instrument into `out` (cleared first), then master
    /// soft-clip so a dense arrangement stays bounded.
    fn render_mix(synths: &mut [TrackInstrument], out: &mut [f32]) {
        for s in out.iter_mut() {
            *s = 0.0;
        }
        for synth in synths.iter_mut() {
            synth.render_add(out);
        }
        for s in out.iter_mut() {
            *s = s.tanh();
        }
    }

    /// Fill `out` with one render block, dispatching scheduled events at their
    /// exact sample offset (block split at event boundaries) and routing each to
    /// its track's instrument. When paused the voices still run so release tails
    /// ring out, but the clock doesn't move.
    pub fn process(&mut self, out: &mut [f32]) {
        let n = out.len();
        if !self.playing || n == 0 {
            Self::render_mix(&mut self.synths, out);
            return;
        }

        let bps = self.bpm / 60.0 / self.sample_rate; // beats per sample
        let mut written = 0usize; // samples already filled in `out`

        // Outer loop iterates once per block normally; a second time only when
        // looping wraps the playhead mid-block.
        while written < n {
            let span_start = written;
            let span_start_beat = self.playhead;

            // How far can this span run: to end-of-block, or to the loop point
            // if looping would wrap first.
            let mut span = n - written;
            let mut wrap = false;
            if self.loop_enabled
                && self.loop_end > self.loop_start
                && span_start_beat + span as f32 * bps >= self.loop_end
            {
                // ceil so we never under-run the boundary by a sample.
                let to_end = ((self.loop_end - span_start_beat) / bps).ceil();
                span = (to_end as usize).clamp(1, n - written);
                wrap = true;
            }
            let span_stop = span_start + span;
            let span_end_beat = span_start_beat + span as f32 * bps;

            // Dispatch every event whose beat falls within this span.
            while self.event_cursor < self.schedule.len() {
                let ev = self.schedule[self.event_cursor];
                if ev.beat >= span_end_beat {
                    break;
                }
                let off_in_span = if ev.beat <= span_start_beat {
                    0
                } else {
                    ((ev.beat - span_start_beat) / bps).round() as usize
                };
                let off = (span_start + off_in_span).clamp(written, span_stop);
                if off > written {
                    Self::render_mix(&mut self.synths, &mut out[written..off]);
                    written = off;
                }
                if ev.vel > 0.0 {
                    // Resolve the bend before the mutable synth borrow (both touch
                    // `self`): `bend_secs_for` borrows `&self.bends`, returns owned.
                    let bend = self.bend_secs_for(ev.beat, ev.midi, ev.track);
                    if let Some(synth) = self.synths.get_mut(ev.track) {
                        match bend {
                            // Struck at its head, so the bend clock starts at 0.
                            Some(b) => synth.note_on_bend(ev.midi, ev.vel, &b, 0.0),
                            None => synth.note_on(ev.midi, ev.vel),
                        }
                    }
                } else if let Some(synth) = self.synths.get_mut(ev.track) {
                    synth.note_off(ev.midi);
                }
                if let Some(set) = self.active.get_mut(ev.track) {
                    let m = ev.midi.round() as i32;
                    if ev.vel > 0.0 {
                        if !set.contains(&m) {
                            set.push(m);
                        }
                    } else {
                        set.retain(|&x| x != m);
                    }
                }
                self.event_cursor += 1;
            }

            // Fill the remainder of the span.
            if span_stop > written {
                Self::render_mix(&mut self.synths, &mut out[written..span_stop]);
                written = span_stop;
            }

            self.playhead = span_end_beat;
            if wrap {
                self.playhead = self.loop_start;
                self.resync_cursor();
                // Notes still sounding at the loop boundary would otherwise hang
                // (their note-off lies past loop_end). Reconcile to release them.
                self.reconcile();
            }
        }
    }

    /// Reposition `event_cursor` to the first event at or after the playhead.
    fn resync_cursor(&mut self) {
        self.event_cursor = self
            .schedule
            .iter()
            .position(|e| e.beat >= self.playhead)
            .unwrap_or(self.schedule.len());
    }

    /// Make the sounding scheduled voices match what the schedule says should be
    /// playing at the current playhead: release notes that shouldn't sound, and
    /// trigger notes that should (and aren't already). Called at every playhead
    /// or schedule discontinuity (play, seek, edit, loop wrap). Live-played
    /// voices aren't tracked in `active`, so they're left alone.
    fn reconcile(&mut self) {
        let desired = self.schedule_state_at(self.playhead);
        for track in 0..self.active.len() {
            let want = &desired[track];
            // Release sounding notes that are no longer wanted.
            let mut i = 0;
            while i < self.active[track].len() {
                let midi = self.active[track][i];
                if want.iter().any(|&(m, _, _)| m == midi) {
                    i += 1;
                } else {
                    if let Some(s) = self.synths.get_mut(track) {
                        s.note_off(midi as f32);
                    }
                    self.active[track].swap_remove(i);
                }
            }
            // Trigger wanted notes that aren't already sounding. A note started
            // mid-glide (the playhead sits past its head) resumes its bend at the
            // elapsed offset, so seeking/looping into a bent note tracks pitch
            // instead of replaying the head pitch.
            for &(midi, vel, head_beat) in want {
                if self.active[track].contains(&midi) {
                    continue;
                }
                let bend = self.bend_secs_for(head_beat, midi as f32, track);
                let elapsed_secs = (self.playhead - head_beat).max(0.0) * 60.0 / self.bpm;
                if let Some(s) = self.synths.get_mut(track) {
                    match bend {
                        Some(b) => s.note_on_bend(midi as f32, vel, &b, elapsed_secs),
                        None => s.note_on(midi as f32, vel),
                    }
                }
                self.active[track].push(midi);
            }
        }
    }

    /// The notes sounding per track at `beat` as `(midi, velocity, head_beat)` —
    /// walk the sorted schedule, applying on/off events strictly before `beat`.
    /// `head_beat` (the most recent note-on's beat) lets reconcile resume a
    /// pitch-bend at the right offset when starting mid-note.
    fn schedule_state_at(&self, beat: f32) -> Vec<Vec<(i32, f32, f32)>> {
        let mut state: Vec<Vec<(i32, f32, f32)>> = vec![Vec::new(); self.synths.len()];
        for ev in &self.schedule {
            if ev.beat >= beat {
                break; // schedule is sorted by beat
            }
            if let Some(set) = state.get_mut(ev.track) {
                let m = ev.midi.round() as i32;
                if ev.vel > 0.0 {
                    match set.iter_mut().find(|(mm, _, _)| *mm == m) {
                        // Re-strike updates velocity and head beat.
                        Some(slot) => {
                            slot.1 = ev.vel;
                            slot.2 = ev.beat;
                        }
                        None => set.push((m, ev.vel, ev.beat)),
                    }
                } else {
                    set.retain(|&(mm, _, _)| mm != m);
                }
            }
        }
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn seq_with(events: &[f32], bpm: f32) -> Sequencer {
        let mut s = Sequencer::new(SR);
        s.set_tempo(bpm);
        s.set_schedule(events);
        s
    }

    #[test]
    fn paused_does_not_advance_playhead() {
        let mut s = seq_with(&[0.0, 60.0, 1.0, 0.0], 120.0);
        let mut buf = [0.0f32; 256];
        s.process(&mut buf);
        assert_eq!(s.playhead_beats(), 0.0);
    }

    #[test]
    fn playhead_advances_by_expected_beats() {
        // 120 BPM → 2 beats/sec. One block of `SR` samples = 1 second = 2 beats.
        let mut s = seq_with(&[], 120.0);
        s.play();
        let mut buf = vec![0.0f32; SR as usize];
        s.process(&mut buf);
        assert!((s.playhead_beats() - 2.0).abs() < 1e-3, "got {}", s.playhead_beats());
    }

    #[test]
    fn note_on_lands_at_sample_accurate_offset() {
        // Event at beat 1.0; at 120 BPM that's 0.5s = SR/2 samples in.
        let mut s = seq_with(&[1.0, 69.0, 1.0, 0.0], 120.0);
        s.play();
        let mut buf = vec![0.0f32; SR as usize];
        s.process(&mut buf);
        let half = (SR / 2.0) as usize;
        // Before the note (allowing for attack from any prior state) the first
        // quarter is silent; after the onset, energy appears.
        let pre: f32 = buf[..half / 2].iter().map(|x| x.abs()).sum();
        let post: f32 = buf[half..half + half / 2].iter().map(|x| x.abs()).sum();
        assert!(pre < 1e-4, "expected silence before note, got {pre}");
        assert!(post > 1e-2, "expected sound after note, got {post}");
    }

    #[test]
    fn loop_wraps_playhead_within_window() {
        let mut s = seq_with(&[], 120.0);
        s.set_loop(0.0, 1.0, true); // 1-beat loop = 0.5s at 120 BPM
        s.play();
        // 2 seconds of audio = 4 beats of advance, must stay within [0, 1).
        let block = (SR / 4.0) as usize; // 0.5 beat per block
        let mut buf = vec![0.0f32; block];
        for _ in 0..16 {
            s.process(&mut buf);
            assert!(
                s.playhead_beats() >= 0.0 && s.playhead_beats() < 1.0 + 1e-3,
                "playhead escaped loop: {}",
                s.playhead_beats()
            );
        }
    }

    #[test]
    fn seek_resyncs_event_cursor() {
        let mut s = seq_with(&[0.0, 60.0, 1.0, 0.0, 4.0, 64.0, 1.0, 0.0], 120.0);
        s.seek(3.0);
        // Next event at/after beat 3 is the one at beat 4 (index 1 → cursor 1).
        assert_eq!(s.event_cursor, 1);
    }

    #[test]
    fn schedule_routes_events_to_their_track() {
        let mut s = Sequencer::new(SR);
        s.set_track_count(2);
        s.set_tempo(120.0);
        // A note on track 0 and a note on track 1, both at beat 0.
        s.set_schedule(&[0.0, 60.0, 1.0, 0.0, 0.0, 64.0, 1.0, 1.0]);
        s.play();
        let mut buf = vec![0.0f32; 1024];
        s.process(&mut buf);
        let energy: f32 = buf.iter().map(|x| x.abs()).sum();
        assert!(energy > 1e-2, "both tracks should sound, got {energy}");
    }

    #[test]
    fn out_of_range_track_is_ignored_not_panicked() {
        let mut s = Sequencer::new(SR); // one track (index 0)
        s.set_tempo(120.0);
        s.set_schedule(&[0.0, 60.0, 1.0, 5.0]); // track 5 doesn't exist
        s.play();
        let mut buf = vec![0.0f32; 256];
        s.process(&mut buf); // must not panic
    }

    fn energy(buf: &[f32]) -> f32 {
        buf.iter().map(|x| x.abs()).sum()
    }

    #[test]
    fn deleting_a_sounding_note_releases_it() {
        let mut s = Sequencer::new(SR);
        s.set_tempo(120.0);
        // Note on at beat 0, off at beat 8 (long).
        s.set_schedule(&[0.0, 60.0, 1.0, 0.0, 8.0, 60.0, 0.0, 0.0]);
        s.play();
        let mut buf = vec![0.0f32; (SR / 4.0) as usize];
        s.process(&mut buf);
        assert!(energy(&buf) > 1e-2, "note should be sounding");

        s.set_schedule(&[]); // delete it mid-note while playing
        let mut fade = vec![0.0f32; SR as usize];
        s.process(&mut fade); // contains the release fade
        let mut late = vec![0.0f32; SR as usize];
        s.process(&mut late); // ~1 s later — must be silent
        assert!(energy(&late) < 1e-3, "deleted note must release, got {}", energy(&late));
    }

    #[test]
    fn starting_playback_inside_a_note_sounds_it() {
        let mut s = Sequencer::new(SR);
        s.set_tempo(120.0);
        s.set_schedule(&[0.0, 60.0, 1.0, 0.0, 8.0, 60.0, 0.0, 0.0]); // beats 0..8
        s.seek(4.0); // jump into the middle while paused (no trigger yet)
        s.play(); // play() reconciles → starts the spanning note
        let mut buf = vec![0.0f32; (SR / 4.0) as usize];
        s.process(&mut buf);
        assert!(energy(&buf) > 1e-2, "note spanning the start point should sound");
    }

    #[test]
    fn drum_kind_track_routes_and_sounds() {
        let mut s = Sequencer::new(SR);
        s.set_track_count(1);
        s.set_instrument_kind(0, 1); // make track 0 a drum kit
        s.set_tempo(120.0);
        // A kick (MIDI 36) at beat 0 on the drum track.
        s.set_schedule(&[0.0, 36.0, 1.0, 0.0]);
        s.play();
        let mut buf = vec![0.0f32; 1024];
        s.process(&mut buf);
        assert!(energy(&buf) > 1e-2, "drum hit should sound, got {}", energy(&buf));
    }

    #[test]
    fn set_instrument_kind_is_idempotent() {
        // A redundant resend of the same kind must not panic or rebuild away a
        // sounding voice (guarded no-op).
        let mut s = Sequencer::new(SR);
        s.set_instrument_kind(0, 0); // already a poly synth → no-op
        s.set_instrument_kind(5, 1); // out-of-range track → ignored, no panic
        s.note_on(0, 60.0, 1.0);
        let mut buf = vec![0.0f32; 256];
        s.process(&mut buf);
        assert!(energy(&buf) > 1e-2, "poly synth note should still sound");
    }

    #[test]
    fn seeking_past_a_notes_end_releases_it() {
        let mut s = Sequencer::new(SR);
        s.set_tempo(120.0);
        s.set_schedule(&[0.0, 60.0, 1.0, 0.0, 2.0, 60.0, 0.0, 0.0]); // note 0..2
        s.play();
        let mut buf = vec![0.0f32; (SR / 4.0) as usize];
        s.process(&mut buf); // sounding
        s.seek(5.0); // jump past the note's end
        let mut fade = vec![0.0f32; SR as usize];
        s.process(&mut fade);
        let mut late = vec![0.0f32; SR as usize];
        s.process(&mut late);
        assert!(energy(&late) < 1e-3, "seek past end must release, got {}", energy(&late));
    }

    #[test]
    fn scheduled_note_with_bend_glides_pitch() {
        // Full bend path: set_bends parse → match the note-on by (beat, midi,
        // track) → beat→sec conversion → voice glide.
        let mut s = Sequencer::new(SR);
        s.set_tempo(120.0); // 2 beats/sec → 1 beat = 0.5s
        s.set_schedule(&[0.0, 60.0, 1.0, 0.0, 4.0, 60.0, 0.0, 0.0]); // note 0..4 beats
        // Head (beat 0, midi 60, track 0); one breakpoint 2 beats in (= 1.0s) to
        // midi 72 (an octave up).
        s.set_bends(&[0.0, 60.0, 0.0, 1.0, 2.0, 72.0]);
        s.play();
        let n = SR as usize; // 1 second of audio = the full glide
        let mut buf = vec![0.0f32; n];
        s.process(&mut buf);
        let zc = |sl: &[f32]| sl.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        let early = zc(&buf[..n / 10]);
        let late = zc(&buf[n - n / 10..]);
        assert!(late > early, "scheduled bend should glide up: early {early} late {late}");
    }

    #[test]
    fn note_with_no_matching_bend_stays_flat() {
        let mut s = Sequencer::new(SR);
        s.set_tempo(120.0);
        s.set_schedule(&[0.0, 60.0, 1.0, 0.0, 4.0, 60.0, 0.0, 0.0]);
        s.set_bends(&[0.0, 67.0, 0.0, 1.0, 2.0, 79.0]); // bend keyed to midi 67, not 60
        s.play();
        let n = SR as usize;
        let mut buf = vec![0.0f32; n];
        s.process(&mut buf);
        let zc = |sl: &[f32]| sl.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        let early = zc(&buf[n / 10..2 * n / 10]) as i32;
        let late = zc(&buf[n - 2 * n / 10..n - n / 10]) as i32;
        assert!((early - late).abs() <= 3, "unmatched bend must not pitch-shift: {early} vs {late}");
    }

    #[test]
    fn playing_into_a_bent_note_resumes_the_glide() {
        let zc = |sl: &[f32]| sl.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        let n = (SR / 4.0) as usize; // 0.25s window

        // A 4-beat note (0..4) gliding 60 → 72, reaching 72 at beat 2 (= 1s).
        let mut s = Sequencer::new(SR);
        s.set_tempo(120.0); // 1 beat = 0.5s
        s.set_schedule(&[0.0, 60.0, 1.0, 0.0, 4.0, 60.0, 0.0, 0.0]);
        s.set_bends(&[0.0, 60.0, 0.0, 1.0, 2.0, 72.0]);
        s.seek(1.0); // beat 1 = 0.5s into the glide → ~66
        s.play(); // reconcile triggers the spanning note mid-glide
        let mut mid = vec![0.0f32; n];
        s.process(&mut mid);

        // Reference: the same note with no bend — held flat at the head pitch 60.
        let mut flat = Sequencer::new(SR);
        flat.set_tempo(120.0);
        flat.set_schedule(&[0.0, 60.0, 1.0, 0.0, 4.0, 60.0, 0.0, 0.0]);
        flat.seek(1.0);
        flat.play();
        let mut flatbuf = vec![0.0f32; n];
        flat.process(&mut flatbuf);

        // Skip the (re)attack transient, then compare fundamentals.
        let mid_zc = zc(&mid[n / 4..]);
        let flat_zc = zc(&flatbuf[n / 4..]);
        assert!(
            mid_zc > flat_zc,
            "playing into the bend should track pitch (>head), got {mid_zc} vs {flat_zc}"
        );
    }
}
