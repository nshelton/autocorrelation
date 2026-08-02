//! Polyphonic subtractive synth — the audio-thread counterpart to `Dsp`, but
//! inverted: it *generates* samples instead of analyzing them.
//!
//! A `PolySynth` owns a fixed, pre-allocated pool of `Voice`s (one per
//! simultaneous note) plus a note-stealing allocator. Each voice is:
//!   oscillator (+ optional detuned partner) → pre-filter drive → Cytomic TPT
//!   lowpass (modulated by a filter ADSR / LFO) → post-filter drive → linear
//!   ADSR amp envelope (× LFO tremolo).
//! Voices sum *additively* into the output, then a master `tanh` soft-clip keeps
//! a dense chord from clipping. The pool is pre-allocated so the realtime audio
//! path never allocates. The extra stages stay inert at their default params
//! (detune/drive/filter-env-amount/LFO-depth = 0), so a default voice is exactly
//! the original saw → filter → ADSR and skips the per-sample modulation work.
//!
//! Deliberately simple where it doesn't matter yet:
//!   - the saw is naive (`2·phase − 1`), so it aliases above a few kHz. PolyBLEP
//!     anti-aliasing is a later refinement, not a correctness issue.
//!   - the envelope is linear, not exponential.
//!   - all voices share one timbre (one instrument). Per-track timbres are a
//!     later phase.
//!
//! `PolySynth`/`Voice` are plain internal structs — the `Sequencer`
//! (sequencer.rs) owns the `PolySynth` and is the wasm-bindgen surface.

/// Maximum simultaneous voices. New notes past this steal the quietest voice.
pub const MAX_VOICES: usize = 16;

/// Filter-envelope sweep range: `filterEnvAmount` ∈ [-1, 1] scales the cutoff by
/// up to ±this many octaves at full envelope level.
const FILTER_ENV_OCTAVES: f32 = 6.0;
/// LFO→pitch range in semitones at full depth (lfo ∈ [-1, 1]).
const LFO_PITCH_SEMIS: f32 = 12.0;
/// LFO→cutoff range in octaves at full depth.
const LFO_CUTOFF_OCTAVES: f32 = 4.0;
/// Waveshaper pre-gain at full `drive` (1 + drive·this, fed through `tanh`).
const DRIVE_GAIN: f32 = 15.0;

/// Oscillator engine — selects how a voice generates its raw waveform before
/// the shared filter + envelope. Adding an engine is a new variant + a branch
/// in `Voice::add`; everything downstream (filter, envelope, polyphony) is
/// shared. Index mapping (JS ↔ Rust) lives in `PolySynth::set_engine`.
#[derive(Clone, Copy, PartialEq)]
pub enum Engine {
    /// Naive sawtooth — the original subtractive voice.
    Subtractive,
    /// Single-cycle wavetable (generated JS-side from simplex noise).
    Simplex,
}

/// ADSR stages. `Idle` means fully released and silent — the voice is free for
/// reallocation.
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Linear ADSR. Times are in seconds; `level`/`sustain` are 0..1 amplitudes.
struct Adsr {
    stage: Stage,
    level: f32,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    dt: f32,
}

impl Adsr {
    fn new(dt: f32) -> Adsr {
        Adsr {
            stage: Stage::Idle,
            level: 0.0,
            attack: 0.005,
            decay: 0.1,
            sustain: 0.7,
            release: 0.2,
            dt,
        }
    }

    fn trigger(&mut self) {
        self.stage = Stage::Attack;
    }

    fn release_now(&mut self) {
        // Release from wherever the level currently is, so a key lifted mid-attack
        // doesn't jump to full before fading.
        if self.stage != Stage::Idle {
            self.stage = Stage::Release;
        }
    }

    /// Advance one sample and return the current amplitude. Rates are derived
    /// per call from the stage times so live param edits take effect mid-note.
    fn next(&mut self) -> f32 {
        match self.stage {
            Stage::Idle => self.level = 0.0,
            Stage::Attack => {
                // `attack == 0` would divide by zero → jump straight to full.
                let rate = if self.attack > 0.0 { self.dt / self.attack } else { 1.0 };
                self.level += rate;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.stage = Stage::Decay;
                }
            }
            Stage::Decay => {
                let rate = if self.decay > 0.0 { self.dt / self.decay } else { 1.0 };
                self.level -= rate * (1.0 - self.sustain);
                if self.level <= self.sustain {
                    self.level = self.sustain;
                    self.stage = Stage::Sustain;
                }
            }
            Stage::Sustain => self.level = self.sustain,
            Stage::Release => {
                let rate = if self.release > 0.0 { self.dt / self.release } else { 1.0 };
                // Release rate is relative to the sustain level so the tail length
                // is roughly the configured `release` regardless of where it starts.
                self.level -= rate * self.sustain.max(1e-4);
                if self.level <= 0.0 {
                    self.level = 0.0;
                    self.stage = Stage::Idle;
                }
            }
        }
        self.level
    }
}

/// Per-voice low-frequency oscillator for vibrato / tremolo / filter wobble.
/// Phase in [0, 1); retriggered per note so the modulation starts consistently
/// rather than wherever a free-running shared LFO happens to be.
struct Lfo {
    phase: f32,
    dt: f32,
}

impl Lfo {
    fn new(dt: f32) -> Lfo {
        Lfo { phase: 0.0, dt }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Advance one sample and return the waveform value in [-1, 1]. `shape`:
    /// 0 = sine, 1 = triangle, 2 = square, 3 = rising saw.
    fn next(&mut self, rate: f32, shape: u32) -> f32 {
        let p = self.phase;
        let v = match shape {
            // Triangle phased to start at 0 like the sine (0 → 1 → -1 → 0).
            1 => {
                if p < 0.25 {
                    4.0 * p
                } else if p < 0.75 {
                    2.0 - 4.0 * p
                } else {
                    4.0 * p - 4.0
                }
            }
            2 => {
                if p < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            3 => 2.0 * p - 1.0,
            _ => (p * std::f32::consts::TAU).sin(),
        };
        self.phase += rate * self.dt;
        if self.phase >= 1.0 {
            self.phase -= self.phase.floor();
        }
        v
    }
}

/// A single synth voice: oscillator(s) + filter + envelopes + modulation.
struct Voice {
    sample_rate: f32,

    // Oscillator. Pitch is derived from `current_note` (+ instrument offset /
    // bend / LFO) each block, so the voice carries no separate `freq`.
    phase: f32,

    // Cytomic TPT state-variable filter state + params.
    // (Andrew Simper, "Solving the continuous SVF equations", 2013.)
    ic1eq: f32,
    ic2eq: f32,
    cutoff: f32,
    resonance: f32,

    env: Adsr,
    velocity: f32,
    /// MIDI note this voice is sounding, so note-off / steal decisions can find
    /// the right voice. Stays set after release until the voice is reused. This
    /// is the note's *base* pitch and stays fixed even while `bend` re-pitches
    /// the oscillator, so note-off still matches by the authored midi.
    current_note: f32,
    gain: f32,

    // Oscillator pitch offset (instrument tuning) + unison detune. `octave`/
    // `semi`/`fine` shift every note's pitch; `detune` (cents) adds a second
    // oscillator at +detune for a fatter unison (skipped entirely when 0).
    octave: f32,
    semi: f32,
    fine: f32,
    detune: f32,
    /// Second oscillator phase for unison detune (free-running like `phase`;
    /// only summed when `detune > 0`).
    phase2: f32,

    // Filter envelope: a second ADSR that scales the cutoff by up to
    // `filter_env_amount * FILTER_ENV_OCTAVES` octaves. Amount 0 (the default)
    // bypasses it and the per-sample filter-coefficient recompute it forces.
    filter_env: Adsr,
    filter_env_amount: f32,

    // Low-frequency modulation. `lfo_target`: 0 = pitch, 1 = cutoff, 2 = amp.
    // `lfo_shape`: 0 = sine, 1 = triangle, 2 = square, 3 = saw. Inert at depth 0.
    lfo: Lfo,
    lfo_rate: f32,
    lfo_depth: f32,
    lfo_target: u32,
    lfo_shape: u32,

    // Waveshaping drive (`tanh`). `drive_mode`: 0 = pre-filter, 1 = post-filter.
    drive: f32,
    drive_mode: u32,

    /// Pitch-bend breakpoints `(seconds_from_trigger, midi)`, sorted by time and
    /// starting with the head at `(0, base midi)`. Empty = no bend (fixed
    /// `freq`). The seconds are pre-converted from beats at note-on using the
    /// tempo then, so a mid-note tempo change can drift the bend timing slightly.
    bend: Vec<(f32, f32)>,
    /// Samples elapsed since the note was triggered — the bend's clock.
    bend_elapsed: u32,
}

impl Voice {
    fn new(sample_rate: f32) -> Voice {
        Voice {
            sample_rate,
            phase: 0.0,
            ic1eq: 0.0,
            ic2eq: 0.0,
            cutoff: 4000.0,
            resonance: 0.7,
            env: Adsr::new(1.0 / sample_rate),
            velocity: 1.0,
            current_note: f32::NAN,
            gain: 0.25,
            octave: 0.0,
            semi: 0.0,
            fine: 0.0,
            detune: 0.0,
            phase2: 0.0,
            filter_env: {
                let mut e = Adsr::new(1.0 / sample_rate);
                // Pluck-friendly default shape (decay to no sustain); inert until
                // `filter_env_amount` is turned up, so it doesn't affect a default
                // voice. Must mirror the `f*` keys in INSTRUMENT_DEFAULTS.
                e.decay = 0.2;
                e.sustain = 0.0;
                e
            },
            filter_env_amount: 0.0,
            lfo: Lfo::new(1.0 / sample_rate),
            lfo_rate: 5.0,
            lfo_depth: 0.0,
            lfo_target: 0,
            lfo_shape: 0,
            drive: 0.0,
            drive_mode: 0,
            bend: Vec::new(),
            bend_elapsed: 0,
        }
    }

    fn note_on(&mut self, midi: f32, velocity: f32) {
        self.note_on_bend(midi, velocity, &[], 0.0);
    }

    /// Trigger with a pitch-bend curve. `bend_pts` is stride-2 `[sec, midi, ...]`
    /// breakpoints *after* the head (each `sec` is time from the note's head,
    /// `midi` the absolute target pitch); the head `(0, midi)` is prepended here.
    /// Empty `bend_pts` → a flat note at `midi` (the `note_on` case), so the
    /// per-sample bend cost is only paid by notes that bend. `initial_secs` seeds
    /// the bend clock: 0 for a note struck at its head, or the elapsed time when a
    /// note is started mid-glide (reconcile after a seek/loop into the middle of
    /// it) so it resumes at the correct pitch instead of jumping back to the head.
    fn note_on_bend(&mut self, midi: f32, velocity: f32, bend_pts: &[f32], initial_secs: f32) {
        self.velocity = velocity.clamp(0.0, 1.0);
        self.current_note = midi;
        self.bend.clear();
        if !bend_pts.is_empty() {
            self.bend.push((0.0, midi)); // implicit head at the base pitch
            for p in bend_pts.chunks_exact(2) {
                self.bend.push((p[0], p[1]));
            }
        }
        self.bend_elapsed = (initial_secs.max(0.0) * self.sample_rate) as u32;
        self.env.trigger();
        self.filter_env.trigger();
        self.lfo.reset();
    }

    fn release(&mut self) {
        self.env.release_now();
        self.filter_env.release_now();
    }

    fn matches(&self, midi: f32) -> bool {
        !self.current_note.is_nan() && (self.current_note - midi).abs() < 0.5
    }

    fn is_active(&self) -> bool {
        self.env.stage != Stage::Idle
    }

    fn is_releasing(&self) -> bool {
        self.env.stage == Stage::Release
    }

    /// Current envelope amplitude — the steal heuristic prefers the quietest.
    fn level(&self) -> f32 {
        self.env.level
    }

    fn set_param(&mut self, key: &str, value: f32) {
        match key {
            "cutoff" => self.cutoff = value.clamp(20.0, self.sample_rate * 0.49),
            "resonance" => self.resonance = value.clamp(0.0, 1.0),
            "attack" => self.env.attack = value.max(0.0),
            "decay" => self.env.decay = value.max(0.0),
            "sustain" => self.env.sustain = value.clamp(0.0, 1.0),
            "release" => self.env.release = value.max(0.0),
            "gain" => self.gain = value.max(0.0),
            // Oscillator tuning / unison.
            "octave" => self.octave = value.clamp(-4.0, 4.0),
            "semi" => self.semi = value.clamp(-24.0, 24.0),
            "fine" => self.fine = value.clamp(-100.0, 100.0),
            "detune" => self.detune = value.clamp(0.0, 100.0),
            // Filter envelope (its own ADSR + a signed octave amount).
            "filterEnvAmount" => self.filter_env_amount = value.clamp(-1.0, 1.0),
            "fAttack" => self.filter_env.attack = value.max(0.0),
            "fDecay" => self.filter_env.decay = value.max(0.0),
            "fSustain" => self.filter_env.sustain = value.clamp(0.0, 1.0),
            "fRelease" => self.filter_env.release = value.max(0.0),
            // LFO. Discrete selectors arrive as floats → round to the index.
            "lfoRate" => self.lfo_rate = value.clamp(0.01, 40.0),
            "lfoDepth" => self.lfo_depth = value.clamp(0.0, 1.0),
            "lfoTarget" => self.lfo_target = value.max(0.0).round() as u32,
            "lfoShape" => self.lfo_shape = value.max(0.0).round() as u32,
            // Drive.
            "drive" => self.drive = value.clamp(0.0, 1.0),
            "driveMode" => self.drive_mode = value.max(0.0).round() as u32,
            _ => {}
        }
    }

    /// Add this voice's contribution into `out` (does not clear it). The
    /// oscillator source is chosen by `engine`; `wavetable` is the shared
    /// single-cycle table used by `Engine::Simplex`.
    fn add(&mut self, out: &mut [f32], engine: Engine, wavetable: &[f32]) {
        let inv_sr = 1.0 / self.sample_rate;
        let bending = !self.bend.is_empty();

        // Instrument pitch offset (octave/semitone/fine), applied to every note.
        let pitch_offset = self.octave * 12.0 + self.semi + self.fine * 0.01;
        // Unison detune: a second oscillator at +detune cents. Off when 0 so a
        // plain voice pays neither the extra oscillator nor the ratio `powf`.
        let detuned = self.detune > 0.0;
        let detune_ratio = if detuned { 2.0f32.powf(self.detune / 1200.0) } else { 1.0 };

        // LFO routing (depth 0 ⇒ inert, and the LFO isn't even advanced).
        let lfo_active = self.lfo_depth > 0.0;
        let lfo_pitch = lfo_active && self.lfo_target == 0;
        let lfo_cut = lfo_active && self.lfo_target == 1;
        let lfo_amp = lfo_active && self.lfo_target == 2;

        let fenv_on = self.filter_env_amount != 0.0;
        // Whether the per-sample pitch / filter-coefficient paths are needed at
        // all. When neither is, we keep the original cheap once-per-block math.
        let per_sample_pitch = bending || lfo_pitch;
        let per_sample_coeffs = fenv_on || lfo_cut;
        let drive_on = self.drive > 0.0;

        // Flat-path phase increments and static filter coefficients; the
        // per-sample branches below override them when modulation is active.
        let mut phase_inc = midi_to_hz(self.current_note + pitch_offset) * inv_sr;
        let mut phase_inc2 = phase_inc * detune_ratio;
        let base_coeffs = self.svf_coeffs(self.cutoff);

        for s in out.iter_mut() {
            let lfo = if lfo_active { self.lfo.next(self.lfo_rate, self.lfo_shape) } else { 0.0 };

            if per_sample_pitch {
                let mut m = if bending {
                    let t = self.bend_elapsed as f32 * inv_sr;
                    self.bend_elapsed += 1;
                    interp_bend(&self.bend, t)
                } else {
                    self.current_note
                };
                m += pitch_offset;
                if lfo_pitch {
                    m += LFO_PITCH_SEMIS * self.lfo_depth * lfo;
                }
                phase_inc = midi_to_hz(m) * inv_sr;
                phase_inc2 = phase_inc * detune_ratio;
            }

            // Oscillator (+ optional detuned partner, averaged to keep level).
            let mut osc = osc_sample(engine, wavetable, self.phase);
            if detuned {
                osc = 0.5 * (osc + osc_sample(engine, wavetable, self.phase2));
                self.phase2 += phase_inc2;
                if self.phase2 >= 1.0 {
                    self.phase2 -= 1.0;
                }
            }
            self.phase += phase_inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }

            // Pre-filter waveshaping.
            let mut x = osc;
            if drive_on && self.drive_mode == 0 {
                x = (x * (1.0 + self.drive * DRIVE_GAIN)).tanh();
            }

            // Filter coefficients: static per block, unless the cutoff is being
            // modulated (filter envelope and/or LFO), in which case recompute
            // per sample from the modulated cutoff.
            let (a1, a2, a3) = if per_sample_coeffs {
                let mut fc = self.cutoff;
                if fenv_on {
                    let fenv = self.filter_env.next();
                    fc *= 2.0f32.powf(self.filter_env_amount * FILTER_ENV_OCTAVES * fenv);
                }
                if lfo_cut {
                    fc *= 2.0f32.powf(LFO_CUTOFF_OCTAVES * self.lfo_depth * lfo);
                }
                self.svf_coeffs(fc.clamp(20.0, self.sample_rate * 0.49))
            } else {
                base_coeffs
            };

            // Cytomic TPT SVF, lowpass tap (v2).
            let v3 = x - self.ic2eq;
            let v1 = a1 * self.ic1eq + a2 * v3;
            let v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3;
            self.ic1eq = 2.0 * v1 - self.ic1eq;
            self.ic2eq = 2.0 * v2 - self.ic2eq;

            // Post-filter waveshaping.
            let mut y = v2;
            if drive_on && self.drive_mode == 1 {
                y = (y * (1.0 + self.drive * DRIVE_GAIN)).tanh();
            }

            let mut amp = self.env.next() * self.velocity * self.gain;
            if lfo_amp {
                // Unipolar tremolo: drops amplitude by up to `depth`, never above
                // unity, so it can't push the summed mix into the master clip.
                amp *= 1.0 - 0.5 * self.lfo_depth * (1.0 - lfo);
            }
            *s += y * amp;
        }
    }

    /// SVF coefficients for a given cutoff. Computed once per block on the flat
    /// path (no `tan` per sample); the modulated path calls this per sample.
    fn svf_coeffs(&self, cutoff: f32) -> (f32, f32, f32) {
        let g = (std::f32::consts::PI * cutoff / self.sample_rate).tan();
        // resonance 0..1 → damping k = 1/Q; resonance→1 gives high Q (k→~0.1).
        let k = 2.0 - 1.9 * self.resonance;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        (a1, a2, a3)
    }
}

/// Polyphonic instrument: a pool of voices plus the note-stealing allocator.
pub struct PolySynth {
    voices: Vec<Voice>,
    engine: Engine,
    /// Shared single-cycle wavetable for `Engine::Simplex`, generated JS-side
    /// from simplex noise and pushed down. Empty (→ silence) until set.
    wavetable: Vec<f32>,
}

impl PolySynth {
    pub fn new(sample_rate: f32) -> PolySynth {
        PolySynth {
            voices: (0..MAX_VOICES).map(|_| Voice::new(sample_rate)).collect(),
            engine: Engine::Subtractive,
            wavetable: Vec::new(),
        }
    }

    /// Select the oscillator engine (0 = subtractive saw, 1 = simplex wavetable).
    /// Unknown indices fall back to subtractive.
    pub fn set_engine(&mut self, index: u32) {
        self.engine = match index {
            1 => Engine::Simplex,
            _ => Engine::Subtractive,
        };
    }

    /// Replace the shared single-cycle wavetable used by `Engine::Simplex`.
    pub fn set_wavetable(&mut self, data: &[f32]) {
        self.wavetable.clear();
        self.wavetable.extend_from_slice(data);
    }

    pub fn note_on(&mut self, midi: f32, velocity: f32) {
        let idx = self.alloc_voice();
        self.voices[idx].note_on(midi, velocity);
    }

    /// Trigger a note carrying a pitch-bend curve. `bend_pts` is stride-2
    /// `[sec, midi, ...]` breakpoints after the head (see `Voice::note_on_bend`);
    /// `initial_secs` is the bend clock offset for a note started mid-glide.
    pub fn note_on_bend(&mut self, midi: f32, velocity: f32, bend_pts: &[f32], initial_secs: f32) {
        let idx = self.alloc_voice();
        self.voices[idx].note_on_bend(midi, velocity, bend_pts, initial_secs);
    }

    /// Release every voice currently sounding `midi` (and not already
    /// releasing). Multiple voices can match if a pitch was retriggered.
    pub fn note_off(&mut self, midi: f32) {
        for v in &mut self.voices {
            if v.matches(midi) && v.is_active() && !v.is_releasing() {
                v.release();
            }
        }
    }

    /// Release all voices — used by the transport on pause/stop.
    pub fn release_all(&mut self) {
        for v in &mut self.voices {
            v.release();
        }
    }

    /// Shared instrument params apply to every voice.
    pub fn set_param(&mut self, key: &str, value: f32) {
        for v in &mut self.voices {
            v.set_param(key, value);
        }
    }

    /// Sum all active voices into `out` (cleared first), then soft-clip so a
    /// dense chord stays bounded in [-1, 1].
    /// Sum this instrument's active voices into `out` **additively** (no clear,
    /// no soft-clip) so the sequencer can mix several instruments before a
    /// single master clip.
    pub fn render_add(&mut self, out: &mut [f32]) {
        // Disjoint field borrows: engine is Copy, wavetable immutable, voices
        // mutable — all distinct fields, so the borrow checker is happy.
        let engine = self.engine;
        let wavetable = &self.wavetable;
        for v in &mut self.voices {
            if v.is_active() {
                v.add(out, engine, wavetable);
            }
        }
    }

    /// Standalone render: clear, sum voices, master soft-clip. Test-only — the
    /// sequencer mixes tracks itself (render_add + one master clip).
    #[cfg(test)]
    pub fn process(&mut self, out: &mut [f32]) {
        for s in out.iter_mut() {
            *s = 0.0;
        }
        self.render_add(out);
        for s in out.iter_mut() {
            *s = s.tanh();
        }
    }

    /// Pick a voice for a new note: prefer an idle one, else steal the quietest.
    fn alloc_voice(&mut self) -> usize {
        if let Some(i) = self.voices.iter().position(|v| !v.is_active()) {
            return i;
        }
        let mut best = 0;
        let mut best_level = f32::INFINITY;
        for (i, v) in self.voices.iter().enumerate() {
            if v.level() < best_level {
                best_level = v.level();
                best = i;
            }
        }
        best
    }
}

/// MIDI note number → frequency in Hz (A4 = 69 = 440 Hz, equal temperament).
fn midi_to_hz(midi: f32) -> f32 {
    440.0 * 2.0_f32.powf((midi - 69.0) / 12.0)
}

/// Linear-interpolate a pitch-bend curve at time `t` seconds. `points` is sorted
/// ascending by time and starts at `t = 0`; the endpoints are held (constant
/// before the first / after the last), so a note rings on at its final bent
/// pitch past the last breakpoint. Interpolation is in the MIDI (log-frequency)
/// domain so a glide between two notes is a constant musical-interval ramp.
fn interp_bend(points: &[(f32, f32)], t: f32) -> f32 {
    match points.first() {
        None => 0.0,
        Some(&(t0, m0)) if t <= t0 => m0,
        Some(_) => {
            for w in points.windows(2) {
                let (t0, m0) = w[0];
                let (t1, m1) = w[1];
                if t <= t1 {
                    let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
                    return m0 + f * (m1 - m0);
                }
            }
            points[points.len() - 1].1
        }
    }
}

/// One oscillator sample for `engine` at `phase` ∈ [0, 1) — a naive saw for
/// `Subtractive`, the shared single-cycle table for `Simplex`. Factored out so
/// the unison-detune path can sample two phases through the same waveform.
fn osc_sample(engine: Engine, wavetable: &[f32], phase: f32) -> f32 {
    match engine {
        Engine::Subtractive => 2.0 * phase - 1.0, // naive saw in [-1, 1]
        Engine::Simplex => sample_wavetable(wavetable, phase),
    }
}

/// Linear-interpolated single-cycle wavetable lookup. `phase` in [0, 1); the
/// table wraps so the last sample interpolates back to the first.
fn sample_wavetable(table: &[f32], phase: f32) -> f32 {
    let n = table.len();
    if n == 0 {
        return 0.0;
    }
    let pos = phase * n as f32;
    let i0 = (pos as usize) % n;
    let i1 = (i0 + 1) % n;
    let frac = pos - pos.floor();
    table[i0] * (1.0 - frac) + table[i1] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn active(p: &PolySynth) -> usize {
        p.voices.iter().filter(|v| v.is_active()).count()
    }

    #[test]
    fn chord_allocates_distinct_voices_and_makes_sound() {
        let mut p = PolySynth::new(SR);
        p.note_on(60.0, 1.0);
        p.note_on(64.0, 1.0);
        p.note_on(67.0, 1.0);
        assert_eq!(active(&p), 3);

        let mut buf = [0.0f32; 256];
        p.process(&mut buf);
        let energy: f32 = buf.iter().map(|x| x.abs()).sum();
        assert!(energy > 1e-2, "expected a sounding chord, got {energy}");
        // Soft-clip keeps every sample bounded.
        assert!(buf.iter().all(|x| x.abs() <= 1.0));
    }

    #[test]
    fn note_off_releases_only_the_matching_voice() {
        let mut p = PolySynth::new(SR);
        p.note_on(60.0, 1.0);
        p.note_on(64.0, 1.0);
        p.note_off(60.0);

        // Run well past the release time (0.2 s ≈ 9600 samples).
        let mut buf = vec![0.0f32; 4096];
        for _ in 0..6 {
            p.process(&mut buf);
        }
        // Only the held note (64) should still be active.
        assert_eq!(active(&p), 1);
    }

    #[test]
    fn exceeding_max_voices_steals_rather_than_overflowing() {
        let mut p = PolySynth::new(SR);
        for n in 0..(MAX_VOICES + 5) {
            p.note_on(48.0 + n as f32, 1.0);
        }
        // The pool never grows past MAX_VOICES; all are active after triggering.
        assert_eq!(p.voices.len(), MAX_VOICES);
        assert_eq!(active(&p), MAX_VOICES);
    }

    #[test]
    fn simplex_engine_reads_the_wavetable() {
        let mut p = PolySynth::new(SR);
        // A simple non-DC single cycle.
        let table: Vec<f32> = (0..64)
            .map(|i| ((i as f32 / 64.0) * std::f32::consts::TAU).sin())
            .collect();
        p.set_wavetable(&table);
        p.set_engine(1); // Simplex
        p.note_on(60.0, 1.0);
        let mut buf = [0.0f32; 256];
        p.process(&mut buf);
        let energy: f32 = buf.iter().map(|x| x.abs()).sum();
        assert!(energy > 1e-2, "simplex engine should sound, got {energy}");
    }

    #[test]
    fn simplex_engine_is_silent_without_a_wavetable() {
        let mut p = PolySynth::new(SR);
        p.set_engine(1);
        p.note_on(60.0, 1.0);
        let mut buf = [0.0f32; 256];
        p.process(&mut buf);
        assert!(buf.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn interp_bend_is_piecewise_linear_with_held_ends() {
        let pts = [(0.0, 60.0), (1.0, 72.0)];
        assert_eq!(interp_bend(&pts, -0.5), 60.0); // before start: held
        assert!((interp_bend(&pts, 0.5) - 66.0).abs() < 1e-4); // midpoint
        assert_eq!(interp_bend(&pts, 2.0), 72.0); // after end: held
        assert_eq!(interp_bend(&[], 0.3), 0.0); // empty is harmless
    }

    /// Count upward zero crossings — a cheap proxy for fundamental frequency on
    /// the (mostly monotonic-per-cycle) saw.
    fn upward_crossings(s: &[f32]) -> usize {
        s.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count()
    }

    #[test]
    fn bend_glides_pitch_upward() {
        let mut p = PolySynth::new(SR);
        // Glide midi 60 → 72 (an octave, so ~2× frequency) over 1 second.
        p.note_on_bend(60.0, 1.0, &[1.0, 72.0], 0.0);
        let n = SR as usize; // 1 second
        let mut buf = vec![0.0f32; n];
        p.process(&mut buf);
        let early = upward_crossings(&buf[..n / 10]);
        let late = upward_crossings(&buf[n - n / 10..]);
        assert!(late > early, "pitch should rise over the glide: early {early} late {late}");
    }

    #[test]
    fn bend_started_mid_glide_resumes_at_the_right_pitch() {
        // The 60→72 glide over 1s, but started 0.5s in (as reconcile does when you
        // play into the middle of it). It must sound near the halfway pitch (~66),
        // higher than a flat note still at the head pitch (60).
        let n = (SR / 4.0) as usize; // 0.25s window

        let mut mid = PolySynth::new(SR);
        mid.note_on_bend(60.0, 1.0, &[1.0, 72.0], 0.5);
        let mut midbuf = vec![0.0f32; n];
        mid.process(&mut midbuf);

        let mut flat = PolySynth::new(SR);
        flat.note_on(60.0, 1.0); // reference voice held at the head pitch
        let mut flatbuf = vec![0.0f32; n];
        flat.process(&mut flatbuf);

        // Skip the attack transient, then compare fundamentals via zero crossings.
        let mid_zc = upward_crossings(&midbuf[n / 4..]);
        let flat_zc = upward_crossings(&flatbuf[n / 4..]);
        assert!(
            mid_zc > flat_zc,
            "mid-glide start should sound higher than the head pitch: {mid_zc} vs {flat_zc}"
        );
    }

    #[test]
    fn note_on_without_bend_holds_a_steady_pitch() {
        let mut p = PolySynth::new(SR);
        p.note_on(67.0, 1.0); // flat note → bend path inactive
        let n = SR as usize;
        let mut buf = vec![0.0f32; n];
        p.process(&mut buf);
        let early = upward_crossings(&buf[n / 10..2 * n / 10]);
        let late = upward_crossings(&buf[n - 2 * n / 10..n - n / 10]);
        // Same-width windows on a steady tone agree within a couple of cycles.
        assert!((early as i32 - late as i32).abs() <= 2, "flat pitch drifted: {early} vs {late}");
    }

    #[test]
    fn release_all_silences_everything() {
        let mut p = PolySynth::new(SR);
        p.note_on(60.0, 1.0);
        p.note_on(72.0, 1.0);
        p.release_all();
        let mut buf = vec![0.0f32; 4096];
        for _ in 0..6 {
            p.process(&mut buf);
        }
        assert_eq!(active(&p), 0);
    }

    fn render(setup: impl Fn(&mut PolySynth), midi: f32, n: usize) -> Vec<f32> {
        let mut p = PolySynth::new(SR);
        setup(&mut p);
        p.note_on(midi, 1.0);
        let mut buf = vec![0.0f32; n];
        p.process(&mut buf);
        buf
    }

    fn energy(buf: &[f32]) -> f32 {
        buf.iter().map(|x| x.abs()).sum()
    }

    #[test]
    fn octave_offset_raises_pitch() {
        let n = SR as usize;
        let lo = render(|_| {}, 60.0, n);
        let hi = render(|p| p.set_param("octave", 1.0), 60.0, n);
        // An octave up ≈ doubles the fundamental → ~twice the saw zero-crossings.
        assert!(
            upward_crossings(&hi) > upward_crossings(&lo),
            "octave offset should raise pitch: lo {} hi {}",
            upward_crossings(&lo),
            upward_crossings(&hi)
        );
    }

    #[test]
    fn detune_thickens_the_timbre() {
        let n = 4096;
        let single = render(|_| {}, 60.0, n);
        let unison = render(|p| p.set_param("detune", 20.0), 60.0, n);
        let diff: f32 = single.iter().zip(&unison).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-2, "detune should add a second, drifting oscillator: {diff}");
    }

    #[test]
    fn filter_envelope_opens_the_cutoff() {
        // Resonance 0 (no resonant boost to confound the comparison) and a base
        // cutoff below the note's fundamental, so amount 0 is heavily attenuated
        // and a full positive filter-env sweep opens it up to pass far more.
        let mk = |amount: f32| {
            render(
                |p| {
                    p.set_param("resonance", 0.0);
                    p.set_param("cutoff", 150.0);
                    p.set_param("fAttack", 0.0);
                    p.set_param("fDecay", 0.5);
                    p.set_param("fSustain", 1.0);
                    p.set_param("filterEnvAmount", amount);
                },
                60.0, // ~261 Hz fundamental, above the 150 Hz closed cutoff
                (SR / 2.0) as usize,
            )
        };
        assert!(
            energy(&mk(1.0)) > energy(&mk(0.0)) * 2.0,
            "filter envelope should open the filter: closed {} open {}",
            energy(&mk(0.0)),
            energy(&mk(1.0))
        );
    }

    #[test]
    fn drive_waveshapes_the_output() {
        let dry = render(|p| p.set_param("cutoff", 16000.0), 57.0, 2048);
        let wet = render(
            |p| {
                p.set_param("cutoff", 16000.0);
                p.set_param("drive", 1.0);
            },
            57.0,
            2048,
        );
        let diff: f32 = dry.iter().zip(&wet).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-2, "drive should waveshape the signal: {diff}");
    }

    #[test]
    fn lfo_tremolo_modulates_amplitude() {
        // Sine LFO → amplitude, fast enough to swing several times within 1s.
        let n = SR as usize;
        let buf = render(
            |p| {
                p.set_param("lfoTarget", 2.0); // amp
                p.set_param("lfoShape", 0.0); // sine
                p.set_param("lfoRate", 8.0);
                p.set_param("lfoDepth", 1.0);
            },
            60.0,
            n,
        );
        let win = n / 16;
        let energies: Vec<f32> =
            (0..16).map(|i| energy(&buf[i * win..(i + 1) * win])).collect();
        let max = energies.iter().cloned().fold(0.0f32, f32::max);
        let min = energies.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max > min * 1.5, "tremolo should swing amplitude: min {min} max {max}");
    }
}
