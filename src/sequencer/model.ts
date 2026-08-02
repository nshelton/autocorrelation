//! Sequencer data model — the main-thread source of truth. A project is a set
//! of tracks; each track owns an instrument (its synth state) and a list of
//! clips placed on the arrangement timeline; each clip holds notes positioned
//! relative to the clip start. Flattened to the stride-4 `[beat, midi,
//! velocity, track]` schedule the Rust `Sequencer` consumes (one instrument per
//! track). Times are in **beats** to match the Sequencer's beat-domain playhead.
//!
//! Pitches are remapped through the project `tuning` at flatten time (see
//! `tuning.ts`): the editor stores plain integer semitones, but the *sounding*
//! pitch can be just-intonation tuned. The DSP plays fractional MIDI either way.

import { DEFAULT_TUNING, tuneMidi, type Tuning } from "./tuning";

/** One pitch-bend breakpoint on a note's pitch envelope. `t` is beats from the
 *  note's start; `offset` is semitones from the note's base `midi` (so a bend
 *  transposes/moves rigidly with its note). The note has an implicit head point
 *  at `(t: 0, offset: 0)`; breakpoints are sorted by `t` in `(0, duration]` and
 *  the last value is held to the note's end. Linear interpolation between
 *  points = a glide. Synth tracks only — drum `midi` is a lane index. */
export interface BendPoint {
  t: number;
  offset: number;
}

export interface Note {
  /** Start in beats, relative to the containing clip. */
  start: number;
  /** Length in beats. */
  duration: number;
  /** MIDI note number (the base pitch; a `bend` envelope, if any, is relative). */
  midi: number;
  /** 0..1. */
  velocity: number;
  /** Optional pitch-bend envelope. Absent/empty = flat at `midi`. */
  bend?: BendPoint[];
}

export interface Clip {
  id: string;
  /** Placement on the arrangement timeline, in beats. */
  start: number;
  /** Clip length in beats (its block bounds on the timeline). */
  length: number;
  /** Notes, positioned relative to `start`. */
  notes: Note[];
}

export interface Instrument {
  /** Which Rust instrument this track drives: a polyphonic synth (piano-roll
   *  melodic content) or a drum kit (step-grid percussion). Defaults to "synth"
   *  for older saves; see `persistence.normalize`. */
  kind?: "synth" | "drums";
  /** Oscillator engine index (0 = subtractive, 1 = simplex). Synth only. */
  engine: number;
  /** Synth params keyed by the Rust `set_param` keys (drums use { gain }). */
  params: Record<string, number>;
}

/** Persisted piano-roll view for a track: the time window `[start, end]` (beats)
 *  and the pitch view (`center` MIDI + `octaves` count). Restored when the track's
 *  clip is opened; absent → the editor auto-fits to the track's content. */
export interface PianoRollView {
  start: number;
  end: number;
  center: number;
  octaves: number;
}

export interface Track {
  id: string;
  name: string;
  /** CSS color used by the piano-roll and track UI. */
  color: string;
  instrument: Instrument;
  clips: Clip[];
  /** Last piano-roll zoom/scroll for this track (UI state; optional). */
  view?: PianoRollView;
}

export interface Project {
  bpm: number;
  /** View window start (beats) — currently always 0. Also the base loop start. */
  loopStart: number;
  /** View window length (beats), set by the 4/8/16/32 buttons. Also the base
   *  loop end used when the loop region is disabled. */
  loopEnd: number;
  /** Gold loop toggle: when true, playback loops [loopRegionStart, loopRegionEnd]
   *  instead of the whole view. */
  loopEnabled: boolean;
  loopRegionStart: number; // beats
  loopRegionEnd: number; // beats
  tracks: Track[];
  /** How pitches are tuned (equal temperament vs just intonation from a root).
   *  Optional for back-compat; `persistence.normalize` backfills the default. */
  tuning?: Tuning;
}

/** Default synth params — the single source of truth, shared with the panel.
 *  Values mirror the Rust `Voice` defaults so UI and DSP start in agreement.
 *  The added oscillator/filter-env/LFO/drive params default to "inert" values
 *  (offsets/amounts/depths/drive = 0), so a fresh instrument sounds exactly like
 *  the original saw → filter → ADSR voice until the new controls are touched. */
export const INSTRUMENT_DEFAULTS: Record<string, number> = {
  // Oscillator tuning / unison.
  octave: 0,
  semi: 0,
  fine: 0,
  detune: 0,
  // Filter.
  cutoff: 4000,
  resonance: 0.7,
  // Filter envelope (its own ADSR + signed octave amount; inert at amount 0).
  filterEnvAmount: 0,
  fAttack: 0.005,
  fDecay: 0.2,
  fSustain: 0,
  fRelease: 0.2,
  // Amp envelope.
  attack: 0.005,
  decay: 0.1,
  sustain: 0.7,
  release: 0.2,
  // LFO (target: 0=pitch, 1=cutoff, 2=amp; shape: 0=sine,1=tri,2=square,3=saw).
  lfoRate: 5,
  lfoDepth: 0,
  lfoTarget: 0,
  lfoShape: 0,
  // Drive (mode: 0=pre-filter, 1=post-filter).
  drive: 0,
  driveMode: 0,
  // Output.
  gain: 0.25,
};

export function defaultInstrument(): Instrument {
  return { kind: "synth", engine: 0, params: { ...INSTRUMENT_DEFAULTS } };
}

/** A drum-kit instrument. Drums synthesize their own voices (one per lane), so
 *  the only param is a master `gain`; engine is unused (kept 0 for shape). */
export function defaultDrumInstrument(): Instrument {
  return { kind: "drums", engine: 0, params: { gain: 0.9 } };
}

const TRACK_COLORS = [
  "#7fd1ff",
  "#ff9f7f",
  "#9fff7f",
  "#ff7fd1",
  "#d1ff7f",
  "#7f9fff",
  "#ffd17f",
  "#7fffd1",
];

export function trackColor(index: number): string {
  return TRACK_COLORS[index % TRACK_COLORS.length];
}

/** A new empty synth track at `index` (one empty clip over a 4-beat bar). */
export function makeSynthTrack(index: number): Track {
  return {
    id: `synth-${index}`,
    name: `Synth ${index + 1}`,
    color: trackColor(index),
    instrument: defaultInstrument(),
    clips: [{ id: `synth-${index}-clip`, start: 0, length: 4, notes: [] }],
  };
}

/** A new drum track at `index`, pre-loaded with a basic starter beat so it
 *  makes sound immediately. MIDI numbers reference the lanes in `drumkit.ts`. */
export function makeDrumTrack(index: number): Track {
  return {
    id: `drums-${index}`,
    name: "Drums",
    color: trackColor(index),
    instrument: defaultDrumInstrument(),
    clips: [{ id: `drums-${index}-clip`, start: 0, length: 4, notes: drumStarterNotes() }],
  };
}

/** Basic four-on-the-floor-ish starter: kick on the downbeats, snare on 2 & 4,
 *  closed hat on every eighth. Step grid is 16 steps over a 4-beat bar. */
function drumStarterNotes(): Note[] {
  const step = 0.25; // 1/16 note (4 beats / 16 steps)
  const hit = (s: number, midi: number, velocity: number): Note => ({
    start: s * step,
    duration: step * 0.5,
    midi,
    velocity,
  });
  const notes: Note[] = [];
  for (const s of [0, 8]) notes.push(hit(s, 36, 1.0)); // Kick on beats 1 & 3
  for (const s of [4, 12]) notes.push(hit(s, 38, 1.0)); // Snare on beats 2 & 4
  for (let s = 0; s < 16; s += 2) notes.push(hit(s, 42, 0.66)); // Closed hat eighths
  return notes;
}

/**
 * Flatten every track's clips into the sorted on/off event stream the Rust
 * `Sequencer` expects: stride-4 `[beat, midi, velocity, track]`, velocity 0 =
 * note-off. Note positions are offset by their clip's `start` to get absolute
 * arrangement beats.
 */
export function flattenToSchedule(project: Project): Float32Array {
  const events: Array<[number, number, number, number]> = [];
  project.tracks.forEach((track, ti) => {
    // Drum MIDI is a lane index, not a pitch — never retune it.
    const isDrum = track.instrument.kind === "drums";
    for (const clip of track.clips) {
      for (const note of clip.notes) {
        const start = clip.start + note.start;
        const midi = isDrum ? note.midi : tuneMidi(note.midi, project.tuning);
        events.push([start, midi, note.velocity, ti]);
        events.push([start + note.duration, midi, 0, ti]);
      }
    }
  });
  // Sort by beat; on a tie, note-off (vel 0) before note-on so a re-struck
  // pitch releases the old voice before the new one starts.
  events.sort((a, b) => a[0] - b[0] || a[2] - b[2]);

  const out = new Float32Array(events.length * 4);
  for (let i = 0; i < events.length; i++) {
    out[i * 4] = events[i][0];
    out[i * 4 + 1] = events[i][1];
    out[i * 4 + 2] = events[i][2];
    out[i * 4 + 3] = events[i][3];
  }
  return out;
}

/**
 * Flatten every note's pitch-bend envelope into the self-describing payload the
 * Rust `Sequencer.set_bends` consumes — a sequence of records
 * `[headBeat, headMidi, track, n, (t, midi)×n]`, where the head identifies the
 * note-on (absolute beat, base midi, track) and each breakpoint carries its beat
 * offset from the head and its **absolute** midi (base + offset). Notes with no
 * bend, and drum tracks (whose `midi` is a lane, not a pitch), are skipped.
 */
export function flattenBends(project: Project): Float32Array {
  const out: number[] = [];
  project.tracks.forEach((track, ti) => {
    if (track.instrument.kind === "drums") return; // drums are unpitched
    for (const clip of track.clips) {
      for (const note of clip.notes) {
        const bend = note.bend;
        if (!bend || bend.length === 0) continue;
        // Tune the base pitch (consistent with the schedule); bend breakpoints
        // ride as ET-semitone offsets from the tuned base.
        const base = tuneMidi(note.midi, project.tuning);
        out.push(clip.start + note.start, base, ti, bend.length);
        for (const p of bend) out.push(p.t, base + p.offset);
      }
    }
  });
  return new Float32Array(out);
}

/**
 * A small looping riff on a single track — proves the transport + piano-roll
 * end to end. Replaced by authored / MIDI-imported content.
 */
export function demoProject(): Project {
  const RIFF = [0, 3, 5, 7, 10, 7, 5, 3]; // semitone offsets, minor-ish
  const STEP = 0.5; // eighth notes
  const GATE = 0.9; // fraction of a step the note is held
  const ROOT = 60; // C4
  const notes: Note[] = RIFF.map((semi, i) => ({
    start: i * STEP,
    duration: STEP * GATE,
    midi: ROOT + semi,
    velocity: 0.8,
  }));
  const length = RIFF.length * STEP; // 4 beats
  return {
    bpm: 120,
    loopStart: 0,
    loopEnd: length,
    loopEnabled: false,
    loopRegionStart: 0,
    loopRegionEnd: length,
    tuning: { ...DEFAULT_TUNING },
    tracks: [
      {
        id: "lead",
        name: "Lead",
        color: trackColor(0),
        instrument: defaultInstrument(),
        clips: [{ id: "lead-1", start: 0, length, notes }],
      },
      // A drum track so the step grid is reachable on a fresh load (and audible).
      makeDrumTrack(1),
    ],
  };
}
