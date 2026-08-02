//! Tuning — how a note's MIDI number becomes a pitch. The DSP plays whatever
//! fractional MIDI it's handed (`midi_to_hz` in synth.rs is pure 12-TET), so a
//! tuning system is just a remap applied to the base pitch when a project is
//! flattened to the schedule (and to live keys): `effectiveMidi = midi + offset`.
//!
//! **Equal**: the identity — standard 12-tone equal temperament (12th root of 2),
//! where every key is equally, slightly out of tune.
//!
//! **Just**: 5-limit just intonation relative to a movable **root** (tonic).
//! Each pitch class gets a fixed cents offset from ET so intervals from the root
//! land on exact small-integer ratios (pure, beat-less chords on the tonic).
//! Changing `root` shifts which pitch class is 1/1 and re-derives every interval.
//! Caveat (inherent to fixed-root JI): one or two chords in the key contain a
//! "wolf" interval — e.g. in C the D–A fifth is ~22¢ flat — which is expected,
//! not a bug. Drum tracks are never tuned (their MIDI is a lane index).

export type TuningMode = "equal" | "just";

export interface Tuning {
  mode: TuningMode;
  /** Tonic pitch class, 0..11 (0 = C). Only meaningful when mode === "just". */
  root: number;
}

export const DEFAULT_TUNING: Tuning = { mode: "equal", root: 0 };

export const PITCH_CLASS_NAMES = [
  "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/**
 * 5-limit just-intonation cents deviation from equal temperament, indexed by
 * `(midi - root) mod 12`. Ratios (octave-complementary set):
 *   1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 16/9, 15/8.
 * The fifth is +2¢ (≈ pure); the major third is −14¢ (the audible win); the
 * tritone/sevenths are the most approximate (the usual JI ambiguities).
 */
const JI_CENTS = [
  0.0, 11.73, 3.91, 15.64, -13.69, -1.96, -9.78, 1.96, 13.69, -15.64, -3.91, -11.73,
];

/**
 * Remap a base MIDI note to its tuned (fractional) MIDI. `equal` (or undefined)
 * is the identity. For `just`, add the pitch class's cents offset (÷100 →
 * semitones) relative to the tuning root. Caller skips drum tracks.
 */
export function tuneMidi(midi: number, tuning: Tuning | undefined): number {
  if (!tuning || tuning.mode !== "just") return midi;
  const pc = (((Math.round(midi) - tuning.root) % 12) + 12) % 12;
  return midi + JI_CENTS[pc] / 100;
}
