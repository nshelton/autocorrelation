//! Canonical drum-kit definition — the single source of truth for the drum
//! machine's lanes and the H/M/S velocity scheme. Shared between the
//! `DrumMachine` UI (which authors notes) and `model.ts` (starter patterns).
//!
//! The `midi` numbers here MUST agree with `crates/dsp/src/drums.rs`: the grid
//! emits a note at a lane's MIDI number, and the Rust `DrumKit` routes that note
//! back to the matching voice. This is the drum analogue of how
//! `INSTRUMENT_DEFAULTS` mirrors the Rust `Voice` defaults.

export interface DrumLane {
  /** Display name in the grid's row label. */
  name: string;
  /** MIDI note routed to this lane (matches the Rust `DrumKit` spec). */
  midi: number;
}

/** Lane order = top-to-bottom row order in the grid; matches `LANES` in drums.rs. */
export const DRUM_LANES: DrumLane[] = [
  { name: "Kick", midi: 36 },
  { name: "Snare", midi: 38 },
  { name: "Closed Hat", midi: 42 },
  { name: "Open Hat", midi: 46 },
  { name: "Clap", midi: 39 },
  { name: "Rim", midi: 37 },
  { name: "Tom Lo", midi: 41 },
  { name: "Tom Mid", midi: 45 },
  { name: "Tom Hi", midi: 48 },
  { name: "Cymbal", midi: 49 },
];

/** A pattern is this many steps wide (one bar of 1/16 notes at 4 beats). */
export const DRUM_STEPS = 16;

/** The three ReDrum-style per-step dynamics. */
export type Level = "soft" | "med" | "hard";

/** Velocity each level writes into a note (and the brightness the grid shows). */
export const LEVEL_VELOCITY: Record<Level, number> = {
  soft: 0.33,
  med: 0.66,
  hard: 1.0,
};

/** Bucket a note's stored velocity back into one of the three levels (for
 *  displaying / re-toggling an existing step). Thresholds sit between the three
 *  `LEVEL_VELOCITY` values. */
export function velocityToLevel(velocity: number): Level {
  if (velocity >= 0.83) return "hard";
  if (velocity >= 0.5) return "med";
  return "soft";
}

/** Beats per step for a pattern spanning `[loopStart, loopEnd]`. The grid auto-
 *  fits the loop window, so the default 4-beat loop yields 1/16-note steps. */
export function stepDuration(loopStart: number, loopEnd: number, steps = DRUM_STEPS): number {
  return (loopEnd - loopStart) / steps;
}

/** Step index a note at absolute `beat` lands on, or null if it doesn't align
 *  with the grid (e.g. a note authored elsewhere at an off-grid position). */
export function beatToStep(
  beat: number,
  loopStart: number,
  stepBeats: number,
  steps = DRUM_STEPS,
): number | null {
  if (stepBeats <= 0) return null;
  const i = Math.round((beat - loopStart) / stepBeats);
  return i >= 0 && i < steps ? i : null;
}
