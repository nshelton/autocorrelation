import { describe, expect, it } from "vitest";
import {
  DRUM_LANES,
  DRUM_STEPS,
  LEVEL_VELOCITY,
  beatToStep,
  stepDuration,
  velocityToLevel,
} from "../../src/sequencer/drumkit";
import { flattenToSchedule, makeDrumTrack, type Project } from "../../src/sequencer/model";

describe("drumkit step math", () => {
  it("derives 1/16-note steps for the default 4-beat loop", () => {
    expect(stepDuration(0, 4)).toBeCloseTo(0.25);
    expect(stepDuration(0, 8)).toBeCloseTo(0.5); // grid auto-fits a longer loop
  });

  it("maps note beats onto step indices and rejects off-grid / out-of-range", () => {
    const sb = stepDuration(0, 4);
    expect(beatToStep(0, 0, sb)).toBe(0);
    expect(beatToStep(2, 0, sb)).toBe(8); // beat 2 = step 8 at 1/16
    expect(beatToStep(4, 0, sb)).toBeNull(); // == DRUM_STEPS, past the last step
    expect(beatToStep(-1, 0, sb)).toBeNull();
  });

  it("round-trips velocity through the H/M/S levels", () => {
    for (const level of ["soft", "med", "hard"] as const) {
      expect(velocityToLevel(LEVEL_VELOCITY[level])).toBe(level);
    }
  });
});

describe("drum track", () => {
  it("uses lane MIDI numbers that all resolve to a known lane", () => {
    const track = makeDrumTrack(1);
    expect(track.instrument.kind).toBe("drums");
    const laneMidis = new Set(DRUM_LANES.map((l) => l.midi));
    for (const note of track.clips[0].notes) {
      expect(laneMidis.has(note.midi)).toBe(true);
    }
  });

  it("flattens a drum clip into a routed on/off schedule", () => {
    const track = makeDrumTrack(0);
    const project: Project = {
      bpm: 120,
      loopStart: 0,
      loopEnd: 4,
      loopEnabled: false,
      loopRegionStart: 0,
      loopRegionEnd: 4,
      tracks: [track],
    };
    const sched = flattenToSchedule(project);
    expect(sched.length % 4).toBe(0); // stride-4 [beat, midi, vel, track]
    // Two events (on + off) per note, all routed to track 0.
    expect(sched.length / 4).toBe(track.clips[0].notes.length * 2);
    for (let i = 0; i < sched.length; i += 4) {
      expect(sched[i + 3]).toBe(0); // track index
    }
    // Every step lands within the pattern window.
    const sb = stepDuration(0, 4);
    for (const note of track.clips[0].notes) {
      expect(beatToStep(note.start, 0, sb)).not.toBeNull();
    }
    expect(DRUM_STEPS).toBe(16);
  });
});
