import { describe, expect, it } from "vitest";
import { DEFAULT_TUNING, tuneMidi, type Tuning } from "../../src/sequencer/tuning";
import {
  defaultDrumInstrument,
  defaultInstrument,
  flattenToSchedule,
  type Project,
} from "../../src/sequencer/model";

const just = (root: number): Tuning => ({ mode: "just", root });

describe("tuneMidi", () => {
  it("is the identity in equal temperament (and when undefined)", () => {
    expect(tuneMidi(64, DEFAULT_TUNING)).toBe(64);
    expect(tuneMidi(64, undefined)).toBe(64);
    expect(tuneMidi(64, { mode: "equal", root: 5 })).toBe(64);
  });

  it("leaves the root (and its octaves) untouched in just intonation", () => {
    expect(tuneMidi(60, just(0))).toBeCloseTo(60, 6); // C, root C
    expect(tuneMidi(72, just(0))).toBeCloseTo(72, 6);
    expect(tuneMidi(48, just(0))).toBeCloseTo(48, 6);
    expect(tuneMidi(62, just(2))).toBeCloseTo(62, 6); // D, root D
  });

  it("flattens the just major third by ~13.7 cents and barely moves the fifth", () => {
    // Root C: E (64) is the major third → −13.69¢; G (67) → +1.96¢.
    expect(tuneMidi(64, just(0))).toBeCloseTo(64 - 0.1369, 4);
    expect(tuneMidi(67, just(0))).toBeCloseTo(67 + 0.0196, 4);
    // All offsets stay within a quarter-tone.
    for (let m = 60; m < 72; m++) {
      expect(Math.abs(tuneMidi(m, just(0)) - m)).toBeLessThan(0.2);
    }
  });

  it("re-derives intervals when the root moves", () => {
    // With root D, C is a minor seventh below the tonic → pc 10 → −3.91¢.
    expect(tuneMidi(60, just(2))).toBeCloseTo(60 - 0.0391, 4);
    // The same C is the tonic when root is C → unchanged.
    expect(tuneMidi(60, just(0))).toBeCloseTo(60, 6);
  });
});

describe("flattenToSchedule tuning", () => {
  function project(tuning: Tuning): Project {
    return {
      bpm: 120,
      loopStart: 0,
      loopEnd: 4,
      loopEnabled: false,
      loopRegionStart: 0,
      loopRegionEnd: 4,
      tuning,
      tracks: [
        {
          id: "s",
          name: "Synth",
          color: "#7fd1ff",
          instrument: defaultInstrument(),
          clips: [{ id: "s1", start: 0, length: 4, notes: [{ start: 0, duration: 1, midi: 64, velocity: 1 }] }],
        },
        {
          id: "d",
          name: "Drums",
          color: "#ff9f7f",
          instrument: defaultDrumInstrument(),
          clips: [{ id: "d1", start: 0, length: 4, notes: [{ start: 0, duration: 0.1, midi: 38, velocity: 1 }] }],
        },
      ],
    };
  }

  it("tunes synth notes but leaves drum lane MIDI exact", () => {
    const sched = flattenToSchedule(project(just(0)));
    // stride-4 [beat, midi, vel, track]; collect note-on midis per track.
    const synthOn = [...Array(sched.length / 4)]
      .map((_, i) => sched.slice(i * 4, i * 4 + 4))
      .find((e) => e[3] === 0 && e[2] > 0)!;
    const drumOn = [...Array(sched.length / 4)]
      .map((_, i) => sched.slice(i * 4, i * 4 + 4))
      .find((e) => e[3] === 1 && e[2] > 0)!;
    expect(synthOn[1]).toBeCloseTo(64 - 0.1369, 3); // E tuned down
    expect(drumOn[1]).toBe(38); // snare lane untouched
  });

  it("leaves synth notes at integer MIDI in equal temperament", () => {
    const sched = flattenToSchedule(project(DEFAULT_TUNING));
    const synthOn = [...Array(sched.length / 4)]
      .map((_, i) => sched.slice(i * 4, i * 4 + 4))
      .find((e) => e[3] === 0 && e[2] > 0)!;
    expect(synthOn[1]).toBe(64);
  });
});
