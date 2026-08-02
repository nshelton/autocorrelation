import { describe, it, expect } from "vitest";
import {
  flattenToSchedule,
  demoProject,
  defaultInstrument,
  type Note,
  type Project,
} from "../../src/sequencer/model";

const single = (notes: Note[]): Project => ({
  bpm: 120,
  loopStart: 0,
  loopEnd: 4,
  loopEnabled: false,
  loopRegionStart: 0,
  loopRegionEnd: 4,
  tracks: [
    {
      id: "a",
      name: "a",
      color: "#fff",
      instrument: defaultInstrument(),
      clips: [{ id: "c", start: 0, length: 4, notes }],
    },
  ],
});

describe("flattenToSchedule", () => {
  it("emits a note-on and note-off per note, stride 4 with track index", () => {
    // velocity 0.5 is exact in f32, so the round-trip compares cleanly.
    const s = flattenToSchedule(
      single([{ start: 1, duration: 0.5, midi: 60, velocity: 0.5 }]),
    );
    expect(s.length).toBe(8);
    expect(Array.from(s.slice(0, 4))).toEqual([1, 60, 0.5, 0]); // on, track 0
    expect(Array.from(s.slice(4, 8))).toEqual([1.5, 60, 0, 0]); // off
  });

  it("offsets note positions by their clip start", () => {
    const p: Project = {
      bpm: 120,
      loopStart: 0,
      loopEnd: 8,
      loopEnabled: false,
      loopRegionStart: 0,
      loopRegionEnd: 8,
      tracks: [
        {
          id: "a",
          name: "a",
          color: "#fff",
          instrument: defaultInstrument(),
          clips: [
            { id: "c", start: 2, length: 4, notes: [{ start: 1, duration: 1, midi: 60, velocity: 1 }] },
          ],
        },
      ],
    };
    expect(flattenToSchedule(p)[0]).toBe(3); // clip.start 2 + note.start 1
  });

  it("tags each event with its track index", () => {
    const track = (id: string, midi: number) => ({
      id,
      name: id,
      color: "#fff",
      instrument: defaultInstrument(),
      clips: [{ id: `${id}-c`, start: 0, length: 4, notes: [{ start: 0, duration: 1, midi, velocity: 1 }] }],
    });
    const p: Project = {
      bpm: 120,
      loopStart: 0,
      loopEnd: 4,
      loopEnabled: false,
      loopRegionStart: 0,
      loopRegionEnd: 4,
      tracks: [track("a", 60), track("b", 64)],
    };
    const s = flattenToSchedule(p);
    const onTracks = new Set<number>();
    for (let i = 0; i < s.length; i += 4) if (s[i + 2] > 0) onTracks.add(s[i + 3]);
    expect(onTracks).toEqual(new Set([0, 1]));
  });

  it("sorts by beat, with note-off before note-on on a tie", () => {
    const s = flattenToSchedule(
      single([
        { start: 1, duration: 1, midi: 62, velocity: 1 }, // off at beat 2
        { start: 2, duration: 1, midi: 64, velocity: 1 }, // on at beat 2
      ]),
    );
    expect([s[0], s[4], s[8], s[12]]).toEqual([1, 2, 2, 3]);
    expect(s[6]).toBe(0); // beat-2 tie: off (vel 0) first
    expect(s[10]).toBe(1); // then on (vel 1)
  });

  it("demoProject yields a non-empty, well-formed schedule", () => {
    const s = flattenToSchedule(demoProject());
    expect(s.length).toBeGreaterThan(0);
    expect(s.length % 4).toBe(0);
  });
});
