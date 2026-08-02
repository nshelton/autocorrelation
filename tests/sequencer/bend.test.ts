import { describe, it, expect } from "vitest";
import {
  flattenBends,
  defaultDrumInstrument,
  defaultInstrument,
  type Note,
  type Project,
} from "../../src/sequencer/model";

const project = (notes: Note[], clipStart = 0): Project => ({
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
      clips: [{ id: "c", start: clipStart, length: 4, notes }],
    },
  ],
});

describe("flattenBends", () => {
  it("emits nothing for notes without a bend", () => {
    const out = flattenBends(project([{ start: 0, duration: 1, midi: 60, velocity: 1 }]));
    expect(out.length).toBe(0);
  });

  it("encodes a record [headBeat, headMidi, track, n, (t, absMidi)...] with offsets resolved to absolute midi", () => {
    const out = flattenBends(
      project([
        {
          start: 1,
          duration: 2,
          midi: 60,
          velocity: 1,
          // glide up 4 semitones at 1 beat in, then down 3 from base at 2 beats.
          bend: [
            { t: 1, offset: 4 },
            { t: 2, offset: -3 },
          ],
        },
      ]),
    );
    // head: beat 1, midi 60, track 0, 2 points; then (1, 64), (2, 57).
    expect(Array.from(out)).toEqual([1, 60, 0, 2, 1, 64, 2, 57]);
  });

  it("offsets the head beat by the clip start", () => {
    const out = flattenBends(
      project([{ start: 1, duration: 1, midi: 60, velocity: 1, bend: [{ t: 0.5, offset: 2 }] }], 2),
    );
    expect(out[0]).toBe(3); // clip.start 2 + note.start 1
    expect(out[5]).toBe(62); // base 60 + offset 2
  });

  it("skips drum tracks (their midi is a lane, not a pitch)", () => {
    const p = project([{ start: 0, duration: 1, midi: 36, velocity: 1, bend: [{ t: 0.5, offset: 5 }] }]);
    p.tracks[0].instrument = defaultDrumInstrument();
    expect(flattenBends(p).length).toBe(0);
  });

  it("tags each record with its track index", () => {
    const mk = (id: string, midi: number): Project["tracks"][number] => ({
      id,
      name: id,
      color: "#fff",
      instrument: defaultInstrument(),
      clips: [
        {
          id: `${id}-c`,
          start: 0,
          length: 4,
          notes: [{ start: 0, duration: 1, midi, velocity: 1, bend: [{ t: 0.5, offset: 1 }] }],
        },
      ],
    });
    const p: Project = {
      bpm: 120,
      loopStart: 0,
      loopEnd: 4,
      loopEnabled: false,
      loopRegionStart: 0,
      loopRegionEnd: 4,
      tracks: [mk("a", 60), mk("b", 67)],
    };
    const out = flattenBends(p);
    // Two records of length 6 each: [beat, midi, track, 1, t, absMidi].
    expect(out.length).toBe(12);
    expect(out[2]).toBe(0); // first record track 0
    expect(out[8]).toBe(1); // second record track 1
  });
});
