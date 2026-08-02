import { describe, it, expect } from "vitest";
import { Midi } from "@tonejs/midi";
import { parseMidi } from "../../src/sequencer/midi";

// Build a MIDI in memory, encode it, and parse it back — exercises the real
// @tonejs/midi codec the importer depends on, no fixture file needed.
describe("parseMidi", () => {
  it("round-trips ticks→beats, tempo, tracks, and loop length", () => {
    const midi = new Midi();
    midi.header.setTempo(140);
    const ppq = midi.header.ppq || 480;
    const track = midi.addTrack();
    track.name = "Bass";
    track.addNote({ midi: 60, ticks: 0, durationTicks: ppq }); // beat 0, 1 beat
    track.addNote({ midi: 67, ticks: ppq * 2, durationTicks: ppq / 2 }); // beat 2, 0.5

    const project = parseMidi(midi.toArray().buffer as ArrayBuffer);

    expect(project.bpm).toBeCloseTo(140, 0);
    expect(project.tracks).toHaveLength(1);
    expect(project.tracks[0].name).toBe("Bass");

    const notes = project.tracks[0].clips[0].notes;
    expect(notes).toHaveLength(2);
    expect(notes[0].start).toBeCloseTo(0, 4);
    expect(notes[0].duration).toBeCloseTo(1, 4);
    expect(notes[0].midi).toBe(60);
    expect(notes[1].start).toBeCloseTo(2, 4);
    expect(notes[1].duration).toBeCloseTo(0.5, 4);

    // Last note ends at beat 2.5 → loopEnd rounds up to 3.
    expect(project.loopEnd).toBe(3);
  });

  it("skips empty tracks and throws when there is no note data", () => {
    const midi = new Midi();
    midi.addTrack(); // control-only / empty
    expect(() => parseMidi(midi.toArray().buffer as ArrayBuffer)).toThrow(
      /no note data/,
    );
  });
});
