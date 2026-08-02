//! MIDI file import. Parses a `.mid` with `@tonejs/midi` and maps it into our
//! `Project` model. Note positions are converted ticks → **beats**
//! (`ticks / ppq`) so they're tempo-independent and land directly in the
//! Sequencer's beat domain — no seconds round-trip.
//!
//! Limitations (refinements for later phases):
//!   - only the first tempo is used; tempo-map changes mid-song are ignored.
//!   - the synth is still monophonic, so a multi-track file plays last-note-
//!     wins. All tracks are kept for the piano-roll view regardless.

import { Midi } from "@tonejs/midi";
import { defaultInstrument, trackColor, type Note, type Project, type Track } from "./model";

export function parseMidi(data: ArrayBuffer, name = "imported"): Project {
  const midi = new Midi(data);
  const ppq = midi.header.ppq || 480; // ticks per quarter note = ticks per beat
  const bpm = midi.header.tempos[0]?.bpm ?? 120;

  const tracks: Track[] = [];
  let maxEndBeat = 0;

  midi.tracks.forEach((t, i) => {
    if (t.notes.length === 0) return; // skip control-only / empty tracks
    const notes: Note[] = t.notes.map((n) => {
      const start = n.ticks / ppq;
      const duration = n.durationTicks / ppq;
      if (start + duration > maxEndBeat) maxEndBeat = start + duration;
      return { start, duration, midi: n.midi, velocity: n.velocity };
    });
    // One clip per MIDI track for now; the arrangement view can split later.
    const length = notes.reduce((m, n) => Math.max(m, n.start + n.duration), 1);
    tracks.push({
      id: `t${i}`,
      name: t.name || t.instrument?.name || `Track ${i + 1}`,
      color: trackColor(tracks.length),
      instrument: defaultInstrument(),
      clips: [{ id: `t${i}-clip`, start: 0, length, notes }],
    });
  });

  if (tracks.length === 0) {
    throw new Error(`${name} contains no note data`);
  }

  const loopEnd = Math.max(1, Math.ceil(maxEndBeat)); // round up past the last note
  return {
    bpm,
    loopStart: 0,
    loopEnd,
    loopEnabled: false,
    loopRegionStart: 0,
    loopRegionEnd: loopEnd,
    tracks,
  };
}
