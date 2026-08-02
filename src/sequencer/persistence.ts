//! Project persistence — serialize the whole `Project` (notes, per-track
//! instruments, clips, tempo, loop) plus the selected track to localStorage.
//! The project is plain JSON-serializable data, so this is just stringify /
//! parse with a version stamp and a shape guard so stale/corrupt data falls
//! back to the demo instead of crashing.

import { INSTRUMENT_DEFAULTS, type Project } from "./model";

const KEY = "autocorrelation.project";
const VERSION = 1;

interface Stored {
  version: number;
  selectedTrack: number;
  project: Project;
}

export function saveState(project: Project, selectedTrack: number): void {
  try {
    const data: Stored = { version: VERSION, selectedTrack, project };
    localStorage.setItem(KEY, JSON.stringify(data));
  } catch {
    /* quota exceeded / storage disabled — non-fatal */
  }
}

export function loadState(): { project: Project; selectedTrack: number } | null {
  let raw: string | null;
  try {
    raw = localStorage.getItem(KEY);
  } catch {
    return null;
  }
  if (!raw) return null;
  try {
    const data = JSON.parse(raw) as Partial<Stored>;
    if (data.version !== VERSION || !isProject(data.project)) return null;
    return {
      project: normalize(data.project),
      selectedTrack: Math.max(0, data.selectedTrack ?? 0),
    };
  } catch {
    return null;
  }
}

/** Backfill fields added after the initial schema so older saves load cleanly
 *  (cheaper and friendlier than a version bump that discards the saved set). */
function normalize(p: Project): Project {
  if (typeof p.loopEnabled !== "boolean") p.loopEnabled = false;
  if (typeof p.loopRegionStart !== "number") p.loopRegionStart = p.loopStart;
  if (typeof p.loopRegionEnd !== "number") p.loopRegionEnd = p.loopEnd;
  // Instruments saved before the drum machine existed have no `kind`; default
  // them to "synth" so they keep loading as the polyphonic synth they were.
  for (const t of p.tracks) {
    const inst = t.instrument;
    if (inst && inst.kind !== "drums" && inst.kind !== "synth") inst.kind = "synth";
    // Backfill synth params added after a save (osc tuning, filter env, LFO,
    // drive) so the panel shows real values and the full set gets pushed to the
    // worklet. Drums keep only their `gain` — don't pollute them with synth keys.
    if (inst && inst.kind === "synth") {
      for (const key in INSTRUMENT_DEFAULTS) {
        if (typeof inst.params[key] !== "number") inst.params[key] = INSTRUMENT_DEFAULTS[key];
      }
    }
  }
  // Tuning was added after the initial schema; default to equal temperament.
  if (!p.tuning || (p.tuning.mode !== "equal" && p.tuning.mode !== "just")) {
    p.tuning = { mode: "equal", root: 0 };
  }
  return p;
}

export function clearState(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* non-fatal */
  }
}

/** Shallow structural guard — enough to reject old/corrupt shapes safely. */
function isProject(p: unknown): p is Project {
  if (!p || typeof p !== "object") return false;
  const proj = p as Record<string, unknown>;
  if (typeof proj.bpm !== "number") return false;
  if (typeof proj.loopStart !== "number" || typeof proj.loopEnd !== "number") return false;
  if (!Array.isArray(proj.tracks)) return false;
  return proj.tracks.every((t) => {
    if (!t || typeof t !== "object") return false;
    const tr = t as Record<string, unknown>;
    const inst = tr.instrument as Record<string, unknown> | undefined;
    return (
      typeof tr.name === "string" &&
      !!inst &&
      typeof inst.engine === "number" &&
      !!inst.params &&
      Array.isArray(tr.clips) &&
      tr.clips.every((c) => {
        const clip = c as Record<string, unknown>;
        return !!clip && typeof clip.start === "number" && Array.isArray(clip.notes);
      })
    );
  });
}
