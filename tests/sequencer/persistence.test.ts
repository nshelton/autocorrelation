import { describe, it, expect, beforeEach } from "vitest";
import { saveState, loadState, clearState } from "../../src/sequencer/persistence";
import { demoProject } from "../../src/sequencer/model";

describe("persistence", () => {
  beforeEach(() => clearState());

  it("round-trips a project + selected track", () => {
    const p = demoProject();
    p.bpm = 137;
    p.tracks[0].instrument.engine = 1;
    p.tracks[0].instrument.params.cutoff = 1234;
    p.tracks[0].clips[0].notes.push({ start: 0, duration: 1, midi: 48, velocity: 0.5 });

    saveState(p, 0);
    const loaded = loadState();

    expect(loaded).not.toBeNull();
    expect(loaded!.project.bpm).toBe(137);
    expect(loaded!.project.tracks[0].instrument.engine).toBe(1);
    expect(loaded!.project.tracks[0].instrument.params.cutoff).toBe(1234);
    expect(loaded!.project.tracks[0].clips[0].notes).toHaveLength(
      p.tracks[0].clips[0].notes.length,
    );
    expect(loaded!.selectedTrack).toBe(0);
  });

  it("round-trips a per-track piano-roll view", () => {
    const p = demoProject();
    p.tracks[0].view = { start: 4, end: 12, center: 72, octaves: 3 };
    saveState(p, 0);
    const loaded = loadState();
    expect(loaded).not.toBeNull();
    expect(loaded!.project.tracks[0].view).toEqual({ start: 4, end: 12, center: 72, octaves: 3 });
    // A track without a saved view stays undefined (→ editor auto-fits).
    expect(loaded!.project.tracks[1].view).toBeUndefined();
  });

  it("returns null when nothing is stored", () => {
    expect(loadState()).toBeNull();
  });

  it("backfills loop-region fields for saves from before they existed", () => {
    const p = demoProject() as unknown as Record<string, unknown>;
    delete p.loopEnabled;
    delete p.loopRegionStart;
    delete p.loopRegionEnd;
    localStorage.setItem(
      "autocorrelation.project",
      JSON.stringify({ version: 1, selectedTrack: 0, project: p }),
    );
    const loaded = loadState();
    expect(loaded).not.toBeNull();
    expect(loaded!.project.loopEnabled).toBe(false);
    expect(loaded!.project.loopRegionStart).toBe(0);
    expect(loaded!.project.loopRegionEnd).toBe(loaded!.project.loopEnd);
  });

  it("rejects corrupt / wrong-shape data instead of throwing", () => {
    localStorage.setItem(
      "autocorrelation.project",
      JSON.stringify({ version: 1, project: { bpm: "nope" } }),
    );
    expect(loadState()).toBeNull();
  });
});
