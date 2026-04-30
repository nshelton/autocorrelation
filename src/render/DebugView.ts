import { Scene } from "three";
import { DebugGrid } from "./DebugGrid";
import { TimeSeriesRenderer, type LineLayout } from "./TimeSeriesRenderer";
import { TimeSeriesLineRenderer } from "./TimeSeriesLineRenderer";
import type { FeatureStore } from "../store/FeatureStore";
import type { ParamStore } from "../params/ParamStore";

export interface DebugViewDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
}

type LineStoreKey =
  | "waveform"
  | "bufferAcf"
  | "spectrum"
  | "rmsLow"
  | "rmsMid"
  | "rmsHigh"
  | "rms"
  | "onset"
  | "onsetAcf"
  | "onsetAcfEnhanced"
  | "tea";

interface LineSpec {
  key: LineStoreKey;
  color: number;
  layout?: LineLayout;
  position?: [number, number, number];
  scale?: [number, number, number];
  autoGain?: boolean;
}

const LINE_COLORS: readonly LineSpec[] = [
  { key: "waveform", color: 0x66ffcc },
  { key: "bufferAcf", color: 0xcc99ff },
  { key: "spectrum", color: 0xffaa66, layout: "log" },
  { key: "rmsLow", color: 0xaa2222, autoGain: true },
  { key: "rmsMid", color: 0x22aa22, autoGain: true },
  { key: "rmsHigh", color: 0x2222aa, autoGain: true },
  { key: "rms", color: 0xffffff, autoGain: true },
  { key: "onset", color: 0xff9966 },
  { key: "onsetAcf", color: 0x6666bb },
  { key: "onsetAcfEnhanced", color: 0xff99cc },
  { key: "tea", color: 0xffff66 },
];

export class DebugView {
  private lines = new Map<LineStoreKey, TimeSeriesRenderer>();
  private grid = new DebugGrid();

  constructor(private deps: DebugViewDeps) {
    this.deps.scene.add(this.grid.object3d);
    this.createLines();
  }

  update(): void {
    for (const line of this.lines.values()) line.update();
  }

  dispose(): void {
    for (const line of this.lines.values()) line.dispose();
    this.lines.clear();
  }

  private createLines(): void {
    const { store, scene, paramStore, audioContext } = this.deps;

    for (const spec of LINE_COLORS) {
      const line = new TimeSeriesLineRenderer({
        source: () => store.get(spec.key),
        color: spec.color,
        layout: spec.layout,
        autoGain: spec.autoGain
          ? {
              tauSecs: () => paramStore.get("dsp.autoGain"),
              dtSecs: () =>
                paramStore.get("dsp.hopSize") / audioContext.sampleRate,
            }
          : undefined,
      });
      if (spec.position) line.object3d.position.set(...spec.position);
      if (spec.scale) line.object3d.scale.set(...spec.scale);
      scene.add(line.object3d);
      this.lines.set(spec.key, line);
    }
  }
}
