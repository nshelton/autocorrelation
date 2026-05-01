import { Scene, Object3D } from "three";
import { DebugGrid } from "./DebugGrid";
import { BeatGridMarkers } from "./BeatGridMarkers";
import { DebugLabels } from "./DebugLabels";
import { PeakMarkers } from "./PeakMarkers";
import { TimeSeriesRenderer, type TimeSeriesScale } from "./TimeSeriesRenderer";
import { TimeSeriesBarRenderer } from "./TimeSeriesBarRenderer";
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
  autoGain?: boolean;
  type?: "line" | "bar";
  scale?: TimeSeriesScale;
}

const LINE_COLORS: readonly LineSpec[] = [
  { key: "waveform", color: 0x66ffcc },
  { key: "bufferAcf", color: 0xcc99ff },
  { key: "spectrum", color: 0xffaa66, type: "bar", scale: "logx" },
  { key: "rmsLow", color: 0xaa0000, type: "bar", autoGain: true },
  { key: "rmsMid", color: 0x00aa00, type: "bar", autoGain: true },
  { key: "rmsHigh", color: 0x0000aa, type: "bar", autoGain: true },
  { key: "rms", color: 0xffffff, autoGain: true },
  { key: "onset", type: "bar", color: 0xff9966, autoGain: true },
  { key: "onsetAcf", color: 0x6666bb },
  { key: "onsetAcfEnhanced", color: 0xff99cc },
  { key: "tea", color: 0xffff66 },
];

export class DebugView {
  private lines = new Map<LineStoreKey, TimeSeriesRenderer>();
  private backgroundGrid = new DebugGrid();
  private peakMarkers: PeakMarkers;
  private beatGridMarkers: BeatGridMarkers;
  private labels: DebugLabels;

  constructor(private deps: DebugViewDeps) {
    this.peakMarkers = new PeakMarkers({ baseColor: 0xffff66 });
    this.beatGridMarkers = new BeatGridMarkers({ baseColor: 0x66ccff });
    this.labels = new DebugLabels(deps);
    this.deps.scene.add(this.backgroundGrid.object3d);
    this.deps.scene.add(this.peakMarkers.object3d);
    this.deps.scene.add(this.labels.object3d);
    this.createLines();
  }

  update(): void {
    for (const line of this.lines.values()) line.update();
    this.peakMarkers.update(
      this.deps.store.get("candidates"),
      this.deps.store.get("onsetAcf").length,
    );
    this.beatGridMarkers.update(
      this.deps.store.get("beatGrid"),
      this.deps.store.get("rms").length,
    );
    this.labels.update();
  }

  dispose(): void {
    for (const line of this.lines.values()) line.dispose();
    this.lines.clear();
    this.peakMarkers.dispose();
    this.beatGridMarkers.dispose();
    this.labels.dispose();
  }

  private createLines(): void {
    const { store, scene, paramStore, audioContext } = this.deps;

    for (const spec of LINE_COLORS) {
      const options = {
        source: () => store.get(spec.key),
        color: spec.color,
        scale: spec.scale,
        autoGain: spec.autoGain
          ? {
              tauSecs: () => paramStore.get("dsp.autoGain"),
              dtSecs: () =>
                paramStore.get("dsp.hopSize") / audioContext.sampleRate,
            }
          : undefined,
      };
      const line =
        spec.type === "bar"
          ? new TimeSeriesBarRenderer(options)
          : new TimeSeriesLineRenderer(options);
      scene.add(line.object3d);

      this.lines.set(spec.key, line);
    }

    this.lines.get("waveform")!.object3d.position.set(-2, 1.5, 0);
    this.lines.get("bufferAcf")!.object3d.position.set(-1, 1.5, 0);
    this.lines.get("bufferAcf")!.object3d.scale.set(1, 0.5, 1);
    this.lines.get("spectrum")!.object3d.position.set(0, 1, 0);
    this.lines.get("spectrum")!.object3d.scale.set(2, 1, 1);

    scene.remove(
      this.lines.get("rmsLow")!.object3d,
      this.lines.get("rmsMid")!.object3d,
      this.lines.get("rmsHigh")!.object3d,
      this.lines.get("rms")!.object3d,
    );
    const group = new Object3D();
    scene.add(group);
    group.add(
      this.lines.get("rmsLow")!.object3d,
      this.lines.get("rmsMid")!.object3d,
      this.lines.get("rmsHigh")!.object3d,
      this.lines.get("rms")!.object3d,
      this.beatGridMarkers.object3d,
    );

    group.position.set(-2, 0, 0);
    group.scale.set(4, 0.5, 1);

    this.lines.get("onset")!.object3d.position.set(-2, -0.5, 0);
    this.lines.get("onset")!.object3d.scale.set(4, 0.5, 1);

    this.lines.get("onsetAcf")!.object3d.position.set(-2, -2, 0);
    this.lines.get("onsetAcf")!.object3d.scale.set(4, 5, 1);

    this.lines.get("onsetAcfEnhanced")!.object3d.position.set(-2, -2, 0);
    this.lines.get("onsetAcfEnhanced")!.object3d.scale.set(4, 2, 1);
    this.peakMarkers.object3d.position.set(-2, -2, 0);
    this.peakMarkers.object3d.scale.set(4, 1, 1);

    this.lines.get("tea")!.object3d.position.set(-2, -2, 0);
    this.lines.get("tea")!.object3d.scale.set(4, 2, 1);
  }
}
