import { Scene, Object3D } from "three";
import { DebugGrid } from "./DebugGrid";
import { BeatGridMarkers } from "./BeatGridMarkers";
import { StaticBeatGridMarkers } from "./StaticBeatGridMarkers";
import { DebugLabels } from "./DebugLabels";
import { TimeSeriesRenderer, type TimeSeriesScale } from "./TimeSeriesRenderer";
import { TimeSeriesBarRenderer } from "./TimeSeriesBarRenderer";
import { TimeSeriesLineRenderer } from "./TimeSeriesLineRenderer";
import type { FeatureStore } from "../store/FeatureStore";
import type { ParamStore } from "../params/ParamStore";
import { BeatPulseSquares } from "./BeatPulseSquares";

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
  | "onsetAcfEnhanced";

interface LineSpec {
  key: LineStoreKey;
  color: number;
  type?: "line" | "bar";
  scale?: TimeSeriesScale;
  colorByValue?: boolean;
}

const LINE_COLORS: readonly LineSpec[] = [
  { key: "waveform", color: 0x00ffff },
  { key: "bufferAcf", color: 0xff00ff },
  { key: "spectrum", color: 0xffff00, type: "bar", scale: "logx" },
  { key: "rmsLow", color: 0xaa0000, type: "bar", colorByValue: true },
  { key: "rmsMid", color: 0x00aa00, type: "bar", colorByValue: true },
  { key: "rmsHigh", color: 0x0000aa, type: "bar", colorByValue: true },
  { key: "rms", color: 0xffffff, colorByValue: true },
  { key: "onset", type: "bar", color: 0xbbbbbb, colorByValue: true },
  { key: "onsetAcf", color: 0xaaaaff },
  { key: "onsetAcfEnhanced", color: 0x00ffff },
];

export class DebugView {
  private lines = new Map<LineStoreKey, TimeSeriesRenderer>();
  private backgroundGrid = new DebugGrid();
  private scrollingBeatGridMarkers: BeatGridMarkers;
  private staticBeatGridMarkers: StaticBeatGridMarkers;
  private beatPulseSquares: BeatPulseSquares;
  private labels: DebugLabels;

  constructor(private deps: DebugViewDeps) {
    this.beatPulseSquares = new BeatPulseSquares();
    this.deps.scene.add(this.beatPulseSquares.object3d);

    this.scrollingBeatGridMarkers = new BeatGridMarkers({
      baseColor: 0x00ff00,
    });
    this.staticBeatGridMarkers = new StaticBeatGridMarkers({
      baseColor: 0x888888,
    });
    this.deps.scene.add(this.scrollingBeatGridMarkers.object3d);
    this.deps.scene.add(this.staticBeatGridMarkers.object3d);
    this.labels = new DebugLabels(deps);
    this.deps.scene.add(this.backgroundGrid.object3d);
    this.deps.scene.add(this.labels.object3d);
    this.createLines();
  }

  update(): void {
    for (const line of this.lines.values()) line.update();
    this.scrollingBeatGridMarkers.update(
      this.deps.store.get("beatGrid"),
      this.deps.store.get("rms").length,
    );

    this.staticBeatGridMarkers.update(
      this.deps.store.get("beatGrid"),
      this.deps.store.get("onsetAcfEnhanced").length,
    );

    this.labels.update();
    this.beatPulseSquares.update(this.deps.store.get("beatPulses"));
  }

  dispose(): void {
    for (const line of this.lines.values()) line.dispose();
    this.lines.clear();
    this.scrollingBeatGridMarkers.dispose();
    this.staticBeatGridMarkers.dispose();
    this.labels.dispose();
    this.beatPulseSquares.dispose();
  }

  private createLines(): void {
    const { store, scene } = this.deps;

    for (const spec of LINE_COLORS) {
      const options = {
        source: () => store.get(spec.key),
        color: spec.color,
        scale: spec.scale,
        colorByValue: spec.colorByValue,
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

    group.add(
      this.lines.get("rmsLow")!.object3d,
      this.lines.get("rmsMid")!.object3d,
      this.lines.get("rmsHigh")!.object3d,
      this.lines.get("rms")!.object3d,
    );
    scene.add(group);

    group.scale.set(4, 0.5, 1);
    group.position.set(-2, 0.5, 0);

    this.scrollingBeatGridMarkers.object3d.position.set(-2, 0, 0);
    this.scrollingBeatGridMarkers.object3d.scale.set(4, 1, 1);

    this.staticBeatGridMarkers.object3d.position.set(-2, -1, 0);

    this.lines.get("onset")!.object3d.position.set(-2, 0, 0);
    this.lines.get("onset")!.object3d.scale.set(4, 0.5, 1);

    this.lines.get("onsetAcf")!.object3d.position.set(-2, -1, 0);
    this.lines.get("onsetAcf")!.object3d.scale.set(4, 5, 1);

    this.lines.get("onsetAcfEnhanced")!.object3d.position.set(-2, -1, 0);
    this.lines.get("onsetAcfEnhanced")!.object3d.scale.set(4, 5, 1);
  }
}
