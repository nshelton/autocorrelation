import { Scene } from "three";
import { DebugGrid } from "./DebugGrid";
import { TimeSeriesRenderer, type LineLayout } from "./TimeSeriesRenderer";
import { TimeSeriesLineRenderer } from "./TimeSeriesLineRenderer";
import { PeakMarkers } from "./PeakMarkers";
import { BeatDebugView } from "./BeatDebugView";
import type { FeatureStore } from "../store/FeatureStore";
import type { ParamStore } from "../params/ParamStore";

export interface DebugViewDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
}

export interface DebugFeatures {
  waveform?: Float32Array;
  spectrum?: Float32Array;
  rms?: Float32Array;
  bufferAcf?: Float32Array;
  onset?: Float32Array;
  onsetAcf?: Float32Array;
  onsetAcfEnhanced?: Float32Array;
  tea?: Float32Array;
  candidates?: Float32Array;
  beatGrid?: Float32Array;
  beatPulses?: Float32Array;
  beatState?: Float32Array;
  rmsLow?: Float32Array;
  rmsMid?: Float32Array;
  rmsHigh?: Float32Array;
}

export interface DebugSizes {
  waveformLen: number;
  spectrumLen: number;
  bufferAcfLen: number;
  rmsLen: number;
  onsetLen: number;
  onsetAcfLen: number;
  teaLen: number;
  candidatesLen: number;
  beatGridLen: number;
  beatPulsesLen: number;
  beatStateLen: number;
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

/**
 * Visualization layer. Owns the renderers; doesn't own the FeatureStore —
 * App allocates and writes buffers, renderers pull them via `source`
 * callbacks. Line renderers are built once and auto-resize via
 * {@link TimeSeriesRenderer}'s allocate(n) path on length change.
 *
 * App calls {@link applyConfigured} on each worklet "configured" message to
 * rebuild the non-resizable renderers (peak markers + beat debug) at the
 * new sizes.
 */
export class DebugView {
  private lines = new Map<LineStoreKey, TimeSeriesRenderer>();
  private grid = new DebugGrid();
  private peakMarkers?: PeakMarkers;
  private beatDebug: BeatDebugView;

  constructor(private deps: DebugViewDeps) {
    this.deps.scene.add(this.grid.object3d);
    this.beatDebug = new BeatDebugView(deps.scene, deps.store);
    this.createLines();
  }

  applyConfigured(sizes: DebugSizes): void {
    this.peakMarkers?.dispose();
    const { store, scene } = this.deps;
    this.peakMarkers = new PeakMarkers({
      source: () => store.get("candidates"),
      maxPeaks: 10,
      lagDomain: sizes.onsetAcfLen,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: (lag, n) => -2 + 4 * (lag / (n - 1)),
      baseColor: 0x888888,
    });
    scene.add(this.peakMarkers.object3d);
    this.beatDebug.applyConfigured(sizes);
  }

  update(): void {
    for (const line of this.lines.values()) line.update();
    this.peakMarkers?.update();
    this.beatDebug.update();
  }

  dispose(): void {
    for (const line of this.lines.values()) line.dispose();
    this.lines.clear();
    this.peakMarkers?.dispose();
    this.beatDebug.dispose();
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
