import { Scene } from "three";
import { LineRenderer, type LineLayout } from "./LineRenderer";
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
  | "rmsLowAuto"
  | "rmsMidAuto"
  | "rmsHighAuto"
  | "rmsAuto"
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
}

const LINE_SPECS: readonly LineSpec[] = [
  { key: "waveform", color: 0x66ffcc, position: [-1, 1, 0] },
  {
    key: "bufferAcf",
    color: 0xcc99ff,
    position: [1, 0.8, 0],
    scale: [1, 0.5, 1],
  },
  {
    key: "spectrum",
    color: 0xffaa66,
    layout: "log",
    position: [0, 1.23, 0],
    scale: [2, 0.5, 1],
  },
  {
    key: "rmsLowAuto",
    color: 0xaa2222,
    scale: [4, 0.5, 1],
    position: [-2, 0, 0],
  },
  {
    key: "rmsMidAuto",
    color: 0x22aa22,
    scale: [4, 0.5, 1],
    position: [-2, 0, 0],
  },
  {
    key: "rmsHighAuto",
    color: 0x2222aa,
    scale: [4, 0.5, 1],
    position: [-2, 0, 0],
  },
  {
    key: "rmsAuto",
    color: 0xffffff,
    scale: [4, 0.5, 1],
    position: [-2, 0, 0],
  },
  {
    key: "onset",
    color: 0xff9966,
    scale: [4, 5, 1],
    position: [-2, -0.2, 0],
  },
  {
    key: "onsetAcf",
    color: 0x66ffff,
    position: [1, -1, 0],
    scale: [1, 10, 1],
  },
  {
    key: "onsetAcfEnhanced",
    color: 0xff99cc,
    position: [1, -1.5, 0],
    scale: [1, 10, 1],
  },
  {
    key: "tea",
    color: 0xffff66,
    position: [-1, -1, 0],
    scale: [1, 2, 1],
  },
];

/**
 * Owns every visualization layer for the audio-feature stream:
 *   - waveform / spectrum / buffer-ACF
 *   - full-band RMS + 3 multiband RMS (low / mid / high) — with per-channel
 *     autogain in front of the renderers
 *   - rms_acf / rms_acf_accum / low_rms_acf + peak markers
 *   - beat-debug overlays (delegated to {@link BeatDebugView}: static grid,
 *     scrolling grid, pulse squares)
 *
 * Lifecycle mirrors the worklet's message contract:
 *   - {@link applyFeatures} on each "features" message — writes into the
 *     FeatureStore and runs autogain for the RMS-history channels.
 *   - {@link applyConfigured} on each "configured" message — disposes the old
 *     renderers, allocates fresh NaN-filled buffers in the store, builds new
 *     renderers at the new sizes. Idempotent.
 *   - {@link update} per frame.
 *   - {@link dispose} on App teardown.
 *
 * Owns its renderers and store keys; App-level concerns (camera, RAF,
 * audio I/O wiring) stay in App.
 */
export class DebugView {
  private lines = new Map<LineStoreKey, LineRenderer>();
  private peakMarkers?: PeakMarkers;
  // private beatDebug: BeatDebugView;

  // Per-channel running peak for the autogained RMS-history lines. See
  // `applyAutoGain` for the EMA-style decay semantics. Zero means
  // uninitialized — the next features message seeds from the full incoming
  // buffer (handles HMR + cold start).
  private runningMax: Record<string, number> = {
    rms: 0,
    rmsLow: 0,
    rmsMid: 0,
    rmsHigh: 0,
  };

  constructor(private deps: DebugViewDeps) {
    // this.beatDebug = new BeatDebugView(deps.scene, deps.store);
  }

  applyFeatures(msg: DebugFeatures): void {
    const { store } = this.deps;
    if (msg.waveform) store.set("waveform", msg.waveform);
    if (msg.spectrum) store.set("spectrum", msg.spectrum);
    if (msg.bufferAcf) store.set("bufferAcf", msg.bufferAcf);
    if (msg.onset) store.set("onset", msg.onset);
    if (msg.onsetAcf) store.set("onsetAcf", msg.onsetAcf);
    if (msg.onsetAcfEnhanced)
      store.set("onsetAcfEnhanced", msg.onsetAcfEnhanced);
    if (msg.tea) store.set("tea", msg.tea);
    if (msg.candidates) store.set("candidates", msg.candidates);
    if (msg.rms) {
      store.set("rms", msg.rms);
      this.applyAutoGain("rms", msg.rms);
    }
    if (msg.rmsLow) {
      store.set("rmsLow", msg.rmsLow);
      this.applyAutoGain("rmsLow", msg.rmsLow);
    }
    if (msg.rmsMid) {
      store.set("rmsMid", msg.rmsMid);
      this.applyAutoGain("rmsMid", msg.rmsMid);
    }
    if (msg.rmsHigh) {
      store.set("rmsHigh", msg.rmsHigh);
      this.applyAutoGain("rmsHigh", msg.rmsHigh);
    }
    // this.beatDebug.applyFeatures(msg);
  }

  applyConfigured(sizes: DebugSizes): void {
    this.disposeLines();
    this.peakMarkers?.dispose();

    const { store, scene } = this.deps;

    store.set("waveform", new Float32Array(sizes.waveformLen));
    store.set("spectrum", new Float32Array(sizes.spectrumLen));
    store.set("rms", new Float32Array(sizes.rmsLen));
    store.set("bufferAcf", new Float32Array(sizes.bufferAcfLen));
    store.set("onset", new Float32Array(sizes.onsetLen));
    store.set("onsetAcf", new Float32Array(sizes.onsetAcfLen));
    store.set("onsetAcfEnhanced", new Float32Array(sizes.onsetAcfLen));
    store.set("tea", new Float32Array(sizes.teaLen));
    const candidatesInit = new Float32Array(sizes.candidatesLen);
    candidatesInit.fill(NaN);
    store.set("candidates", candidatesInit);
    store.set("rmsLow", new Float32Array(sizes.rmsLen));
    store.set("rmsMid", new Float32Array(sizes.rmsLen));
    store.set("rmsHigh", new Float32Array(sizes.rmsLen));
    // Parallel autogained buffers — `applyAutoGain` writes here on each
    // features message, line renderers read from here. Reset runningMax so
    // the next features message seeds from its full incoming buffer (no slow
    // fill-in).
    store.set("rmsAuto", new Float32Array(sizes.rmsLen));
    store.set("rmsLowAuto", new Float32Array(sizes.rmsLen));
    store.set("rmsMidAuto", new Float32Array(sizes.rmsLen));
    store.set("rmsHighAuto", new Float32Array(sizes.rmsLen));
    for (const k of Object.keys(this.runningMax)) this.runningMax[k] = 0;

    this.createLines();

    this.peakMarkers = new PeakMarkers({
      source: () => store.get("candidates"),
      maxPeaks: 10,
      lagDomain: sizes.onsetAcfLen,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: (lag, n) => (n <= 1 ? 0 : (lag / (n - 1)) * 2 - 1),
      baseColor: 0x888888,
    });
    scene.add(this.peakMarkers.object3d);

    // this.beatDebug.applyConfigured(sizes);
  }

  update(): void {
    for (const line of this.lines.values()) line.update();
    this.peakMarkers?.update();
    // this.beatDebug.update();
  }

  dispose(): void {
    this.disposeLines();
    this.peakMarkers?.dispose();
    // this.beatDebug.dispose();
  }

  private createLines(): void {
    const { store, scene } = this.deps;

    for (const spec of LINE_SPECS) {
      const line = new LineRenderer({
        source: () => store.get(spec.key),
        color: spec.color,
        layout: spec.layout,
      });
      if (spec.position) line.object3d.position.set(...spec.position);
      if (spec.scale) line.object3d.scale.set(...spec.scale);
      scene.add(line.object3d);
      this.lines.set(spec.key, line);
    }
  }

  private disposeLines(): void {
    for (const line of this.lines.values()) line.dispose();
    this.lines.clear();
  }

  /**
   * Per-channel autogain over the RMS history lines. Tracks a running peak
   * per band; on each features message updates the peak from the newest
   * sample, decays the prior peak by exp(-dt/τ) where τ = `dsp.autoGain` in
   * seconds and dt = hopSize/sampleRate, and appends `latest / peak` to a
   * parallel `${key}Auto` ring. Older entries in the autogained ring keep
   * their time-of-arrival normalization, so the displayed line doesn't pump
   * when the peak rises or falls — only the newest sample uses the newest
   * peak.
   *
   * Cold start / HMR: when `runningMax[key]` is zero we seed the autogained
   * ring from the full incoming buffer in one pass, so the line is fully
   * populated immediately rather than slowly filling in over ~10 s.
   *
   * Divisor floor: 1e-3 prevents divide-by-near-zero blowups during silence
   * — at the lower end of the slider the running max can decay arbitrarily
   * close to zero, and any tiny new sample would otherwise produce huge
   * normalized output.
   */
  private applyAutoGain(key: string, raw: Float32Array): void {
    if (raw.length === 0) return;
    const auto = this.deps.store.get(`${key}Auto`);
    const eps = 1e-3;
    if (this.runningMax[key] === 0) {
      let mx = 0;
      for (let i = 0; i < raw.length; i++) if (raw[i] > mx) mx = raw[i];
      this.runningMax[key] = mx;
      const denom = Math.max(mx, eps);
      for (let i = 0; i < raw.length; i++) auto[i] = raw[i] / denom;
      return;
    }
    const tauSecs = this.deps.paramStore.get("dsp.autoGain");
    const dt =
      this.deps.paramStore.get("dsp.hopSize") /
      this.deps.audioContext.sampleRate;
    const retention = Math.exp(-dt / tauSecs);
    const latest = raw[raw.length - 1];
    this.runningMax[key] = Math.max(latest, retention * this.runningMax[key]);
    const denom = Math.max(this.runningMax[key], eps);
    auto.copyWithin(0, 1);
    auto[auto.length - 1] = latest / denom;
  }
}
