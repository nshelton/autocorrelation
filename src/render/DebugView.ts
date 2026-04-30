import { Scene } from "three";
import { LineRenderer } from "./LineRenderer";
import { PeakMarkers } from "./PeakMarkers";
import { BeatDebugView } from "./BeatDebugView";
import { linearLayout, logSpectrumLayout } from "./LineLayouts";
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
  rmsAcf?: Float32Array;
  rmsAcfAccum?: Float32Array;
  acfPeaks?: Float32Array;
  beatGrid?: Float32Array;
  beatPulses?: Float32Array;
  beatState?: Float32Array;
  rmsLow?: Float32Array;
  rmsMid?: Float32Array;
  rmsHigh?: Float32Array;
  rmsAcfLow?: Float32Array;
}

export interface DebugSizes {
  waveformLen: number;
  spectrumLen: number;
  bufferAcfLen: number;
  rmsLen: number;
  rmsAcfLen: number;
  acfPeaksLen: number;
  beatGridLen: number;
  beatPulsesLen: number;
  beatStateLen: number;
}

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
  private waveformLine?: LineRenderer;
  private spectrumLine?: LineRenderer;
  private rmsLine?: LineRenderer;
  private bufferAcfLine?: LineRenderer;
  private rmsAcfLine?: LineRenderer;
  private rmsAcfAccumLine?: LineRenderer;
  private peakMarkers?: PeakMarkers;
  private lowRmsLine?: LineRenderer;
  private midRmsLine?: LineRenderer;
  private highRmsLine?: LineRenderer;
  private lowRmsAcfLine?: LineRenderer;
  private beatDebug: BeatDebugView;

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
    this.beatDebug = new BeatDebugView(deps.scene, deps.store);
  }

  applyFeatures(msg: DebugFeatures): void {
    const { store } = this.deps;
    if (msg.waveform) store.set("waveform", msg.waveform);
    if (msg.spectrum) store.set("spectrum", msg.spectrum);
    if (msg.bufferAcf) store.set("bufferAcf", msg.bufferAcf);
    if (msg.rmsAcf) store.set("rmsAcf", msg.rmsAcf);
    if (msg.rmsAcfAccum) store.set("rmsAcfAccum", msg.rmsAcfAccum);
    if (msg.acfPeaks) store.set("acfPeaks", msg.acfPeaks);
    if (msg.rmsAcfLow) store.set("rmsAcfLow", msg.rmsAcfLow);
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
    this.beatDebug.applyFeatures(msg);
  }

  applyConfigured(sizes: DebugSizes): void {
    for (const line of [
      this.waveformLine,
      this.bufferAcfLine,
      this.spectrumLine,
      this.lowRmsLine,
      this.midRmsLine,
      this.highRmsLine,
      this.rmsLine,
      this.lowRmsAcfLine,
      this.rmsAcfLine,
      this.rmsAcfAccumLine,
    ]) {
      line?.dispose();
    }
    this.peakMarkers?.dispose();

    const { store, scene } = this.deps;

    store.set("waveform", new Float32Array(sizes.waveformLen));
    store.set("spectrum", new Float32Array(sizes.spectrumLen));
    store.set("rms", new Float32Array(sizes.rmsLen));
    store.set("bufferAcf", new Float32Array(sizes.bufferAcfLen));
    store.set("rmsAcf", new Float32Array(sizes.rmsAcfLen));
    store.set("rmsAcfAccum", new Float32Array(sizes.rmsAcfLen));
    const peaksInit = new Float32Array(sizes.acfPeaksLen);
    peaksInit.fill(NaN);
    store.set("acfPeaks", peaksInit);
    store.set("rmsLow", new Float32Array(sizes.rmsLen));
    store.set("rmsMid", new Float32Array(sizes.rmsLen));
    store.set("rmsHigh", new Float32Array(sizes.rmsLen));
    store.set("rmsAcfLow", new Float32Array(sizes.rmsAcfLen));
    // Parallel autogained buffers — `applyAutoGain` writes here on each
    // features message, line renderers read from here. Reset runningMax so
    // the next features message seeds from its full incoming buffer (no slow
    // fill-in).
    store.set("rmsAuto", new Float32Array(sizes.rmsLen));
    store.set("rmsLowAuto", new Float32Array(sizes.rmsLen));
    store.set("rmsMidAuto", new Float32Array(sizes.rmsLen));
    store.set("rmsHighAuto", new Float32Array(sizes.rmsLen));
    for (const k of Object.keys(this.runningMax)) this.runningMax[k] = 0;

    this.waveformLine = new LineRenderer({
      source: () => store.get("waveform"),
      layout: linearLayout(1.0, 0.4),
      color: 0x66ffcc,
    });
    scene.add(this.waveformLine.object3d);

    this.bufferAcfLine = new LineRenderer({
      source: () => store.get("bufferAcf"),
      layout: linearLayout(0.5, 0.4),
      color: 0xcc99ff,
    });
    scene.add(this.bufferAcfLine.object3d);

    this.spectrumLine = new LineRenderer({
      source: () => store.get("spectrum"),
      layout: logSpectrumLayout(0.0, 0.4),
      color: 0xffaa66,
    });
    scene.add(this.spectrumLine.object3d);

    this.lowRmsLine = new LineRenderer({
      source: () => store.get("rmsLowAuto"),
      layout: linearLayout(-0.5, 0.4),
      color: 0xaa2222,
    });
    scene.add(this.lowRmsLine.object3d);

    this.midRmsLine = new LineRenderer({
      source: () => store.get("rmsMidAuto"),
      layout: linearLayout(-0.5, 0.4),
      color: 0x22aa22,
    });
    scene.add(this.midRmsLine.object3d);

    this.highRmsLine = new LineRenderer({
      source: () => store.get("rmsHighAuto"),
      layout: linearLayout(-0.5, 0.4),
      color: 0x2222aa,
    });
    scene.add(this.highRmsLine.object3d);

    this.rmsLine = new LineRenderer({
      source: () => store.get("rmsAuto"),
      layout: linearLayout(-0.5, 0.4),
      color: 0xffffff,
    });
    scene.add(this.rmsLine.object3d);

    this.lowRmsAcfLine = new LineRenderer({
      source: () => store.get("rmsAcfLow"),
      layout: linearLayout(-1.0, 0.4),
      color: 0xbb8888,
    });
    scene.add(this.lowRmsAcfLine.object3d);

    this.rmsAcfLine = new LineRenderer({
      source: () => store.get("rmsAcf"),
      layout: linearLayout(-1.0, 0.4),
      color: 0x666666,
    });
    scene.add(this.rmsAcfLine.object3d);

    this.rmsAcfAccumLine = new LineRenderer({
      source: () => store.get("rmsAcfAccum"),
      layout: linearLayout(-1.0, 0.4),
      color: 0x66ffff,
    });
    scene.add(this.rmsAcfAccumLine.object3d);

    this.peakMarkers = new PeakMarkers({
      source: () => store.get("acfPeaks"),
      maxPeaks: 10,
      lagDomain: sizes.rmsAcfLen,
      yCenter: -1.0,
      ySpan: 0.4,
      // Mirrors linearLayout's x-formula. If you change one, change the other —
      // markers must sit on the exact x-positions of the accumulator line they
      // annotate, or peaks will visually drift off the line they describe.
      xForLag: (lag, n) => (n <= 1 ? 0 : (lag / (n - 1)) * 2 - 1),
      baseColor: 0x888888,
    });
    scene.add(this.peakMarkers.object3d);

    this.beatDebug.applyConfigured(sizes);
  }

  update(): void {
    this.waveformLine?.update();
    this.bufferAcfLine?.update();
    this.spectrumLine?.update();
    this.lowRmsLine?.update();
    this.midRmsLine?.update();
    this.highRmsLine?.update();
    this.rmsLine?.update();
    this.lowRmsAcfLine?.update();
    this.rmsAcfLine?.update();
    this.rmsAcfAccumLine?.update();
    this.peakMarkers?.update();
    this.beatDebug.update();
  }

  dispose(): void {
    for (const line of [
      this.waveformLine,
      this.bufferAcfLine,
      this.spectrumLine,
      this.lowRmsLine,
      this.midRmsLine,
      this.highRmsLine,
      this.rmsLine,
      this.lowRmsAcfLine,
      this.rmsAcfLine,
      this.rmsAcfAccumLine,
    ]) {
      line?.dispose();
    }
    this.peakMarkers?.dispose();
    this.beatDebug.dispose();
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
      this.deps.paramStore.get("dsp.hopSize") / this.deps.audioContext.sampleRate;
    const retention = Math.exp(-dt / tauSecs);
    const latest = raw[raw.length - 1];
    this.runningMax[key] = Math.max(latest, retention * this.runningMax[key]);
    const denom = Math.max(this.runningMax[key], eps);
    auto.copyWithin(0, 1);
    auto[auto.length - 1] = latest / denom;
  }
}
