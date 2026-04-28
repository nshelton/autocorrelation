import { Vector3 } from "three";
import { createSceneAndCamera } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { LineRenderer } from "./render/LineRenderer";
import { PeakMarkers } from "./render/PeakMarkers";
import { linearLayout, logSpectrumLayout } from "./render/LineLayouts";
import { FeatureStore } from "./store/FeatureStore";
import { FpsOverlay } from "./ui/Stats";
import type { ParamStore } from "./params/ParamStore";
import type { WebGPURenderer } from "three/webgpu";

export interface AppDeps {
  canvas: HTMLCanvasElement;
  renderer: WebGPURenderer;
  audioContext: AudioContext;
  workletNode: AudioWorkletNode;
  paramStore: ParamStore;
}

export class App {
  private rig!: CameraRig;
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
  private store = new FeatureStore();
  private last = 0;
  private fps = new FpsOverlay();
  private scene!: import("three").Scene;
  private rafHandle: number | null = null;
  private keydownHandler: (e: KeyboardEvent) => void = () => {};
  private resizeHandler: () => void = () => {};

  constructor(private deps: AppDeps) {}

  start(): void {
    const { renderer, workletNode } = this.deps;

    const { scene, camera } = createSceneAndCamera();
    this.scene = scene;

    this.rig = new CameraRig(camera);
    this.rig.addPreset("front", { position: new Vector3(0, 0, 4), target: new Vector3(0, 0, 0) });
    this.rig.addPreset("side", { position: new Vector3(4, 0, 0), target: new Vector3(0, 0, 0) });
    this.rig.addPreset("spectrum", { position: new Vector3(0, 0, 1.4), target: new Vector3(0, 0, 0) });
    this.rig.addPreset("rms", { position: new Vector3(0, -0.5, 1.4), target: new Vector3(0, -0.5, 0) });
    this.rig.addPreset("buffer-acf", { position: new Vector3(0, 0.5, 1.4), target: new Vector3(0, 0.5, 0) });
    this.rig.addPreset("rms-acf", { position: new Vector3(0, -1.0, 1.4), target: new Vector3(0, -1.0, 0) });
    void this.rig.goTo("front", { duration: 0 });

    this.fps.mount();

    let toggled = false;
    const presetKeys: Record<string, string> = {
      "1": "front",
      "2": "side",
      "3": "spectrum",
      "4": "rms",
      "5": "buffer-acf",
      "6": "rms-acf",
    };
    this.keydownHandler = (e) => {
      const preset = presetKeys[e.key];
      if (preset) {
        this.rig.goTo(preset, { duration: 0.8 });
        return;
      }
      if (e.key === " ") {
        toggled = !toggled;
        this.rig.goTo(toggled ? "side" : "front", { duration: 0.8 });
      }
    };
    window.addEventListener("keydown", this.keydownHandler);

    this.resizeHandler = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", this.resizeHandler);

    workletNode.port.onmessage = (e) => {
      const msg = e.data as
        | {
            type: "features";
            waveform?: Float32Array;
            spectrum?: Float32Array;
            rms?: Float32Array;
            bufferAcf?: Float32Array;
            rmsAcf?: Float32Array;
            rmsAcfAccum?: Float32Array;
            acfPeaks?: Float32Array;
            rmsLow?: Float32Array;
            rmsMid?: Float32Array;
            rmsHigh?: Float32Array;
            rmsAcfLow?: Float32Array;
          }
        | {
            type: "configured";
            waveformLen: number;
            spectrumLen: number;
            bufferAcfLen: number;
            rmsLen: number;
            rmsAcfLen: number;
            acfPeaksLen: number;
          };
      if (msg.type === "features") {
        if (msg.waveform) this.store.set("waveform", msg.waveform);
        if (msg.spectrum) this.store.set("spectrum", msg.spectrum);
        if (msg.rms) this.store.set("rms", msg.rms);
        if (msg.bufferAcf) this.store.set("bufferAcf", msg.bufferAcf);
        if (msg.rmsAcf) this.store.set("rmsAcf", msg.rmsAcf);
        if (msg.rmsAcfAccum) this.store.set("rmsAcfAccum", msg.rmsAcfAccum);
        if (msg.acfPeaks) this.store.set("acfPeaks", msg.acfPeaks);
        if (msg.rmsLow) this.store.set("rmsLow", msg.rmsLow);
        if (msg.rmsMid) this.store.set("rmsMid", msg.rmsMid);
        if (msg.rmsHigh) this.store.set("rmsHigh", msg.rmsHigh);
        if (msg.rmsAcfLow) this.store.set("rmsAcfLow", msg.rmsAcfLow);
        return;
      }
      if (msg.type === "configured") {
        this.rebuildLineRenderers(msg);
      }
    };

    const loop = (now: number) => {
      this.fps.begin();
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;
      this.rig.update(dt);
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
      renderer.render(scene, camera);
      this.fps.end();
      this.rafHandle = requestAnimationFrame(loop);
    };
    this.rafHandle = requestAnimationFrame(loop);
  }

  private rebuildLineRenderers(sizes: {
    waveformLen: number;
    spectrumLen: number;
    bufferAcfLen: number;
    rmsLen: number;
    rmsAcfLen: number;
    acfPeaksLen: number;
  }): void {
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

    this.store.set("waveform", new Float32Array(sizes.waveformLen));
    this.store.set("spectrum", new Float32Array(sizes.spectrumLen));
    this.store.set("rms", new Float32Array(sizes.rmsLen));
    this.store.set("bufferAcf", new Float32Array(sizes.bufferAcfLen));
    this.store.set("rmsAcf", new Float32Array(sizes.rmsAcfLen));
    this.store.set("rmsAcfAccum", new Float32Array(sizes.rmsAcfLen));
    const peaksInit = new Float32Array(sizes.acfPeaksLen);
    peaksInit.fill(NaN);
    this.store.set("acfPeaks", peaksInit);
    this.store.set("rmsLow", new Float32Array(sizes.rmsLen));
    this.store.set("rmsMid", new Float32Array(sizes.rmsLen));
    this.store.set("rmsHigh", new Float32Array(sizes.rmsLen));
    this.store.set("rmsAcfLow", new Float32Array(sizes.rmsAcfLen));

    this.waveformLine = new LineRenderer({
      source: () => this.store.get("waveform"),
      layout: linearLayout(1.0, 0.4),
      color: 0x66ffcc,
    });
    this.scene.add(this.waveformLine.object3d);

    this.bufferAcfLine = new LineRenderer({
      source: () => this.store.get("bufferAcf"),
      layout: linearLayout(0.5, 0.4),
      color: 0xcc99ff,
    });
    this.scene.add(this.bufferAcfLine.object3d);

    this.spectrumLine = new LineRenderer({
      source: () => this.store.get("spectrum"),
      layout: logSpectrumLayout(0.0, 0.4),
      color: 0xffaa66,
    });
    this.scene.add(this.spectrumLine.object3d);

    this.lowRmsLine = new LineRenderer({
      source: () => this.store.get("rmsLow"),
      layout: linearLayout(-0.5, 0.4),
      color: 0xff4444,
    });
    this.scene.add(this.lowRmsLine.object3d);

    this.midRmsLine = new LineRenderer({
      source: () => this.store.get("rmsMid"),
      layout: linearLayout(-0.5, 0.4),
      color: 0x44ff44,
    });
    this.scene.add(this.midRmsLine.object3d);

    this.highRmsLine = new LineRenderer({
      source: () => this.store.get("rmsHigh"),
      layout: linearLayout(-0.5, 0.4),
      color: 0x4488ff,
    });
    this.scene.add(this.highRmsLine.object3d);

    this.rmsLine = new LineRenderer({
      source: () => this.store.get("rms"),
      layout: linearLayout(-0.5, 0.4),
      color: 0xffffff,
    });
    this.scene.add(this.rmsLine.object3d);

    this.lowRmsAcfLine = new LineRenderer({
      source: () => this.store.get("rmsAcfLow"),
      layout: linearLayout(-1.0, 0.4),
      color: 0xff4444,
    });
    this.scene.add(this.lowRmsAcfLine.object3d);

    this.rmsAcfLine = new LineRenderer({
      source: () => this.store.get("rmsAcf"),
      layout: linearLayout(-1.0, 0.4),
      color: 0xff99cc,
    });
    this.scene.add(this.rmsAcfLine.object3d);

    this.rmsAcfAccumLine = new LineRenderer({
      source: () => this.store.get("rmsAcfAccum"),
      layout: linearLayout(-1.0, 0.4),
      color: 0x66ffff,
    });
    this.scene.add(this.rmsAcfAccumLine.object3d);

    this.peakMarkers = new PeakMarkers({
      source: () => this.store.get("acfPeaks"),
      maxPeaks: 10,
      lagDomain: sizes.rmsAcfLen,
      yCenter: -1.0,
      ySpan: 0.4,
      // Match linearLayout's x-mapping exactly (i / (n-1)) * 2 - 1.
      xForLag: (lag, n) => (n <= 1 ? 0 : (lag / (n - 1)) * 2 - 1),
      baseColor: 0xffff66,
    });
    this.scene.add(this.peakMarkers.object3d);
  }

  dispose(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    window.removeEventListener("keydown", this.keydownHandler);
    window.removeEventListener("resize", this.resizeHandler);
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
    this.fps.unmount();
    this.deps.workletNode.port.onmessage = null;
  }
}
