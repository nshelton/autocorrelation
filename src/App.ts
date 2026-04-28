import { Vector3 } from "three";
import { createScene } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { LineRenderer } from "./render/LineRenderer";
import { linearLayout, logSpectrumLayout } from "./render/LineLayouts";
import dspWorkletUrl from "./audio/dsp-worklet?worker&url";
import dspWasmUrl from "./wasm-pkg/dsp_bg.wasm?url";
import { createMicSource, type AudioSourceBundle } from "./audio/AudioSource";
import { FeatureStore } from "./store/FeatureStore";
import { FpsOverlay } from "./ui/Stats";

export class App {
  private rig!: CameraRig;
  private waveformLine!: LineRenderer;
  private spectrumLine!: LineRenderer;
  private rmsLine!: LineRenderer;
  private store = new FeatureStore();
  private last = 0;
  private fps = new FpsOverlay();

  async start(
    canvas: HTMLCanvasElement,
    sourceFactory: () => Promise<AudioSourceBundle> = createMicSource,
  ): Promise<void> {
    const { scene, camera, renderer } = await createScene(canvas);

    this.rig = new CameraRig(camera);
    this.rig.addPreset("front", {
      position: new Vector3(0, 0, 3),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("side", {
      position: new Vector3(3, 1, 0),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("spectrum", {
      position: new Vector3(0, 0, 1.4),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("rms", {
      position: new Vector3(0, -0.6, 1.4),
      target: new Vector3(0, -0.6, 0),
    });
    await this.rig.goTo("front", { duration: 0 });

    this.store.set("waveform", new Float32Array(2048));
    this.store.set("spectrum", new Float32Array(1024));
    this.store.set("rms", new Float32Array(256));

    this.waveformLine = new LineRenderer({
      source: () => this.store.get("waveform"),
      layout: linearLayout(0.6, 0.5),
      color: 0x66ffcc,
    });
    scene.add(this.waveformLine.object3d);

    this.spectrumLine = new LineRenderer({
      source: () => this.store.get("spectrum"),
      layout: logSpectrumLayout(0.0, 0.5),
      color: 0xffaa66,
    });
    scene.add(this.spectrumLine.object3d);

    this.rmsLine = new LineRenderer({
      source: () => this.store.get("rms"),
      layout: linearLayout(-0.6, 0.5),
      color: 0xffffff,
    });
    scene.add(this.rmsLine.object3d);

    this.fps.mount();

    let toggled = false;
    const presetKeys: Record<string, string> = {
      "1": "front",
      "2": "side",
      "3": "spectrum",
      "4": "rms",
    };
    window.addEventListener("keydown", (e) => {
      const preset = presetKeys[e.key];
      if (preset) {
        this.rig.goTo(preset, { duration: 0.8 });
        return;
      }
      if (e.key === " ") {
        toggled = !toggled;
        this.rig.goTo(toggled ? "side" : "front", { duration: 0.8 });
      }
    });

    const { context, source } = await sourceFactory();
    const wasmModule = await WebAssembly.compileStreaming(fetch(dspWasmUrl));
    await context.audioWorklet.addModule(dspWorkletUrl);
    const node = new AudioWorkletNode(context, "dsp-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { wasmModule },
    });
    source.connect(node);

    node.port.onmessage = (e) => {
      const msg = e.data as {
        type: string;
        waveform?: Float32Array;
        spectrum?: Float32Array;
        rms?: Float32Array;
      };
      if (msg.type !== "features") return;
      if (msg.waveform) this.store.set("waveform", msg.waveform);
      if (msg.spectrum) this.store.set("spectrum", msg.spectrum);
      if (msg.rms) this.store.set("rms", msg.rms);
    };

    const loop = (now: number) => {
      this.fps.begin();
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;
      this.rig.update(dt);
      this.waveformLine.update();
      this.spectrumLine.update();
      this.rmsLine.update();
      renderer.render(scene, camera);
      this.fps.end();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
