import { Vector3 } from "three";
import { createScene } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { LineRenderer } from "./render/LineRenderer";
import dspWorkletUrl from "./audio/dsp-worklet?worker&url";
import { createMicSource } from "./audio/AudioSource";
import { FeatureStore } from "./store/FeatureStore";

export class App {
  private rig!: CameraRig;
  private line!: LineRenderer;
  private store = new FeatureStore();
  private last = 0;

  async start(canvas: HTMLCanvasElement): Promise<void> {
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
    await this.rig.goTo("front", { duration: 0 });

    this.store.set("waveform", new Float32Array(2048));

    this.line = new LineRenderer({
      source: () => this.store.get("waveform"),
      color: 0x66ffcc,
    });
    scene.add(this.line.object3d);

    let toggled = false;
    window.addEventListener("keydown", (e) => {
      if (e.key !== " ") return;
      toggled = !toggled;
      this.rig.goTo(toggled ? "side" : "front", { duration: 0.8 });
    });

    const { context, source } = await createMicSource();
    await context.audioWorklet.addModule(dspWorkletUrl);
    const node = new AudioWorkletNode(context, "dsp-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
    });
    source.connect(node);

    node.port.onmessage = (e) => {
      const msg = e.data as { type: string; buffer: Float32Array };
      if (msg.type === "waveform") {
        this.store.set("waveform", msg.buffer);
      }
    };

    const loop = (now: number) => {
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;
      this.rig.update(dt);
      this.line.update();
      renderer.render(scene, camera);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
