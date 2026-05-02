import { Vector3 } from "three";
import { createSceneAndCamera } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { DebugView } from "./render/debug/DebugView";
import { FeatureStore } from "./store/FeatureStore";
import { FpsOverlay } from "./ui/Stats";
import { BoxView } from "./render/BoxView";

import type { ParamStore } from "./params/ParamStore";
import type { WebGPURenderer } from "three/webgpu";

export interface AppDeps {
  canvas: HTMLCanvasElement;
  renderer: WebGPURenderer;
  audioContext: AudioContext;
  workletNode: AudioWorkletNode;
  paramStore: ParamStore;
}

type WorkletMsg = {
  type: "features";
  buffers: Record<string, Float32Array>;
};

export class App {
  private rig!: CameraRig;
  private store = new FeatureStore();
  private last = 0;
  private fps = new FpsOverlay();
  private rafHandle: number | null = null;
  private keydownHandler: (e: KeyboardEvent) => void = () => {};
  private resizeHandler: () => void = () => {};
  private debugView!: DebugView;
  private boxView!: BoxView;

  constructor(private deps: AppDeps) {}

  start(): void {
    const { renderer, workletNode, paramStore, audioContext } = this.deps;

    const { scene, camera } = createSceneAndCamera();
    this.debugView = new DebugView({
      scene,
      store: this.store,
      paramStore,
      audioContext,
    });

    this.boxView = new BoxView({
      scene,
      store: this.store,
      paramStore,
    });

    this.rig = new CameraRig(camera);
    this.rig.addPreset("front", {
      position: new Vector3(0, 0, 4),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("side", {
      position: new Vector3(4, 0, 0),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("spectrum", {
      position: new Vector3(0, 0, 1.4),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("rms", {
      position: new Vector3(0, -0.5, 1.4),
      target: new Vector3(0, -0.5, 0),
    });
    this.rig.addPreset("buffer-acf", {
      position: new Vector3(0, 0.5, 1.4),
      target: new Vector3(0, 0.5, 0),
    });
    this.rig.addPreset("rms-acf", {
      position: new Vector3(0, -1.0, 1.4),
      target: new Vector3(0, -1.0, 0),
    });
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
      const msg = e.data as WorkletMsg;
      if (msg.type !== "features") return;
      for (const [name, buf] of Object.entries(msg.buffers)) {
        this.store.set(name, buf);
      }
    };

    const loop = (now: number) => {
      this.fps.begin();
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;
      this.rig.update(dt);
      this.debugView.update();
      this.boxView.update();
      renderer.render(scene, camera);
      this.fps.end();
      this.rafHandle = requestAnimationFrame(loop);
    };
    this.rafHandle = requestAnimationFrame(loop);
  }

  dispose(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    window.removeEventListener("keydown", this.keydownHandler);
    window.removeEventListener("resize", this.resizeHandler);
    this.debugView?.dispose();
    this.boxView?.dispose();
    this.fps.unmount();
    this.deps.workletNode.port.onmessage = null;
  }
}
