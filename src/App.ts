import { Vector3 } from "three";
import { createSceneAndCamera } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { PostStack } from "./render/post/PostStack";
import { POST_EFFECTS } from "./render/post";
import { FeatureStore } from "./store/FeatureStore";
import { FpsOverlay } from "./ui/Stats";
import { ComponentManager } from "./render/components/ComponentManager";
import { COMPONENTS } from "./render/components";

import type { ParamStore } from "./params/ParamStore";
import type { WebGPURenderer } from "three/webgpu";
import type { CameraPose } from "./render/CameraRig";
import type { FolderApi } from "tweakpane";

const CAMERA_POSE_KEY = "autocorrelation.camera.pose";

// Order MUST match optionLabels in cameraSchemas.ts (`camera.preset`).
const CAMERA_PRESET_NAMES = ["front", "side", "spectrum", "rms", "buffer-acf", "rms-acf"] as const;

function loadCameraPose(): CameraPose | null {
  const raw = localStorage.getItem(CAMERA_POSE_KEY);
  if (!raw) return null;
  try {
    const o = JSON.parse(raw) as { position: [number, number, number]; target: [number, number, number] };
    return {
      position: new Vector3(...o.position),
      target: new Vector3(...o.target),
    };
  } catch {
    return null;
  }
}

function saveCameraPose(pose: CameraPose): void {
  localStorage.setItem(
    CAMERA_POSE_KEY,
    JSON.stringify({
      position: [pose.position.x, pose.position.y, pose.position.z],
      target: [pose.target.x, pose.target.y, pose.target.z],
    }),
  );
}

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
  private components!: ComponentManager;
  private postStack!: PostStack;
  private cameraUnsub: (() => void) | null = null;

  constructor(private deps: AppDeps) {}

  start(): void {
    const { renderer, workletNode, paramStore, audioContext } = this.deps;

    const { scene, camera } = createSceneAndCamera();

    this.components = new ComponentManager(
      {
        scene,
        store: this.store,
        paramStore,
        audioContext,
        renderer,
      },
      COMPONENTS,
    );
    this.components.start();

    this.postStack = new PostStack(renderer, scene, camera, paramStore, POST_EFFECTS);
    this.postStack.build();

    this.rig = new CameraRig(camera, renderer.domElement);
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
    const saved = loadCameraPose();
    if (saved) {
      this.rig.setPose(saved);
    } else {
      void this.rig.goTo("front", { duration: 0 });
    }
    this.rig.controls.addEventListener("end", () => {
      saveCameraPose(this.rig.getPose());
    });

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
    const savePoseAfter = (p: Promise<void>) => {
      void p.then(() => saveCameraPose(this.rig.getPose()));
    };
    this.keydownHandler = (e) => {
      const preset = presetKeys[e.key];
      if (preset) {
        savePoseAfter(this.rig.goTo(preset, { duration: 0.8 }));
        return;
      }
      if (e.key === " ") {
        toggled = !toggled;
        savePoseAfter(this.rig.goTo(toggled ? "side" : "front", { duration: 0.8 }));
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
      this.components.update();
      void this.postStack.renderAsync();
      this.fps.end();
      this.rafHandle = requestAnimationFrame(loop);
    };
    this.rafHandle = requestAnimationFrame(loop);
  }

  bindUI(parent: import("tweakpane").FolderApi): void {
    this.components.bindUI(parent);
  }

  bindPostUI(folder: FolderApi): void {
    this.postStack.bindUI(folder);
  }

  bindCameraUI(folder: FolderApi): void {
    const store = this.deps.paramStore;
    const camera = this.rig.camera;

    // FOV: live-write to camera + projection update.
    const fovBinding = { fov: store.get("camera.fov") as number };
    folder
      .addBinding(fovBinding, "fov", { label: "FOV", min: 20, max: 120, step: 1 })
      .on("change", (e: { value: number }) => store.set("camera.fov", e.value));

    // Preset: dropdown -> rig.goTo. Stored as integer index.
    const presetBinding = { preset: store.get("camera.preset") as number };
    folder
      .addBinding(presetBinding, "preset", {
        label: "Preset",
        options: Object.fromEntries(CAMERA_PRESET_NAMES.map((name, i) => [name, i])),
      })
      .on("change", (e: { value: number }) => store.set("camera.preset", e.value));

    // Subscribe so persisted-on-load values and external writes apply.
    this.cameraUnsub = store.subscribe((key, value) => {
      if (key === "camera.fov" && typeof value === "number") {
        camera.fov = value;
        camera.updateProjectionMatrix();
        fovBinding.fov = value;
      } else if (key === "camera.preset" && typeof value === "number") {
        const name = CAMERA_PRESET_NAMES[value];
        if (name) void this.rig.goTo(name, { duration: 0.8 });
        presetBinding.preset = value;
      }
    });

    // Apply current persisted values once on bind so reload restores state.
    camera.fov = store.get("camera.fov") as number;
    camera.updateProjectionMatrix();
  }

  dispose(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    window.removeEventListener("keydown", this.keydownHandler);
    window.removeEventListener("resize", this.resizeHandler);
    this.components?.dispose();
    this.postStack?.dispose();
    this.rig?.dispose();
    this.fps.unmount();
    this.cameraUnsub?.();
    this.cameraUnsub = null;
    this.deps.workletNode.port.onmessage = null;
  }
}
