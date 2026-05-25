import { DirectionalLight, Scene, Vector3 } from "three";
import { createSceneAndCamera } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { PostStack } from "./render/post/PostStack";
import { buildPostEffects } from "./render/post";
import { FeatureStore } from "./store/FeatureStore";
import { FpsOverlay } from "./ui/Stats";
import { ComponentManager } from "./render/components/ComponentManager";
import { COMPONENTS } from "./render/components";

import { Modulator } from "./params/Modulator";
import { bindParam } from "./params/bindParam";
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
  public modulator!: Modulator;
  private cameraUnsub: (() => void) | null = null;
  private directionalLight!: DirectionalLight;
  private scene!: Scene;

  constructor(private deps: AppDeps) {}

  start(): void {
    const { renderer, workletNode, paramStore, audioContext } = this.deps;

    const { scene, camera } = createSceneAndCamera();
    this.scene = scene;

    // Direction matches OrbitalCloud's previous hardcoded lightDir so when
    // its cube/splat modes eventually consume this uniform their look stays
    // continuous. Toggle adds/removes from scene in the bindCameraUI handler.
    this.directionalLight = new DirectionalLight(0xffffff, 1.0);
    this.directionalLight.position.set(4.08, 8.66, 3.06);
    if (paramStore.get("light.directional.enabled") as boolean) {
      scene.add(this.directionalLight);
    }

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
    this.modulator = new Modulator(paramStore, this.store);
    this.components.start();

    this.postStack = new PostStack(renderer, scene, camera, paramStore, buildPostEffects(renderer));
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
      this.modulator.tick();
      this.components.update();
      void this.postStack.renderAsync();
      this.fps.end();
      this.rafHandle = requestAnimationFrame(loop);
    };
    this.rafHandle = requestAnimationFrame(loop);
  }

  bindUI(parent: import("tweakpane").FolderApi): void {
    this.components.bindUI(parent, this.modulator);
  }

  bindPostUI(folder: FolderApi): void {
    this.postStack.bindUI(folder, this.modulator);
  }

  bindCameraUI(folder: FolderApi): void {
    const store = this.deps.paramStore;
    const camera = this.rig.camera;

    const fovSchema = store.schemaFor("camera.fov");
    const presetSchema = store.schemaFor("camera.preset");
    const lightSchema = store.schemaFor("light.directional.enabled");
    if (!fovSchema || !presetSchema || !lightSchema) {
      throw new Error("bindCameraUI: required schemas missing");
    }
    bindParam(folder, store, this.modulator, fovSchema);
    bindParam(folder, store, this.modulator, presetSchema);
    bindParam(folder, store, this.modulator, lightSchema);

    // Side-effects subscriber. `camera.fov` is the only key here that may be
    // modulated, so its camera-uniform write runs on every notify. The other
    // two keys (preset, light) are not modulatable; their side-effects are
    // gated on source==="user" defensively — they aren't modulatable and would
    // never fire from the modulator, but gating is cheap.
    this.cameraUnsub = store.subscribe((key, value, source) => {
      if (key === "camera.fov" && typeof value === "number") {
        camera.fov = value;
        camera.updateProjectionMatrix();
      } else if (key === "camera.preset" && typeof value === "number" && source === "user") {
        const name = CAMERA_PRESET_NAMES[value];
        if (name) void this.rig.goTo(name, { duration: 0.8 });
      } else if (key === "light.directional.enabled" && typeof value === "boolean" && source === "user") {
        if (value) this.scene.add(this.directionalLight);
        else this.scene.remove(this.directionalLight);
      }
    });

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
    this.modulator?.dispose();
    this.postStack?.dispose();
    this.rig?.dispose();
    this.fps.unmount();
    this.cameraUnsub?.();
    this.cameraUnsub = null;
    this.deps.workletNode.port.onmessage = null;
  }
}
