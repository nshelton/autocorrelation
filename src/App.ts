import { Vector3 } from "three";
import { PostProcessing } from "three/webgpu";
import { pass, mrt, output, transformedNormalView } from "three/tsl";
// @ts-expect-error - local copy of three's GTAONode example, no .d.ts
import { ao } from "./render/GTAONode.js";
import { createSceneAndCamera } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { FeatureStore } from "./store/FeatureStore";
import { FpsOverlay } from "./ui/Stats";
import { ComponentManager } from "./render/components/ComponentManager";
import { COMPONENTS } from "./render/components";

import type { ParamStore } from "./params/ParamStore";
import type { WebGPURenderer } from "three/webgpu";
import type { CameraPose } from "./render/CameraRig";

const CAMERA_POSE_KEY = "autocorrelation.camera.pose";

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
  private post!: PostProcessing;

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

    // Post-processing: scene pass with MRT (color + view-space normal) → GTAO → multiply.
    const scenePass = pass(scene, camera);
    scenePass.setMRT(
      mrt({
        output,
        normal: transformedNormalView,
      }),
    );
    const sceneColor = scenePass.getTextureNode("output");
    const sceneNormal = scenePass.getTextureNode("normal");
    const sceneDepth = scenePass.getTextureNode("depth");
    const aoNode = ao(sceneDepth, sceneNormal, camera);
    this.post = new PostProcessing(renderer);
    this.post.outputNode = sceneColor.mul(aoNode);

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
      void this.post.renderAsync();
      this.fps.end();
      this.rafHandle = requestAnimationFrame(loop);
    };
    this.rafHandle = requestAnimationFrame(loop);
  }

  bindUI(parent: import("tweakpane").FolderApi): void {
    this.components.bindUI(parent);
  }

  dispose(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    window.removeEventListener("keydown", this.keydownHandler);
    window.removeEventListener("resize", this.resizeHandler);
    this.components?.dispose();
    this.rig?.dispose();
    this.fps.unmount();
    this.deps.workletNode.port.onmessage = null;
  }
}
