import { AmbientLight, DirectionalLight, Vector3 } from "three";
import { createSceneAndCamera } from "./render/Scene";
import { Environment } from "./render/Environment";
import { CameraRig } from "./render/CameraRig";
import { PostStack } from "./render/post/PostStack";
import { buildPostEffects } from "./render/post";
import { FeatureStore } from "./store/FeatureStore";
import { PerfOverlay } from "./ui/PerfOverlay";
import { ComponentManager, type SceneColumnHost } from "./render/components/ComponentManager";
import { COMPONENTS } from "./render/components";
import { stepPhysics, physicsStats } from "./render/components/physics";
import { snapshotCanvas } from "./render/thumbnail";
import { addSystemPresets } from "./params/SystemPresets";
import { shTween } from "./render/orbital/ShTween";

import { Modulator } from "./params/Modulator";
import { PresetStore } from "./params/PresetStore";
import { presetTween } from "./params/PresetTween";
import { addPresetSection } from "./params/PresetSection";
import { bindParam, type ParamProxyRegistry } from "./params/bindParam";
import type { ParamStore } from "./params/ParamStore";
import type { WebGPURenderer } from "three/webgpu";
import type { CameraPose } from "./render/CameraRig";
import type { FolderApi } from "tweakpane";

const CAMERA_POSE_KEY = "autocorrelation.camera.pose";

// Order MUST match optionLabels in cameraSchemas.ts (`camera.preset`).
const CAMERA_PRESET_NAMES = ["front", "side", "spectrum", "rms", "buffer-acf", "rms-acf"] as const;

// `camera.rotate` slider is 0..10; map to deg/s for the orbit (max → 180 deg/s).
const ROTATE_DEG_PER_UNIT = 18;

const DEG2RAD = Math.PI / 180;

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
  private perf = new PerfOverlay();
  private rafHandle: number | null = null;
  private keydownHandler: (e: KeyboardEvent) => void = () => {};
  private resizeHandler: () => void = () => {};
  private components!: ComponentManager;
  private postStack!: PostStack;
  public modulator!: Modulator;
  private presets!: PresetStore;
  // Camera + Post preset sections and their panel subscriptions. Component ones
  // are owned by ComponentManager.
  private presetSections: Array<{ dispose(): void }> = [];
  private cameraUnsub: (() => void) | null = null;
  // Set by captureThumbnail(); the RAF loop resolves it after the next frame.
  private thumbRequest: ((dataUrl: string) => void) | null = null;
  // Last completed frame's main-thread render time (renderAsync start → done).
  private renderMs = NaN;
  private directionalLight!: DirectionalLight;
  private ambientLight!: AmbientLight;
  private environment!: Environment;

  constructor(private deps: AppDeps) {}

  // Spherical position on a fixed radius-10 orbit (the schema defaults land on
  // the retired hardcoded (4.08, 8.66, 3.06)), so the shadow camera's [1, 25]
  // depth bracket stays valid whatever direction the light comes from.
  private applyLightDirection(): void {
    const store = this.deps.paramStore;
    const az = (store.get("light.directional.azimuth") as number) * DEG2RAD;
    const el = (store.get("light.directional.elevation") as number) * DEG2RAD;
    this.directionalLight.position.set(
      10 * Math.cos(el) * Math.cos(az),
      10 * Math.sin(el),
      10 * Math.cos(el) * Math.sin(az),
    );
  }

  start(): void {
    const { renderer, workletNode, paramStore, audioContext } = this.deps;

    const { scene, camera } = createSceneAndCamera();

    // The light stays in the scene permanently even at intensity 0: r170's
    // WebGPU renderer doesn't reliably rebuild existing pipelines when a light
    // is added/removed at runtime, so brightness only ever moves the intensity
    // uniform — see bindCameraUI.
    this.directionalLight = new DirectionalLight(
      paramStore.get("light.directional.color") as number,
      paramStore.get("light.directional.intensity") as number,
    );
    this.applyLightDirection();
    this.directionalLight.castShadow = this.directionalLight.intensity > 0;
    const shadow = this.directionalLight.shadow;
    shadow.mapSize.set(2048, 2048);
    // Ortho box sized to the play area (CONTAINER_HALF/zone scale ~ a few
    // units); light sits ~10 out, so [1, 25] brackets the scene depth.
    shadow.camera.near = 1;
    shadow.camera.far = 25;
    shadow.camera.left = -5;
    shadow.camera.right = 5;
    shadow.camera.top = 5;
    shadow.camera.bottom = -5;
    // Curved instanced surfaces (spheres, capsules) acne under plain depth
    // bias; normal bias pushes samples along the surface normal instead.
    shadow.normalBias = 0.02;
    scene.add(this.directionalLight);
    // Always on — the lit materials' shadow/ambient floor. Without it,
    // toggling the directional light off would black out every lit mesh.
    this.ambientLight = new AmbientLight(
      paramStore.get("light.ambient.color") as number,
      paramStore.get("light.ambient.intensity") as number,
    );
    scene.add(this.ambientLight);

    this.environment = new Environment(scene, paramStore);

    this.components = new ComponentManager(
      {
        scene,
        store: this.store,
        paramStore,
        audioContext,
        renderer,
        camera,
      },
      COMPONENTS,
    );
    this.modulator = new Modulator(paramStore, this.store);
    this.presets = new PresetStore(paramStore, this.modulator);
    this.components.start();

    this.postStack = new PostStack(renderer, scene, camera, paramStore, buildPostEffects());
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

    this.perf.mount();

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
      if (e.key === "p" || e.key === "P") {
        this.perf.toggle();
        return;
      }
      if (e.key === "h" || e.key === "H") {
        // Hide/show the whole GUI (panel + overlays carry the .gui-el class).
        document.body.classList.toggle("gui-hidden");
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
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;

      // three only resets renderer.info inside its own setAnimationLoop; we drive
      // our own RAF, so info.reset() never runs on its own. Without it, drawCalls
      // grow unbounded and the GPU-timestamp accumulator never frame-aligns
      // (previousFrameCalls stays 0 → the per-frame batch boundary is wrong, and
      // the reported timestamp is meaningless). Read last frame's settled counters
      // first, then reset to open a fresh frame's accounting.
      const r = renderer.info.render;
      this.perf.sampleGpu(r.timestamp, r.drawCalls, r.triangles);
      renderer.info.reset();

      const t0 = performance.now();
      this.rig.update(dt);
      const t1 = performance.now();
      shTween.tick(dt, this.deps.paramStore);
      presetTween.tick(dt, this.deps.paramStore);
      this.modulator.tick();
      const t2 = performance.now();
      // Single step for the shared world, before any component reads a body
      // transform. Components never step it themselves.
      stepPhysics(
        paramStore.get("physics.timescale") as number,
        -(paramStore.get("physics.gravity") as number),
      );
      const t3 = performance.now();
      this.components.update();
      const t4 = performance.now();
      const frame = this.postStack.renderAsync();
      // renderAsync suspends at its first await, so nearly all renderer CPU
      // work (node eval, pipeline binding, encoding — scene, shadow AND post
      // passes) runs in the promise continuation after this callback returns.
      // Time it to completion; the sample below reports last frame's value
      // (one frame stale, EMA'd anyway). Includes queue-submit, not GPU wait.
      void frame.then(() => {
        this.renderMs = performance.now() - t4;
      });
      // A WebGPU canvas keeps no preserved drawing buffer, so a thumbnail has
      // to be read right after the frame lands — not whenever the button was
      // clicked, which would sample a black canvas.
      const pending = this.thumbRequest;
      if (pending) {
        this.thumbRequest = null;
        void frame.then(() => pending(snapshotCanvas(renderer.domElement)));
      }

      const dsp = this.store.get("dspPerf");
      const phys = physicsStats();
      this.perf.sample({
        fps: dt > 0 ? 1 / dt : NaN,
        cameraMs: t1 - t0,
        physicsMs: t3 - t2,
        componentsMs: t2 - t1 + (t4 - t3),
        renderMs: this.renderMs,
        analysisMs: dsp.length > 0 ? dsp[0] : NaN,
        analysisHz: dsp.length > 1 ? dsp[1] : NaN,
        bodies: phys.bodies,
        colliders: phys.colliders,
        active: phys.active,
        now,
      });

      this.rafHandle = requestAnimationFrame(loop);
    };
    this.rafHandle = requestAnimationFrame(loop);
  }

  // `scenes` is the toggle column; each enabled scene's params get their own
  // panel from `host`.
  bindUI(scenes: FolderApi, host: SceneColumnHost): void {
    this.components.bindUI(scenes, host, this.modulator, this.presets);
  }

  // Every physics component shares one world, so timescale and gravity are
  // global state — they live with the system settings, not in any scene.
  bindPhysicsUI(folder: FolderApi): void {
    const store = this.deps.paramStore;
    const proxies: ParamProxyRegistry = new Map();
    for (const key of ["physics.timescale", "physics.gravity"]) {
      const schema = store.schemaFor(key);
      if (!schema) throw new Error(`bindPhysicsUI: ${key} schema missing`);
      bindParam(folder, store, this.modulator, schema, proxies);
    }
    const unsub = store.subscribe((key, _value, source) => {
      if (source !== "user") return;
      const refresh = proxies.get(key);
      if (!refresh) return;
      refresh();
      folder.refresh();
    });
    this.presetSections.push({ dispose: unsub });
    this.presetSections.push(
      addPresetSection(folder, this.presets, { id: "physics", prefixes: ["physics"] }),
    );
  }

  // System presets: every param and every modulation in one snapshot, with a
  // thumbnail of the scene. Lives in the far-left System box.
  bindSystemUI(folder: FolderApi): void {
    this.presetSections.push(
      addSystemPresets(folder, this.presets, () => this.captureThumbnail()),
    );
  }

  // Resolves with a JPEG data URL of the scene after the next rendered frame.
  captureThumbnail(): Promise<string> {
    return new Promise((resolve) => {
      this.thumbRequest = resolve;
    });
  }

  bindPostUI(folder: FolderApi): void {
    this.presetSections.push({ dispose: this.postStack.bindUI(folder, this.modulator) });
    // Below the per-effect folders, so the section reads as "presets for the
    // whole post chain" — it captures every post.* param, effect enables
    // included.
    this.presetSections.push(
      addPresetSection(folder, this.presets, { id: "post", prefixes: ["post"] }),
    );
  }

  bindEnvironmentUI(folder: FolderApi): void {
    const proxies: ParamProxyRegistry = new Map();
    this.environment.bindUI(folder, this.modulator, proxies);
    // Gated on source==='user' so per-frame modulator notifies don't jitter
    // the UI; matches the other bind methods.
    const unsub = this.deps.paramStore.subscribe((key, _value, source) => {
      if (source !== "user") return;
      const refresh = proxies.get(key);
      if (!refresh) return;
      refresh();
      folder.refresh();
    });
    this.presetSections.push({ dispose: unsub });
    this.presetSections.push(
      addPresetSection(folder, this.presets, { id: "environment", prefixes: ["env"] }),
    );
  }

  bindCameraUI(folder: FolderApi): void {
    const store = this.deps.paramStore;
    const camera = this.rig.camera;

    const fovSchema = store.schemaFor("camera.fov");
    const presetSchema = store.schemaFor("camera.preset");
    if (!fovSchema || !presetSchema) {
      throw new Error("bindCameraUI: required schemas missing");
    }
    const rotateSchema = store.schemaFor("camera.rotate");
    if (!rotateSchema) throw new Error("bindCameraUI: camera.rotate schema missing");
    // Collects "re-pull from store" callbacks so a preset load moves the
    // widgets instead of leaving them stale; driven from the subscriber below.
    const proxies: ParamProxyRegistry = new Map();
    bindParam(folder, store, this.modulator, fovSchema, proxies);
    bindParam(folder, store, this.modulator, presetSchema, proxies);
    bindParam(folder, store, this.modulator, rotateSchema, proxies);
    for (const key of [
      "light.directional.intensity",
      "light.directional.color",
      "light.directional.azimuth",
      "light.directional.elevation",
      "light.ambient.intensity",
      "light.ambient.color",
    ]) {
      const schema = store.schemaFor(key);
      if (!schema) throw new Error(`bindCameraUI: ${key} schema missing`);
      bindParam(folder, store, this.modulator, schema, proxies);
    }

    const swimKeys = [
      "camera.swim.enabled",
      "camera.swim.posRoughness",
      "camera.swim.posAmplitude",
      "camera.swim.rotRoughness",
      "camera.swim.rotAmplitude",
    ];
    for (const key of swimKeys) {
      const schema = store.schemaFor(key);
      if (!schema) throw new Error(`bindCameraUI: ${key} schema missing`);
      bindParam(folder, store, this.modulator, schema, proxies);
    }
    const applySwim = () => this.rig.setSwim({
      enabled:      store.get("camera.swim.enabled") as boolean,
      posRoughness: store.get("camera.swim.posRoughness") as number,
      posAmplitude: store.get("camera.swim.posAmplitude") as number,
      rotRoughness: store.get("camera.swim.rotRoughness") as number,
      rotAmplitude: (store.get("camera.swim.rotAmplitude") as number) * DEG2RAD,
    });

    // Side-effects subscriber. Continuous modulatable keys (camera.fov,
    // camera.rotate) write through on every notify so the modulator can
    // drive them. preset (discrete) is not modulatable; gated on
    // source==="user" defensively.
    this.cameraUnsub = store.subscribe((key, value, source) => {
      if (key === "camera.fov" && typeof value === "number") {
        camera.fov = value;
        camera.updateProjectionMatrix();
      } else if (key === "camera.preset" && typeof value === "number" && source === "user") {
        const name = CAMERA_PRESET_NAMES[value];
        if (name) void this.rig.goTo(name, { duration: 0.8 });
      } else if (key === "camera.rotate" && typeof value === "number") {
        this.rig.setAutorotate(value * ROTATE_DEG_PER_UNIT);
      } else if (key.startsWith("camera.swim.")) {
        applySwim();
      } else if (key === "light.directional.intensity" && typeof value === "number") {
        // Writes through on every notify (like camera.fov) so the modulator
        // can drive brightness. At 0 the shadow pass is disabled too — no
        // point encoding + sampling a shadow map that contributes nothing.
        // Flipping castShadow rebuilds pipelines (cached after first flip).
        this.directionalLight.intensity = value;
        this.directionalLight.castShadow = value > 0;
      } else if (key === "light.directional.color" && typeof value === "number") {
        this.directionalLight.color.setHex(value);
      } else if (
        (key === "light.directional.azimuth" || key === "light.directional.elevation") &&
        typeof value === "number"
      ) {
        this.applyLightDirection();
      } else if (key === "light.ambient.intensity" && typeof value === "number") {
        this.ambientLight.intensity = value;
      } else if (key === "light.ambient.color" && typeof value === "number") {
        this.ambientLight.color.setHex(value);
      }
      // Widget sync rides along on the same subscription. Gated on
      // source==='user' so per-frame modulator notifies don't jitter the UI.
      if (source !== "user") return;
      const refresh = proxies.get(key);
      if (refresh) {
        refresh();
        folder.refresh();
      }
    });

    // "light" rides along with the camera scope: the directional light toggle
    // is rendered in this folder, so a camera preset should carry it.
    this.presetSections.push(
      addPresetSection(folder, this.presets, { id: "camera", prefixes: ["camera", "light"] }),
    );

    camera.fov = store.get("camera.fov") as number;
    camera.updateProjectionMatrix();
    this.rig.setAutorotate((store.get("camera.rotate") as number) * ROTATE_DEG_PER_UNIT);
    applySwim();
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
    this.environment?.dispose();
    this.rig?.dispose();
    this.perf.unmount();
    this.cameraUnsub?.();
    this.cameraUnsub = null;
    for (const s of this.presetSections) s.dispose();
    this.presetSections = [];
    this.deps.workletNode.port.onmessage = null;
  }
}
