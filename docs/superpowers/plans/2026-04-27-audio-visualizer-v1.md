# Audio Visualizer v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 pipeline-proof of the audio visualizer: mic in → AudioWorklet → Rust/WASM passthrough → main thread → `LineRenderer` showing a live waveform in a Three.js WebGPU scene with a two-preset camera rig.

**Architecture:** Three layers — Rust/WASM DSP (passthrough in v1), AudioWorkletProcessor that bridges Web Audio to WASM and posts feature buffers via transferable `Float32Array`s, and a Three.js + WebGPURenderer main thread that maintains a `FeatureStore`, a `CameraRig`, and a generic `LineRenderer`. Pure-logic components (CameraRig, LineRenderer, FeatureStore) are unit-tested via Vitest; browser integrations (mic, worklet, WASM link, WebGPU rendering) have explicit manual verification gates.

**Tech Stack:** TypeScript + Vite, Three.js with `WebGPURenderer`, Rust compiled via `wasm-pack`, `vite-plugin-wasm`, `vite-plugin-top-level-await`, AudioWorklet, Vitest with `happy-dom`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `vite.config.ts`
- Create: `index.html`
- Create: `src/main.ts`
- Create: `.gitignore`

- [ ] **Step 1: Write `.gitignore`**

```
node_modules
dist
target
pkg
.DS_Store
*.log
.vite
```

- [ ] **Step 2: Write `package.json`**

```json
{
  "name": "autocorrelation",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "wasm": "cd crates/dsp && wasm-pack build --target web --out-dir ../../src/wasm-pkg"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "happy-dom": "^15.0.0",
    "typescript": "^5.6.0",
    "vite": "^5.4.0",
    "vite-plugin-top-level-await": "^1.4.4",
    "vite-plugin-wasm": "^3.3.0",
    "vitest": "^2.0.0"
  },
  "dependencies": {
    "three": "^0.170.0",
    "@types/three": "^0.170.0"
  }
}
```

- [ ] **Step 3: Write `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "esModuleInterop": true,
    "isolatedModules": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,
    "types": ["vite/client"]
  },
  "include": ["src", "tests", "vite.config.ts", "vitest.config.ts"]
}
```

- [ ] **Step 4: Write `vite.config.ts`**

```ts
import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  worker: {
    plugins: () => [wasm(), topLevelAwait()],
  },
  server: {
    port: 5173,
  },
});
```

- [ ] **Step 5: Write `index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>autocorrelation</title>
    <style>
      html, body { margin: 0; padding: 0; height: 100%; background: #0a0a0a; color: #ddd; font-family: ui-sans-serif, system-ui; }
      #app { width: 100vw; height: 100vh; display: block; }
      #start { position: fixed; top: 1rem; left: 1rem; padding: 0.5rem 1rem; background: #222; color: #ddd; border: 1px solid #444; cursor: pointer; }
    </style>
  </head>
  <body>
    <canvas id="app"></canvas>
    <button id="start">Start</button>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 6: Write `src/main.ts` (placeholder)**

```ts
console.log("autocorrelation: bootstrap");
```

- [ ] **Step 7: Install and verify**

Run: `npm install`
Run: `npm run dev`

Expected: Vite serves on `http://localhost:5173`. Page loads with a black background and "Start" button. Console shows `autocorrelation: bootstrap`. No errors.

Stop the dev server (`Ctrl+C`).

- [ ] **Step 8: Commit**

```bash
git add .gitignore package.json tsconfig.json vite.config.ts index.html src/main.ts package-lock.json
git commit -m "chore: scaffold Vite + TS project"
```

---

## Task 2: Vitest setup

**Files:**
- Create: `vitest.config.ts`
- Create: `tests/sanity.test.ts`

- [ ] **Step 1: Write `vitest.config.ts`**

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "happy-dom",
    include: ["tests/**/*.test.ts"],
  },
});
```

- [ ] **Step 2: Write `tests/sanity.test.ts`**

```ts
import { describe, it, expect } from "vitest";

describe("sanity", () => {
  it("runs", () => {
    expect(1 + 1).toBe(2);
  });
});
```

- [ ] **Step 3: Run tests**

Run: `npm test`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add vitest.config.ts tests/sanity.test.ts
git commit -m "chore: add Vitest with happy-dom"
```

---

## Task 3: CameraRig — presets and immediate goTo

**Files:**
- Create: `src/render/CameraRig.ts`
- Create: `tests/render/CameraRig.test.ts`

This task implements `addPreset` and instant (duration=0) `goTo`. Tween logic comes in Task 4.

- [ ] **Step 1: Write the failing test**

```ts
// tests/render/CameraRig.test.ts
import { describe, it, expect } from "vitest";
import { PerspectiveCamera, Vector3 } from "three";
import { CameraRig } from "../../src/render/CameraRig";

describe("CameraRig", () => {
  const makeRig = () => new CameraRig(new PerspectiveCamera(60, 1, 0.1, 100));

  it("registers and immediately moves to a preset with duration 0", async () => {
    const rig = makeRig();
    rig.addPreset("front", {
      position: new Vector3(0, 0, 5),
      target: new Vector3(0, 0, 0),
    });

    await rig.goTo("front", { duration: 0 });

    expect(rig.camera.position.x).toBeCloseTo(0);
    expect(rig.camera.position.y).toBeCloseTo(0);
    expect(rig.camera.position.z).toBeCloseTo(5);
  });

  it("throws when goTo references an unknown preset", async () => {
    const rig = makeRig();
    await expect(rig.goTo("nope", { duration: 0 })).rejects.toThrow(/unknown preset/i);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test -- CameraRig`
Expected: FAIL — `Cannot find module '../../src/render/CameraRig'`.

- [ ] **Step 3: Write minimal implementation**

```ts
// src/render/CameraRig.ts
import { PerspectiveCamera, Vector3 } from "three";

export interface CameraPose {
  position: Vector3;
  target: Vector3;
  fov?: number;
}

export interface GoToOptions {
  duration?: number;
  easing?: (t: number) => number;
}

export type ProceduralController = (dt: number, camera: PerspectiveCamera) => void;

export class CameraRig {
  readonly camera: PerspectiveCamera;
  private presets = new Map<string, CameraPose>();

  constructor(camera: PerspectiveCamera) {
    this.camera = camera;
  }

  addPreset(name: string, pose: CameraPose): void {
    this.presets.set(name, pose);
  }

  async goTo(name: string, opts: GoToOptions = {}): Promise<void> {
    const pose = this.presets.get(name);
    if (!pose) throw new Error(`unknown preset: ${name}`);

    const duration = opts.duration ?? 0;
    if (duration === 0) {
      this.applyPose(pose);
      return;
    }
    // Tween implementation lands in Task 4.
    throw new Error("tweening not yet implemented");
  }

  setProceduralController(_fn: ProceduralController | null): void {
    // Implemented in Task 4.
  }

  update(_dt: number): void {
    // Implemented in Task 4.
  }

  private applyPose(pose: CameraPose): void {
    this.camera.position.copy(pose.position);
    this.camera.lookAt(pose.target);
    if (pose.fov !== undefined) {
      this.camera.fov = pose.fov;
      this.camera.updateProjectionMatrix();
    }
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm test -- CameraRig`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/render/CameraRig.ts tests/render/CameraRig.test.ts
git commit -m "feat(render): CameraRig with preset registration and instant goTo"
```

---

## Task 4: CameraRig — tween and procedural controller

**Files:**
- Modify: `src/render/CameraRig.ts`
- Modify: `tests/render/CameraRig.test.ts`

- [ ] **Step 1: Add failing tests for tween and procedural**

Append to `tests/render/CameraRig.test.ts` inside the `describe`:

```ts
  it("tweens between presets when duration > 0", async () => {
    const rig = makeRig();
    rig.addPreset("front", { position: new Vector3(0, 0, 5), target: new Vector3() });
    rig.addPreset("side", { position: new Vector3(5, 0, 0), target: new Vector3() });

    await rig.goTo("front", { duration: 0 });

    const tween = rig.goTo("side", { duration: 1 });
    rig.update(0.5); // halfway with linear easing
    expect(rig.camera.position.x).toBeGreaterThan(0);
    expect(rig.camera.position.x).toBeLessThan(5);
    rig.update(0.5); // arrive
    await tween;
    expect(rig.camera.position.x).toBeCloseTo(5, 4);
    expect(rig.camera.position.z).toBeCloseTo(0, 4);
  });

  it("invokes the procedural controller on update when set", () => {
    const rig = makeRig();
    rig.addPreset("front", { position: new Vector3(0, 0, 5), target: new Vector3() });
    let called = 0;
    rig.setProceduralController((dt) => {
      called += 1;
      rig.camera.position.x += dt;
    });
    rig.update(0.1);
    rig.update(0.1);
    expect(called).toBe(2);
    expect(rig.camera.position.x).toBeCloseTo(0.2, 4);
  });

  it("clearing the procedural controller stops calls", () => {
    const rig = makeRig();
    let called = 0;
    rig.setProceduralController(() => { called += 1; });
    rig.update(0.1);
    rig.setProceduralController(null);
    rig.update(0.1);
    expect(called).toBe(1);
  });

  it("starting a tween clears any active procedural controller", async () => {
    const rig = makeRig();
    rig.addPreset("front", { position: new Vector3(0, 0, 5), target: new Vector3() });
    rig.addPreset("side", { position: new Vector3(5, 0, 0), target: new Vector3() });
    await rig.goTo("front", { duration: 0 });
    let proceduralRan = false;
    rig.setProceduralController(() => { proceduralRan = true; });
    const tween = rig.goTo("side", { duration: 1 });
    rig.update(0.5);
    rig.update(0.5);
    await tween;
    proceduralRan = false;
    rig.update(0.1);
    expect(proceduralRan).toBe(false);
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- CameraRig`
Expected: 4 new tests fail (tween throws "not yet implemented"; procedural is a no-op).

- [ ] **Step 3: Replace `src/render/CameraRig.ts` with full implementation**

```ts
import { PerspectiveCamera, Vector3 } from "three";

export interface CameraPose {
  position: Vector3;
  target: Vector3;
  fov?: number;
}

export interface GoToOptions {
  duration?: number;
  easing?: (t: number) => number;
}

export type ProceduralController = (dt: number, camera: PerspectiveCamera) => void;

interface ActiveTween {
  from: CameraPose;
  to: CameraPose;
  elapsed: number;
  duration: number;
  easing: (t: number) => number;
  resolve: () => void;
}

const linearEase = (t: number) => t;

export class CameraRig {
  readonly camera: PerspectiveCamera;
  private presets = new Map<string, CameraPose>();
  private currentTarget = new Vector3();
  private tween: ActiveTween | null = null;
  private procedural: ProceduralController | null = null;

  constructor(camera: PerspectiveCamera) {
    this.camera = camera;
  }

  addPreset(name: string, pose: CameraPose): void {
    this.presets.set(name, pose);
  }

  async goTo(name: string, opts: GoToOptions = {}): Promise<void> {
    const target = this.presets.get(name);
    if (!target) throw new Error(`unknown preset: ${name}`);

    const duration = opts.duration ?? 0;
    if (duration === 0) {
      this.applyPose(target);
      return;
    }

    this.procedural = null;
    const from: CameraPose = {
      position: this.camera.position.clone(),
      target: this.currentTarget.clone(),
      fov: this.camera.fov,
    };

    return new Promise<void>((resolve) => {
      this.tween = {
        from,
        to: target,
        elapsed: 0,
        duration,
        easing: opts.easing ?? linearEase,
        resolve,
      };
    });
  }

  setProceduralController(fn: ProceduralController | null): void {
    this.procedural = fn;
  }

  update(dt: number): void {
    if (this.tween) {
      this.tween.elapsed += dt;
      const raw = Math.min(1, this.tween.elapsed / this.tween.duration);
      const t = this.tween.easing(raw);
      this.camera.position.lerpVectors(this.tween.from.position, this.tween.to.position, t);
      const tgt = new Vector3().lerpVectors(this.tween.from.target, this.tween.to.target, t);
      this.camera.lookAt(tgt);
      this.currentTarget.copy(tgt);
      if (this.tween.from.fov !== undefined && this.tween.to.fov !== undefined) {
        this.camera.fov = this.tween.from.fov + (this.tween.to.fov - this.tween.from.fov) * t;
        this.camera.updateProjectionMatrix();
      }
      if (raw >= 1) {
        const resolve = this.tween.resolve;
        this.tween = null;
        resolve();
      }
      return;
    }

    if (this.procedural) {
      this.procedural(dt, this.camera);
    }
  }

  private applyPose(pose: CameraPose): void {
    this.camera.position.copy(pose.position);
    this.camera.lookAt(pose.target);
    this.currentTarget.copy(pose.target);
    if (pose.fov !== undefined) {
      this.camera.fov = pose.fov;
      this.camera.updateProjectionMatrix();
    }
  }
}
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `npm test -- CameraRig`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/render/CameraRig.ts tests/render/CameraRig.test.ts
git commit -m "feat(render): CameraRig tween + procedural controller"
```

---

## Task 5: LineRenderer

**Files:**
- Create: `src/render/LineRenderer.ts`
- Create: `tests/render/LineRenderer.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// tests/render/LineRenderer.test.ts
import { describe, it, expect } from "vitest";
import { BufferGeometry, Line, Vector3 } from "three";
import { LineRenderer } from "../../src/render/LineRenderer";

describe("LineRenderer", () => {
  it("produces a Line with N positions for a Float32Array of length N", () => {
    const data = new Float32Array(8);
    const lr = new LineRenderer({ source: () => data });
    expect(lr.object3d).toBeInstanceOf(Line);
    const geom = (lr.object3d as Line).geometry as BufferGeometry;
    const pos = geom.getAttribute("position");
    expect(pos.count).toBe(8);
  });

  it("default layout maps i in [0, n-1] to x in [-1, 1] and value to y", () => {
    const data = new Float32Array([0, 0.5, 1, -1]);
    const lr = new LineRenderer({ source: () => data });
    lr.update();
    const pos = ((lr.object3d as Line).geometry as BufferGeometry).getAttribute("position");
    // i=0 → x=-1, i=3 → x=1, linear in between
    expect(pos.getX(0)).toBeCloseTo(-1);
    expect(pos.getX(3)).toBeCloseTo(1);
    expect(pos.getY(0)).toBeCloseTo(0);
    expect(pos.getY(1)).toBeCloseTo(0.5);
    expect(pos.getY(2)).toBeCloseTo(1);
    expect(pos.getY(3)).toBeCloseTo(-1);
    expect(pos.getZ(0)).toBeCloseTo(0);
  });

  it("custom layout function drives positions", () => {
    const data = new Float32Array([1, 2]);
    const layout = (i: number, _n: number, value: number) =>
      new Vector3(i * 10, value * 100, 7);
    const lr = new LineRenderer({ source: () => data, layout });
    lr.update();
    const pos = ((lr.object3d as Line).geometry as BufferGeometry).getAttribute("position");
    expect(pos.getX(0)).toBe(0);
    expect(pos.getY(0)).toBe(100);
    expect(pos.getZ(0)).toBe(7);
    expect(pos.getX(1)).toBe(10);
    expect(pos.getY(1)).toBe(200);
  });

  it("update() reflects changes to the source", () => {
    const buf = new Float32Array([0, 0]);
    const lr = new LineRenderer({ source: () => buf });
    lr.update();
    buf[0] = 0.9;
    lr.update();
    const pos = ((lr.object3d as Line).geometry as BufferGeometry).getAttribute("position");
    expect(pos.getY(0)).toBeCloseTo(0.9);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test -- LineRenderer`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/render/LineRenderer.ts`**

```ts
import {
  BufferAttribute,
  BufferGeometry,
  ColorRepresentation,
  DynamicDrawUsage,
  Line,
  LineBasicMaterial,
  Object3D,
  Vector3,
} from "three";

export type LineLayoutFn = (i: number, n: number, value: number) => Vector3;

export interface LineRendererOptions {
  source: () => Float32Array;
  layout?: LineLayoutFn;
  color?: ColorRepresentation;
}

const defaultLayout: LineLayoutFn = (i, n, value) => {
  const x = n <= 1 ? 0 : (i / (n - 1)) * 2 - 1;
  return new Vector3(x, value, 0);
};

export class LineRenderer {
  readonly object3d: Object3D;
  private source: () => Float32Array;
  private layout: LineLayoutFn;
  private positions: Float32Array;
  private positionAttribute: BufferAttribute;
  private lastLength = -1;

  constructor(opts: LineRendererOptions) {
    this.source = opts.source;
    this.layout = opts.layout ?? defaultLayout;

    const initial = this.source();
    this.positions = new Float32Array(initial.length * 3);
    this.positionAttribute = new BufferAttribute(this.positions, 3);
    this.positionAttribute.setUsage(DynamicDrawUsage);

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", this.positionAttribute);

    const material = new LineBasicMaterial({ color: opts.color ?? 0xffffff });
    this.object3d = new Line(geometry, material);

    this.writeFromSource(initial);
  }

  update(): void {
    const buf = this.source();
    if (buf.length !== this.lastLength) {
      this.positions = new Float32Array(buf.length * 3);
      this.positionAttribute = new BufferAttribute(this.positions, 3);
      this.positionAttribute.setUsage(DynamicDrawUsage);
      ((this.object3d as Line).geometry as BufferGeometry).setAttribute(
        "position",
        this.positionAttribute,
      );
    }
    this.writeFromSource(buf);
  }

  private writeFromSource(buf: Float32Array): void {
    const n = buf.length;
    for (let i = 0; i < n; i++) {
      const v = this.layout(i, n, buf[i]);
      this.positions[i * 3] = v.x;
      this.positions[i * 3 + 1] = v.y;
      this.positions[i * 3 + 2] = v.z;
    }
    this.positionAttribute.needsUpdate = true;
    this.lastLength = n;
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npm test -- LineRenderer`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/render/LineRenderer.ts tests/render/LineRenderer.test.ts
git commit -m "feat(render): LineRenderer with pluggable layout"
```

---

## Task 6: Scene + WebGPURenderer + integration with synthetic data

This task wires `CameraRig` and `LineRenderer` into a real Three.js scene, drives them from a synthetic sine wave (no audio yet), and adds the keyboard preset toggle. This proves the rendering pipeline before audio is added.

**Files:**
- Create: `src/render/Scene.ts`
- Modify: `src/main.ts`
- Modify: `src/App.ts` (new file effectively)
- Create: `src/App.ts`

- [ ] **Step 1: Write `src/render/Scene.ts`**

```ts
import { Color, PerspectiveCamera, Scene as ThreeScene } from "three";
import { WebGPURenderer } from "three/webgpu";

export interface SceneBundle {
  scene: ThreeScene;
  camera: PerspectiveCamera;
  renderer: WebGPURenderer;
}

export async function createScene(canvas: HTMLCanvasElement): Promise<SceneBundle> {
  const renderer = new WebGPURenderer({ canvas, antialias: true });
  await renderer.init();
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(new Color(0x0a0a0a), 1);

  const scene = new ThreeScene();
  const camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);

  window.addEventListener("resize", () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  });

  return { scene, camera, renderer };
}
```

- [ ] **Step 2: Write `src/App.ts`**

```ts
import { Vector3 } from "three";
import { createScene } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { LineRenderer } from "./render/LineRenderer";

export class App {
  private rig!: CameraRig;
  private line!: LineRenderer;
  private buffer = new Float32Array(2048);
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

    this.line = new LineRenderer({
      source: () => this.buffer,
      color: 0x66ffcc,
    });
    scene.add(this.line.object3d);

    let toggled = false;
    window.addEventListener("keydown", (e) => {
      if (e.key !== " ") return;
      toggled = !toggled;
      this.rig.goTo(toggled ? "side" : "front", { duration: 0.8 });
    });

    const loop = (now: number) => {
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;

      // Synthetic sine wave for v1 rendering proof.
      const t = now / 1000;
      for (let i = 0; i < this.buffer.length; i++) {
        const phase = (i / this.buffer.length) * Math.PI * 4;
        this.buffer[i] = 0.5 * Math.sin(phase + t * 2);
      }

      this.rig.update(dt);
      this.line.update();
      renderer.render(scene, camera);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
```

- [ ] **Step 3: Replace `src/main.ts`**

```ts
import { App } from "./App";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Running…";
  const app = new App();
  await app.start(canvas);
});
```

- [ ] **Step 4: Manual verification**

Run: `npm run dev`

Open `http://localhost:5173` in a WebGPU-capable browser (Chrome/Edge ≥ 113, Safari ≥ 18, Firefox with WebGPU enabled).

Expected:
- Page loads, dark canvas, "Start" button visible.
- Click "Start": canvas shows an animated sine wave drawn as a horizontal line, scrolling as time advances.
- Press Space: camera tweens smoothly to the side preset; press again, returns to front.
- No console errors.

If WebGPU isn't available, the renderer will throw — note this in the console and confirm the browser supports WebGPU.

Stop the dev server.

- [ ] **Step 5: Commit**

```bash
git add src/render/Scene.ts src/App.ts src/main.ts
git commit -m "feat: WebGPU scene with synthetic-source LineRenderer + camera toggle"
```

---

## Task 7: Rust DSP crate (passthrough)

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/dsp/Cargo.toml`
- Create: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Verify `wasm-pack` is installed**

Run: `wasm-pack --version`

If not installed: `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh` and re-run. Document the version observed.

- [ ] **Step 2: Write workspace `Cargo.toml`**

```toml
[workspace]
members = ["crates/dsp"]
resolver = "2"
```

- [ ] **Step 3: Write `crates/dsp/Cargo.toml`**

```toml
[package]
name = "dsp"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"

[profile.release]
opt-level = 3
lto = true
```

- [ ] **Step 4: Write `crates/dsp/src/lib.rs`**

```rust
use wasm_bindgen::prelude::*;

/// v1: passthrough. Copies the input window into the output buffer.
/// Future versions will compute FFT, autocorrelation, RMS, beat phase.
#[wasm_bindgen]
pub struct Dsp {
    out: Vec<f32>,
}

#[wasm_bindgen]
impl Dsp {
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize) -> Dsp {
        Dsp { out: vec![0.0; window_size] }
    }

    /// Process one analysis window. Returns a borrowed view of the
    /// internal output buffer; the caller must copy before the next call.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let n = input.len().min(self.out.len());
        self.out[..n].copy_from_slice(&input[..n]);
        self.out[..n].to_vec()
    }
}
```

- [ ] **Step 5: Build the WASM module**

Run: `npm run wasm`

Expected: `src/wasm-pkg/` directory created containing `dsp.js`, `dsp_bg.wasm`, `dsp.d.ts`, `package.json`. No build errors.

- [ ] **Step 6: Add `src/wasm-pkg/` to `.gitignore`**

Append to `.gitignore`:
```
src/wasm-pkg
```

(WASM artifacts are build outputs, not source.)

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates .gitignore
git commit -m "feat(dsp): Rust DSP crate with passthrough process()"
```

---

## Task 8: WASM linkage from main thread (verify import works)

**Files:**
- Modify: `src/App.ts`

This is a verification task: confirm Vite resolves the WASM module before we put it inside the worklet. We'll temporarily call `Dsp` from the main thread, then move it into the worklet in Task 11.

- [ ] **Step 1: Modify `src/App.ts` `start()` to import and call Dsp**

Add at the top of the file:

```ts
import init, { Dsp } from "../src/wasm-pkg/dsp";
```

Note: the import path is relative to the file. From `src/App.ts`, it's:

```ts
import init, { Dsp } from "./wasm-pkg/dsp";
```

Inside `start()`, before the loop, add:

```ts
    await init();
    const dsp = new Dsp(2048);
    const probe = new Float32Array(2048);
    probe[0] = 0.42;
    const out = dsp.process(probe);
    console.log("[dsp] passthrough probe:", out[0], "expected 0.42");
```

- [ ] **Step 2: Build WASM and run dev server**

Run: `npm run wasm && npm run dev`

Expected:
- Page loads.
- Click Start; console shows `[dsp] passthrough probe: 0.42 expected 0.42`.
- Existing sine-wave visualization still works.

Stop the dev server.

- [ ] **Step 3: Remove the probe code**

Delete the four added lines (`await init()` through `console.log`) from `start()`. Keep the `import init, { Dsp } from "./wasm-pkg/dsp"` for now — the worklet will use it differently, but verifying the import path is the goal here. Actually, also delete the import — we'll re-add it in the worklet context.

- [ ] **Step 4: Commit verification result**

```bash
git commit --allow-empty -m "chore: verify WASM resolves via vite-plugin-wasm"
```

(The probe code is removed; this commit just records the milestone.)

---

## Task 9: FeatureStore

**Files:**
- Create: `src/store/FeatureStore.ts`
- Create: `tests/store/FeatureStore.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// tests/store/FeatureStore.test.ts
import { describe, it, expect } from "vitest";
import { FeatureStore } from "../../src/store/FeatureStore";

describe("FeatureStore", () => {
  it("returns an empty Float32Array for an unknown key", () => {
    const store = new FeatureStore();
    expect(store.get("nope").length).toBe(0);
  });

  it("stores and returns the latest buffer for a key", () => {
    const store = new FeatureStore();
    const a = new Float32Array([1, 2, 3]);
    store.set("waveform", a);
    expect(store.get("waveform")).toBe(a);
  });

  it("set replaces the previous buffer", () => {
    const store = new FeatureStore();
    store.set("waveform", new Float32Array([1]));
    const second = new Float32Array([9, 9]);
    store.set("waveform", second);
    expect(store.get("waveform")).toBe(second);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test -- FeatureStore`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/store/FeatureStore.ts`**

```ts
const EMPTY = new Float32Array(0);

export class FeatureStore {
  private buffers = new Map<string, Float32Array>();

  set(key: string, buf: Float32Array): void {
    this.buffers.set(key, buf);
  }

  get(key: string): Float32Array {
    return this.buffers.get(key) ?? EMPTY;
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npm test -- FeatureStore`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/store/FeatureStore.ts tests/store/FeatureStore.test.ts
git commit -m "feat(store): FeatureStore for latest feature buffers"
```

---

## Task 10: AudioSource (mic)

**Files:**
- Create: `src/audio/AudioSource.ts`

This wraps `getUserMedia` and creates an `AudioContext`. Pure browser integration — verified manually.

- [ ] **Step 1: Write `src/audio/AudioSource.ts`**

```ts
export interface AudioSourceBundle {
  context: AudioContext;
  source: MediaStreamAudioSourceNode;
  stream: MediaStream;
}

export async function createMicSource(): Promise<AudioSourceBundle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
    video: false,
  });

  const context = new AudioContext();
  if (context.state === "suspended") await context.resume();
  const source = context.createMediaStreamSource(stream);

  return { context, source, stream };
}
```

- [ ] **Step 2: Manual verification (temporary probe)**

In `src/App.ts`, inside `start()`, after `await this.rig.goTo("front", { duration: 0 })`, add temporarily:

```ts
    const { context, source } = await (await import("./audio/AudioSource")).createMicSource();
    console.log("[audio] context state:", context.state, "sampleRate:", context.sampleRate);
    console.log("[audio] source channels:", source.channelCount);
```

Run: `npm run dev`. Click Start; allow mic permission.

Expected console:
- `[audio] context state: running sampleRate: 44100` (or 48000)
- `[audio] source channels: 1` (or 2)

- [ ] **Step 3: Remove the probe lines from `App.ts`**

Delete the three temporary lines.

- [ ] **Step 4: Commit**

```bash
git add src/audio/AudioSource.ts
git commit -m "feat(audio): mic-based AudioSource"
```

---

## Task 11: DSPWorklet (passthrough, no WASM yet)

**Files:**
- Create: `src/audio/dsp-worklet.ts`
- Modify: `src/App.ts`

The worklet runs in a separate JS context (`AudioWorkletGlobalScope`) and is loaded via `audioWorklet.addModule()`. Vite handles the import via `?worker&url` syntax.

- [ ] **Step 1: Write `src/audio/dsp-worklet.ts`**

```ts
/// <reference types="@types/audioworklet" />

const WINDOW_SIZE = 2048;

class DSPProcessor extends AudioWorkletProcessor {
  private window = new Float32Array(WINDOW_SIZE);
  private filled = 0;

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    let i = 0;
    while (i < channel.length) {
      const space = WINDOW_SIZE - this.filled;
      const take = Math.min(space, channel.length - i);
      this.window.set(channel.subarray(i, i + take), this.filled);
      this.filled += take;
      i += take;

      if (this.filled === WINDOW_SIZE) {
        const out = new Float32Array(this.window); // copy
        this.port.postMessage({ type: "waveform", buffer: out }, [out.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
```

If the `@types/audioworklet` package isn't installed, add it:

Run: `npm install --save-dev @types/audioworklet`

- [ ] **Step 2: Wire worklet into `App.ts`**

Modify `src/App.ts`. Add an import at the top:

```ts
import dspWorkletUrl from "./audio/dsp-worklet?worker&url";
import { createMicSource } from "./audio/AudioSource";
import { FeatureStore } from "./store/FeatureStore";
```

Add fields to the class (next to existing `buffer`, `last`):

```ts
  private store = new FeatureStore();
```

Replace the existing buffer initialization. The `App.start()` should now:
1. Set up the scene + rig + line renderer (unchanged).
2. Bind the line renderer's `source` to `() => this.store.get("waveform")`.
3. Start the audio source and worklet.
4. The render loop no longer generates synthetic data.

Delete the existing render loop (the entire `const loop = …; requestAnimationFrame(loop)` block from Task 6) and replace it with the audio setup plus a new render loop:

```ts
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
```

Also change `LineRenderer` construction to read from the store:

```ts
    this.line = new LineRenderer({
      source: () => this.store.get("waveform"),
      color: 0x66ffcc,
    });
```

Remove the `private buffer = new Float32Array(2048)` field; it's no longer used.

`LineRenderer.update()` already handles a zero-length buffer at boot — verify by reading `LineRenderer.writeFromSource`: it loops `for i < n`, which is `0` initially, so nothing happens. Good.

- [ ] **Step 3: Manual verification**

Run: `npm run dev`

Open browser, click Start, allow mic permission.

Expected:
- A line renders that visibly moves in response to mic input (talk, clap, play music near the mic).
- When the room is quiet, the line is near-flat at y=0.
- Camera toggle (Space) still works.
- No console errors.

Stop the dev server.

- [ ] **Step 4: Commit**

```bash
git add src/audio/dsp-worklet.ts src/App.ts package.json package-lock.json
git commit -m "feat(audio): DSPWorklet streaming waveform windows to FeatureStore"
```

---

## Task 12: Wire WASM into the worklet

**Files:**
- Modify: `src/audio/dsp-worklet.ts`

The worklet currently posts the raw window. This task pipes the window through the Rust passthrough function so the WASM linkage is exercised end-to-end. Behavior is identical (passthrough) but the architecture is now correct for v2+.

- [ ] **Step 1: Update `src/audio/dsp-worklet.ts` to load and call WASM**

```ts
/// <reference types="@types/audioworklet" />
import init, { Dsp } from "../wasm-pkg/dsp";

const WINDOW_SIZE = 2048;

class DSPProcessor extends AudioWorkletProcessor {
  private window = new Float32Array(WINDOW_SIZE);
  private filled = 0;
  private dsp: Dsp | null = null;
  private ready = false;

  constructor() {
    super();
    this.boot();
  }

  private async boot() {
    await init();
    this.dsp = new Dsp(WINDOW_SIZE);
    this.ready = true;
  }

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    let i = 0;
    while (i < channel.length) {
      const space = WINDOW_SIZE - this.filled;
      const take = Math.min(space, channel.length - i);
      this.window.set(channel.subarray(i, i + take), this.filled);
      this.filled += take;
      i += take;

      if (this.filled === WINDOW_SIZE) {
        if (this.ready && this.dsp) {
          const processed = this.dsp.process(this.window);
          const out = new Float32Array(processed);
          this.port.postMessage({ type: "waveform", buffer: out }, [out.buffer]);
        }
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
```

- [ ] **Step 2: Manual verification**

Run: `npm run wasm && npm run dev`

Open browser, click Start, allow mic.

Expected:
- Same behavior as Task 11 — the line responds to mic input.
- No console errors; in particular no errors about WASM module loading inside the worklet.

If you see "instantiateStreaming requires Response" or similar: this typically means the worklet needs synchronous WASM instantiation. The fallback is to use `WebAssembly.instantiate(bytes)` directly with the bytes fetched ahead of time. If this surfaces, document the symptom and we'll revise — note this is in the spec's "open questions" already.

Stop the dev server.

- [ ] **Step 3: Commit**

```bash
git add src/audio/dsp-worklet.ts
git commit -m "feat(audio): pipe waveform window through Rust/WASM passthrough"
```

---

## Task 13: Final acceptance pass

**Files:** none modified

- [ ] **Step 1: Run all unit tests**

Run: `npm test`

Expected: all tests pass (Sanity 1, CameraRig 6, LineRenderer 4, FeatureStore 3 = 14).

- [ ] **Step 2: Build production bundle**

Run: `npm run build`

Expected: build succeeds, `dist/` directory created. No TypeScript errors.

- [ ] **Step 3: Smoke test the production build**

Run: `npm run preview`

Open the printed URL. Click Start, allow mic. Verify:
- Line renderer shows live mic input.
- Camera toggle (Space) tweens.
- No console errors.

- [ ] **Step 4: v1 acceptance checklist (from the spec)**

Confirm each item against the running app:

- [ ] `npm run dev` boots a Vite app
- [ ] Click "Start" → mic permission → audio flowing
- [ ] WASM module loaded, worklet calls process (passthrough)
- [ ] LineRenderer shows the live waveform
- [ ] CameraRig with two presets, keyboard tween
- [ ] WebGPURenderer rendering at stable 60fps (eyeball; or use browser perf tools)

- [ ] **Step 5: Commit a tag**

```bash
git tag v1.0.0
git log --oneline
```

v1 complete.
