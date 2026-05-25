# Postprocessing Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inline GTAO node-graph in `App.ts` with a modular PostStack of toggleable effects (AO, Bloom, Tonemap/HDR) and add two new top-level panel folders (Camera, Post). Toggling any effect rebuilds the graph so disabled effects drop their MRT requirements.

**Architecture:** A `PostStack` owns a single `PostProcessing` instance plus an ordered list of `PostEffect`s. On any `post.*.enabled` change, it rebuilds `outputNode`: rebuilds the scene pass with MRT only for `needs.normal` requested by enabled effects, then chains `effect.build(input, ctx)` calls in canonical order. Slider params feed TSL `uniform()` refs the effects own (hot, no rebuild). Each effect subscribes its own non-topology keys.

**Tech Stack:** TypeScript, Three.js 0.170 (`three/webgpu`, `three/tsl`, `three/addons/tsl/display/BloomNode.js`), Tweakpane, Vite. WebGPU renderer with TSL node graphs.

**Spec:** `docs/superpowers/specs/2026-05-24-postprocessing-stack-design.md`

**Manual verification only.** User will test in browser; no vitest tests for the post stack.

---

## File map

```
NEW  src/render/post/PostStack.ts        # owns PostProcessing + rebuild
NEW  src/render/post/PostEffect.ts       # interface + PassCtx
NEW  src/render/post/effects/AoEffect.ts
NEW  src/render/post/effects/TonemapEffect.ts
NEW  src/render/post/effects/BloomEffect.ts
NEW  src/render/post/index.ts            # POST_EFFECTS = canonical list
NEW  src/params/postSchemas.ts
NEW  src/params/cameraSchemas.ts
MOVE src/render/GTAONode.js              → src/render/post/GTAONode.js
EDIT src/App.ts                          # replace inline graph; add bindCameraUI/bindPostUI
EDIT src/main.ts                         # register new schemas; wire panel.camera, panel.post
EDIT src/params/ParamStore.ts            # add optional optionLabels?: string[] on discrete schemas
EDIT src/params/ParamPanel.ts            # add Camera + Post folders; use optionLabels
```

---

### Task 1: Extend discrete schema with optional `optionLabels`

Tonemap mode and camera preset are integer-coded for string concepts. Without this, the dropdown shows `"0", "1", "2", "3"`. Tiny, isolated change.

**Files:**
- Modify: `src/params/ParamStore.ts:3-12` (extend the `discrete` variant)
- Modify: `src/params/ParamPanel.ts:49-53` (use labels when present)

- [ ] **Step 1: Extend the `discrete` schema variant**

In `src/params/ParamStore.ts`, change the `discrete` variant inside the `ParamSchema` union:

```ts
  | { kind: "discrete"; options: number[]; optionLabels?: string[] }
```

(Just adds `optionLabels?: string[]` to the existing line.)

- [ ] **Step 2: Use labels in ParamPanel when present**

In `src/params/ParamPanel.ts`, replace the `discrete` branch of `addWidget` (around line 49):

```ts
    if (schema.kind === "discrete") {
      const labels = schema.optionLabels ?? schema.options.map(String);
      return folder.addBinding(this.bindings, schema.key, {
        label: schema.label,
        options: Object.fromEntries(schema.options.map((v, i) => [labels[i], v])),
      });
    }
```

- [ ] **Step 3: Verify type check passes**

Run: `npx tsc --noEmit`
Expected: PASS (no errors).

- [ ] **Step 4: Commit**

```bash
git add src/params/ParamStore.ts src/params/ParamPanel.ts
git commit -m "feat(params): optional string labels for discrete dropdowns"
```

---

### Task 2: Add `postSchemas.ts` and `cameraSchemas.ts`, register them

Schemas only — no folders yet. Validates that registration + persistence works before anything reads these keys.

**Files:**
- Create: `src/params/postSchemas.ts`
- Create: `src/params/cameraSchemas.ts`
- Modify: `src/main.ts:158-160` (the `if (!paramStore)` block)

- [ ] **Step 1: Create `src/params/postSchemas.ts`**

```ts
import type { ParamSchema } from "./ParamStore";

export const postSchemas: ParamSchema[] = [
  // ── AO ─────────────────────────────────────────────────────────────
  { key: "post.ao.enabled",      label: "AO on",            kind: "boolean",    default: true,  reconfig: false },
  { key: "post.ao.radius",       label: "AO radius",        kind: "continuous", min: 0.05, max: 2.0, step: 0.05, default: 0.25, reconfig: false },
  { key: "post.ao.intensity",    label: "AO intensity",     kind: "continuous", min: 0.0,  max: 2.0, step: 0.05, default: 1.0,  reconfig: false },

  // ── Bloom ──────────────────────────────────────────────────────────
  { key: "post.bloom.enabled",   label: "Bloom on",         kind: "boolean",    default: false, reconfig: false },
  { key: "post.bloom.strength",  label: "Bloom strength",   kind: "continuous", min: 0.0, max: 3.0,  step: 0.01, default: 0.5,  reconfig: false },
  { key: "post.bloom.radius",    label: "Bloom radius",     kind: "continuous", min: 0.0, max: 1.0,  step: 0.01, default: 0.4,  reconfig: false },
  { key: "post.bloom.threshold", label: "Bloom threshold",  kind: "continuous", min: 0.0, max: 2.0,  step: 0.01, default: 0.85, reconfig: false },

  // ── Tonemap ────────────────────────────────────────────────────────
  { key: "post.tonemap.enabled", label: "Tonemap on",       kind: "boolean",    default: true,  reconfig: false },
  // Integer-coded: 0=None, 1=AgX, 2=ACES, 3=Neutral. Mapped to three's
  // ToneMapping constants inside TonemapEffect via TONEMAP_TABLE.
  { key: "post.tonemap.mode",    label: "Tonemap mode",     kind: "discrete",
    options: [0, 1, 2, 3], optionLabels: ["None", "AgX", "ACES", "Neutral"],
    default: 0, reconfig: false },
  { key: "post.tonemap.exposure",label: "Tonemap exposure", kind: "continuous", min: 0.0, max: 4.0, step: 0.01, default: 1.0, reconfig: false },
];
```

- [ ] **Step 2: Create `src/params/cameraSchemas.ts`**

```ts
import type { ParamSchema } from "./ParamStore";

// Integer-coded; index → preset name. Must stay in sync with
// CAMERA_PRESET_LABELS in App.ts (which feeds these to rig.goTo).
export const cameraSchemas: ParamSchema[] = [
  { key: "camera.fov",    label: "FOV",    kind: "continuous", min: 20, max: 120, step: 1, default: 60, reconfig: false },
  { key: "camera.preset", label: "Preset", kind: "discrete",
    options: [0, 1, 2, 3, 4, 5],
    optionLabels: ["front", "side", "spectrum", "rms", "buffer-acf", "rms-acf"],
    default: 0, reconfig: false },
];
```

- [ ] **Step 3: Register both in `src/main.ts`**

Replace `src/main.ts:158-160` (`for (const s of analysisSchemas) paramStore.register(s);`) with:

```ts
      const { postSchemas } = await import("./params/postSchemas");
      const { cameraSchemas } = await import("./params/cameraSchemas");
      for (const s of analysisSchemas) paramStore.register(s);
      for (const s of postSchemas) paramStore.register(s);
      for (const s of cameraSchemas) paramStore.register(s);
```

(Dynamic imports match the existing pattern of `await import("./audio/AudioSource")` in the mic/tab click handlers — keeps `main.ts`'s top-level imports light.)

- [ ] **Step 4: Verify**

Run: `npx tsc --noEmit`
Expected: PASS.

Run dev server: `npm run dev`. Open the page, start the test source (press any key). Open localStorage in DevTools → `autocorrelation.params.v1` should contain the new `post.*` and `camera.*` keys with their defaults. The panel still shows only the Analysis folder for now (no Camera/Post folder yet — Task 3).

- [ ] **Step 5: Commit**

```bash
git add src/params/postSchemas.ts src/params/cameraSchemas.ts src/main.ts
git commit -m "feat(params): register post.* and camera.* schemas"
```

---

### Task 3: Add `Camera` and `Post` folders to `ParamPanel`

Folders only, empty. Lets us wire panel ownership without depending on PostStack or App.

**Files:**
- Modify: `src/params/ParamPanel.ts:6, 35-37` (add public folders)

- [ ] **Step 1: Add the folders**

In `src/params/ParamPanel.ts`, change the class fields and constructor:

```ts
export class ParamPanel {
  public pane: Pane;
  public scenes: FolderApi;
  public camera: FolderApi;
  public post: FolderApi;
  // ... existing fields
```

In the constructor, replace the `this.scenes = ...` line through to (and including) the `Reset to defaults` button:

```ts
    this.scenes = this.pane.addFolder({ title: "Scenes" });
    this.camera = this.pane.addFolder({ title: "Camera", expanded: false });
    this.post = this.pane.addFolder({ title: "Post", expanded: false });
    this.pane.addButton({ title: "Reset to defaults" }).on("click", () => store.reset());
```

- [ ] **Step 2: Verify**

Run: `npx tsc --noEmit` → PASS.

`npm run dev`, start test source. Panel should now show: Analysis, Scenes, Camera (empty), Post (empty), Reset.

- [ ] **Step 3: Commit**

```bash
git add src/params/ParamPanel.ts
git commit -m "feat(panel): add empty Camera and Post folders"
```

---

### Task 4: Camera folder UI in App (FOV + preset)

Self-contained. Doesn't depend on PostStack. Validates that the new folders + optionLabels + ParamStore subscription path all work end-to-end.

**Files:**
- Modify: `src/App.ts` (add `bindCameraUI`, plus a small `CAMERA_PRESET_NAMES` array; subscribe to `camera.fov` and `camera.preset`)
- Modify: `src/main.ts:131` (call the new binder)

- [ ] **Step 1: Wire `bindCameraUI` in App**

In `src/App.ts`, add the import near the top with the others (after the existing `import type { CameraPose }` line):

```ts
import type { ParamStore } from "./params/ParamStore";
import type { FolderApi } from "tweakpane";
```

(If `ParamStore` is already imported as a type — line 13 has it — skip that.)

Add a top-level constant just above the `App` class:

```ts
// Order MUST match optionLabels in cameraSchemas.ts (`camera.preset`).
const CAMERA_PRESET_NAMES = ["front", "side", "spectrum", "rms", "buffer-acf", "rms-acf"] as const;
```

Add the method to `App` (place it after `bindUI`):

```ts
  bindCameraUI(folder: FolderApi): void {
    const store = this.deps.paramStore;
    const camera = this.rig.camera;

    // FOV: live-write to camera + projection update.
    const fovBinding = { fov: store.get("camera.fov") as number };
    folder
      .addBinding(fovBinding, "fov", { label: "FOV", min: 20, max: 120, step: 1 })
      .on("change", (e: { value: number }) => store.set("camera.fov", e.value));

    // Preset: dropdown → rig.goTo. Stored as integer index.
    const presetBinding = { preset: store.get("camera.preset") as number };
    folder
      .addBinding(presetBinding, "preset", {
        label: "Preset",
        options: Object.fromEntries(CAMERA_PRESET_NAMES.map((name, i) => [name, i])),
      })
      .on("change", (e: { value: number }) => store.set("camera.preset", e.value));

    // Subscribe so persisted-on-load values and external writes apply.
    const unsub = store.subscribe((key, value) => {
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
    this.cameraUnsub = unsub;

    // Apply current persisted values once on bind so reload restores state.
    camera.fov = store.get("camera.fov") as number;
    camera.updateProjectionMatrix();
  }
```

Add the field declaration to `App` near the others (after `private post!: PostProcessing;`):

```ts
  private cameraUnsub: (() => void) | null = null;
```

In `App.dispose()`, add (before `this.deps.workletNode.port.onmessage = null;`):

```ts
    this.cameraUnsub?.();
    this.cameraUnsub = null;
```

- [ ] **Step 2: Call the binder from `main.ts`**

In `src/main.ts:131`, after `app.bindUI(panel.scenes);`, add:

```ts
  app.bindCameraUI(panel.camera);
```

- [ ] **Step 3: Verify**

Run: `npx tsc --noEmit` → PASS.

`npm run dev`, start any source. Open the Camera folder:
- FOV slider scrubs the camera live.
- Preset dropdown shows the six names, selecting one triggers an eased tween.
- Reload — FOV value persists; preset persists but the camera *doesn't* auto-tween on reload (it loads from saved pose first, see App.ts:126-131).
  - **Note:** that's expected. Selecting the preset again triggers the tween.

- [ ] **Step 4: Commit**

```bash
git add src/App.ts src/main.ts
git commit -m "feat(camera): FOV + preset dropdown in Camera panel folder"
```

---

### Task 5: `PostEffect` interface + `PassCtx` type

Pure type declarations. Used by Tasks 6–9.

**Files:**
- Create: `src/render/post/PostEffect.ts`

- [ ] **Step 1: Create the file**

```ts
// src/render/post/PostEffect.ts
import type { PerspectiveCamera, Scene } from "three";
import type { ShaderNodeObject } from "three/tsl";
import type { Node, TextureNode } from "three/webgpu";
import type { ParamStore } from "../../params/ParamStore";
import type { FolderApi } from "tweakpane";

/** Context handed to every effect's build() — the scene-pass texture nodes. */
export interface PassCtx {
  scene: Scene;
  camera: PerspectiveCamera;
  /** View-space normal. NULL if no enabled effect requested it. */
  sceneNormal: ShaderNodeObject<TextureNode> | null;
  /** Always available. */
  sceneDepth: ShaderNodeObject<TextureNode>;
}

export interface PostEffect {
  readonly id: string;          // stable; matches param prefix (e.g. "ao")
  readonly label: string;       // panel folder title
  readonly needs: Readonly<{ normal?: boolean }>;

  /** Read by PostStack at build time. Mutated by the post.<id>.enabled subscription. */
  enabled: boolean;

  /** Build this effect's node chain. `input` is the upstream color node. */
  build(input: Node, ctx: PassCtx): Node;

  /** Register param schemas + uniform subscriptions. Called once at construction. */
  registerParams(store: ParamStore, requestRebuild: () => void): void;

  /** Add UI widgets into the effect's sub-folder under "Post". */
  bindUI(folder: FolderApi, store: ParamStore): void;

  dispose(): void;
}
```

- [ ] **Step 2: Verify**

Run: `npx tsc --noEmit` → PASS.

(File is unreferenced; just confirms the types compile.)

- [ ] **Step 3: Commit**

```bash
git add src/render/post/PostEffect.ts
git commit -m "feat(post): PostEffect interface and PassCtx type"
```

---

### Task 6: `PostStack` skeleton + replace `App.ts` inline graph (passthrough)

Lands the new pipeline behavior-free: the stack with an empty effects list renders scene color straight to screen, no AO. Visuals will regress (no AO) until Task 7 ports AO into the new framework — single short-lived regression by design, keeps tasks atomic.

**Files:**
- Create: `src/render/post/PostStack.ts`
- Create: `src/render/post/index.ts`
- Modify: `src/App.ts:1-5, 65, 86-99, 183, 194-205` (remove inline graph, swap to stack)

- [ ] **Step 1: Create `src/render/post/index.ts`**

```ts
// src/render/post/index.ts
import type { PostEffect } from "./PostEffect";

// Canonical order. Effects are reordered only by editing this list.
// Empty for now — populated in subsequent tasks.
export const POST_EFFECTS: PostEffect[] = [];
```

- [ ] **Step 2: Create `src/render/post/PostStack.ts`**

```ts
// src/render/post/PostStack.ts
import { PostProcessing } from "three/webgpu";
import { pass, mrt, output, transformedNormalView } from "three/tsl";
import type { PerspectiveCamera, Scene } from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { ParamStore } from "../../params/ParamStore";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "./PostEffect";

/**
 * Owns a single PostProcessing instance plus an ordered list of effects.
 * Subscribes to ParamStore: any `post.*.enabled` change reads each effect's
 * `enabled` from the store, recomputes MRT requirements, and rebuilds
 * `outputNode`. Topology-changing param keys can also request rebuild via
 * the `requestRebuild` callback passed into `registerParams`.
 *
 * PostProcessing has no dispose(); we keep one instance for the App
 * lifetime and just reassign outputNode + flip needsUpdate on rebuild.
 */
export class PostStack {
  private post: PostProcessing;
  private rebuildScheduled = false;
  private subStore: () => void;
  private effects: PostEffect[];

  constructor(
    private renderer: WebGPURenderer,
    private scene: Scene,
    private camera: PerspectiveCamera,
    private store: ParamStore,
    effects: PostEffect[],
  ) {
    this.effects = effects;
    this.post = new PostProcessing(renderer);

    // Per-effect param subscriptions. Each effect can call requestRebuild()
    // when one of its keys changes graph topology (e.g. tonemap mode).
    for (const effect of this.effects) {
      effect.registerParams(this.store, () => this.scheduleRebuild());
    }

    // Toggle subscription — flips effect.enabled and triggers rebuild.
    this.subStore = this.store.subscribe((key, value) => {
      const m = key.match(/^post\.([^.]+)\.enabled$/);
      if (!m) return;
      const id = m[1];
      const effect = this.effects.find((e) => e.id === id);
      if (!effect) return;
      if (typeof value !== "boolean") return;
      effect.enabled = value;
      this.scheduleRebuild();
    });

    // Seed effect.enabled from current store values.
    for (const effect of this.effects) {
      effect.enabled = this.store.get(`post.${effect.id}.enabled`) as boolean;
    }
  }

  build(): void {
    const enabled = this.effects.filter((e) => e.enabled);
    const needsNormal = enabled.some((e) => e.needs.normal);

    const scenePass = pass(this.scene, this.camera);
    scenePass.setMRT(
      needsNormal
        ? mrt({ output, normal: transformedNormalView })
        : mrt({ output }),
    );

    const sceneColor = scenePass.getTextureNode("output");
    const sceneNormal = needsNormal ? scenePass.getTextureNode("normal") : null;
    const sceneDepth = scenePass.getTextureNode("depth");

    const ctx: PassCtx = {
      scene: this.scene,
      camera: this.camera,
      sceneNormal,
      sceneDepth,
    };

    let node = sceneColor;
    for (const effect of enabled) node = effect.build(node, ctx);

    this.post.outputNode = node;
    this.post.needsUpdate = true;
  }

  bindUI(folder: FolderApi): void {
    for (const effect of this.effects) {
      const sub = folder.addFolder({ title: effect.label, expanded: false });
      effect.bindUI(sub, this.store);
    }
  }

  async renderAsync(): Promise<void> {
    await this.post.renderAsync();
  }

  dispose(): void {
    this.subStore();
    for (const effect of this.effects) effect.dispose();
    // PostProcessing has no dispose. The internal QuadMesh is a module
    // singleton; nothing to free per-instance.
  }

  private scheduleRebuild(): void {
    if (this.rebuildScheduled) return;
    this.rebuildScheduled = true;
    queueMicrotask(() => {
      this.rebuildScheduled = false;
      this.build();
    });
  }
}
```

- [ ] **Step 3: Replace `App.ts` inline graph**

In `src/App.ts`, remove the three legacy postprocessing imports at the top:

```ts
import { PostProcessing } from "three/webgpu";
import { pass, mrt, output, transformedNormalView } from "three/tsl";
// @ts-expect-error - local copy of three's GTAONode example, no .d.ts
import { ao } from "./render/GTAONode.js";
```

Add the new import:

```ts
import { PostStack } from "./render/post/PostStack";
import { POST_EFFECTS } from "./render/post";
```

Replace the field `private post!: PostProcessing;` (line 65) with:

```ts
  private postStack!: PostStack;
```

Replace the entire post-setup block (lines 86–99, the comment plus the seven assignments through `this.post.outputNode = sceneColor.mul(aoNode);`) with:

```ts
    this.postStack = new PostStack(renderer, scene, camera, paramStore, POST_EFFECTS);
    this.postStack.build();
```

Replace the render call `void this.post.renderAsync();` (line 183) with:

```ts
      void this.postStack.renderAsync();
```

In `dispose()`, replace nothing (PostStack.dispose call is added now):

```ts
    this.postStack?.dispose();
```

(Place it just before `this.rig?.dispose();`.)

Add the post-folder binder method to `App` (next to `bindCameraUI`):

```ts
  bindPostUI(folder: FolderApi): void {
    this.postStack.bindUI(folder);
  }
```

- [ ] **Step 4: Wire it in `main.ts`**

After `app.bindCameraUI(panel.camera);` (added in Task 4), add:

```ts
  app.bindPostUI(panel.post);
```

- [ ] **Step 5: Verify**

Run: `npx tsc --noEmit` → PASS.

`npm run dev`, start test source.
- Scene renders with **no AO** (regression — temporary; restored in Task 7).
- Panel's Post folder is empty (no effects in `POST_EFFECTS` yet).
- DevTools console: no errors.
- Reload: still renders. No localStorage corruption.

- [ ] **Step 6: Commit**

```bash
git add src/render/post/PostStack.ts src/render/post/index.ts src/App.ts src/main.ts
git commit -m "feat(post): PostStack skeleton (passthrough; AO regression intentional, restored next task)"
```

---

### Task 7: Migrate AO into the stack as `AoEffect`

Restores AO. First real effect — exercises `needs.normal: true`, slider-driven uniforms, and per-effect UI binding.

**Files:**
- Move (with `git mv`): `src/render/GTAONode.js` → `src/render/post/GTAONode.js`
- Create: `src/render/post/effects/AoEffect.ts`
- Modify: `src/render/post/index.ts` (add to `POST_EFFECTS`)

- [ ] **Step 1: Move GTAONode**

```bash
git mv src/render/GTAONode.js src/render/post/GTAONode.js
```

- [ ] **Step 2: Create `src/render/post/effects/AoEffect.ts`**

```ts
// src/render/post/effects/AoEffect.ts
import { uniform } from "three/tsl";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
// @ts-expect-error - local copy of three's GTAONode example, no .d.ts
import { ao } from "../GTAONode.js";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";

export class AoEffect implements PostEffect {
  readonly id = "ao";
  readonly label = "AO";
  readonly needs = { normal: true } as const;
  enabled = true;

  // Live aoNode handle (rebuilt on each PostStack.build). The radius
  // subscription writes into this — guarded with optional chaining
  // because the subscription can fire before the first build().
  private aoNode: { radius: { value: number } } | null = null;
  private intensityU = uniform(1.0);

  registerParams(store: ParamStore, _requestRebuild: () => void): void {
    // Schemas are registered in main.ts via postSchemas.ts. Here we just
    // subscribe to non-topology keys → hot uniform writes.
    store.subscribe((key, value) => {
      if (key === "post.ao.radius" && typeof value === "number") {
        if (this.aoNode) this.aoNode.radius.value = value;
      } else if (key === "post.ao.intensity" && typeof value === "number") {
        this.intensityU.value = value;
      }
    });
    // Seed intensity uniform from store (radius is seeded in build()).
    this.intensityU.value = store.get("post.ao.intensity") as number;
  }

  build(input: Node, ctx: PassCtx): Node {
    if (!ctx.sceneNormal) throw new Error("AoEffect requires sceneNormal (needs.normal=true)");
    // ao() returns a TempNode with `radius`, `thickness`, `distanceExponent`,
    // `scale` uniform fields. Casting to access .radius.value at runtime.
    const node = ao(ctx.sceneDepth, ctx.sceneNormal, ctx.camera) as unknown as {
      radius: { value: number };
      mul(other: Node): Node;
    };
    // Seed radius from the latest store value, then keep our handle so
    // the subscription can update it.
    // (The store reference comes from PostStack via registerParams; we
    //  re-read on each build using the closure captured below.)
    this.aoNode = node;
    // mix(1, aoColor, intensity) = aoColor * intensity + (1 - intensity).
    // Implemented as: input * (1 - intensity + ao * intensity)
    // Using TSL: input.mul(ao.mul(intensity).add(intensity.oneMinus()))
    const factor = (node as unknown as Node).mul(this.intensityU).add(this.intensityU.oneMinus());
    return input.mul(factor);
  }

  bindUI(folder: FolderApi, store: ParamStore): void {
    const b = {
      enabled: store.get("post.ao.enabled") as boolean,
      radius: store.get("post.ao.radius") as number,
      intensity: store.get("post.ao.intensity") as number,
    };
    folder
      .addBinding(b, "enabled", { label: "Enabled" })
      .on("change", (e: { value: boolean }) => store.set("post.ao.enabled", e.value));
    folder
      .addBinding(b, "radius", { label: "Radius", min: 0.05, max: 2.0, step: 0.05 })
      .on("change", (e: { value: number }) => store.set("post.ao.radius", e.value));
    folder
      .addBinding(b, "intensity", { label: "Intensity", min: 0, max: 2, step: 0.05 })
      .on("change", (e: { value: number }) => store.set("post.ao.intensity", e.value));
  }

  dispose(): void {
    this.aoNode = null;
  }
}
```

**Note on radius seeding:** the subscription writes `aoNode.radius.value` on change, but the *initial* radius after a rebuild comes from the GTAONode default (0.25). Since `postSchemas.ts` defaults `post.ao.radius` to `0.25` and on every rebuild we read the latest value via subscription on the next ParamStore change, the visual default matches. If the user has scrubbed the slider, the value is reapplied to the new aoNode by an explicit seeding step — add it: after `this.aoNode = node;` in `build()`, the next line needs:

```ts
    // Seed the freshly-built aoNode with current radius from store.
    this.aoNode.radius.value = this.lastRadius;
```

…and we need to track `lastRadius`. Simpler: capture the store in registerParams and read in build. Update `AoEffect` like this (revised version of step 2 above):

```ts
export class AoEffect implements PostEffect {
  readonly id = "ao";
  readonly label = "AO";
  readonly needs = { normal: true } as const;
  enabled = true;

  private aoNode: { radius: { value: number } } | null = null;
  private intensityU = uniform(1.0);
  private store: ParamStore | null = null;

  registerParams(store: ParamStore): void {
    this.store = store;
    store.subscribe((key, value) => {
      if (key === "post.ao.radius" && typeof value === "number") {
        if (this.aoNode) this.aoNode.radius.value = value;
      } else if (key === "post.ao.intensity" && typeof value === "number") {
        this.intensityU.value = value;
      }
    });
    this.intensityU.value = store.get("post.ao.intensity") as number;
  }

  build(input: Node, ctx: PassCtx): Node {
    if (!ctx.sceneNormal) throw new Error("AoEffect requires sceneNormal");
    const node = ao(ctx.sceneDepth, ctx.sceneNormal, ctx.camera) as unknown as {
      radius: { value: number };
      mul(other: Node): Node;
    };
    this.aoNode = node;
    if (this.store) this.aoNode.radius.value = this.store.get("post.ao.radius") as number;
    const factor = (node as unknown as Node)
      .mul(this.intensityU)
      .add(this.intensityU.oneMinus());
    return input.mul(factor);
  }

  // bindUI and dispose unchanged from step 2 above.
}
```

Use this revised version.

- [ ] **Step 3: Register AO in `POST_EFFECTS`**

In `src/render/post/index.ts`:

```ts
import type { PostEffect } from "./PostEffect";
import { AoEffect } from "./effects/AoEffect";

export const POST_EFFECTS: PostEffect[] = [
  new AoEffect(),
];
```

- [ ] **Step 4: Verify**

Run: `npx tsc --noEmit` → PASS.

`npm run dev`, start test source.
- AO is back (scene matches pre-Task-6 visuals).
- Post folder has an "AO" sub-folder with Enabled / Radius / Intensity widgets.
- Toggle Enabled off → AO contribution disappears AND the MRT normal pass is skipped (no visible artifact; verify via no crashes and slightly better frame time in the FPS overlay if scene-bound).
- Scrub Radius / Intensity → live update.
- Reload page → values + enable state persist.

- [ ] **Step 5: Commit**

```bash
git add src/render/post/GTAONode.js src/render/GTAONode.js src/render/post/effects/AoEffect.ts src/render/post/index.ts
git commit -m "feat(post): migrate GTAO into AoEffect inside PostStack"
```

(The `git mv` shows up as `R` for `GTAONode.js`; both old and new paths in `git add` cover the rename.)

---

### Task 8: `TonemapEffect`

Implements exposure (uniform multiply) + mode (drives `renderer.toneMapping`, which `PostProcessing` reads in `update()`). Mode change requires `post.needsUpdate = true`, which we get by calling `requestRebuild()`.

**Files:**
- Create: `src/render/post/effects/TonemapEffect.ts`
- Modify: `src/render/post/index.ts` (append to `POST_EFFECTS`)

- [ ] **Step 1: Create `src/render/post/effects/TonemapEffect.ts`**

```ts
// src/render/post/effects/TonemapEffect.ts
import {
  NoToneMapping,
  AgXToneMapping,
  ACESFilmicToneMapping,
  NeutralToneMapping,
  type ToneMapping,
} from "three";
import { uniform } from "three/tsl";
import type { Node, WebGPURenderer } from "three/webgpu";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";

// Index → three.js ToneMapping constant. Must match optionLabels in
// postSchemas.ts (`post.tonemap.mode`): ["None","AgX","ACES","Neutral"].
const TONEMAP_TABLE: ToneMapping[] = [
  NoToneMapping,
  AgXToneMapping,
  ACESFilmicToneMapping,
  NeutralToneMapping,
];

/**
 * Drives the renderer's tone-mapping constant (which PostProcessing applies
 * via renderOutput in update()) and inserts an exposure uniform multiply
 * before that. When disabled, sets renderer.toneMapping = NoToneMapping
 * and skips the multiply.
 *
 * Mode and enabled changes require a rebuild because renderOutput is baked
 * into the QuadMesh material at update() time — flipping needsUpdate is what
 * picks up a new renderer.toneMapping value.
 */
export class TonemapEffect implements PostEffect {
  readonly id = "tonemap";
  readonly label = "Tonemap";
  readonly needs = {} as const;
  enabled = true;

  private renderer: WebGPURenderer;
  private exposureU = uniform(1.0);

  constructor(renderer: WebGPURenderer) {
    this.renderer = renderer;
  }

  registerParams(store: ParamStore, requestRebuild: () => void): void {
    // Seed renderer tone-mapping and exposure from store.
    this.applyMode(store);
    this.exposureU.value = store.get("post.tonemap.exposure") as number;

    store.subscribe((key, value) => {
      if (key === "post.tonemap.exposure" && typeof value === "number") {
        this.exposureU.value = value;
      } else if (key === "post.tonemap.mode") {
        this.applyMode(store);
        requestRebuild();   // bake new renderer.toneMapping into material
      } else if (key === "post.tonemap.enabled") {
        this.applyMode(store);
        // enabled also triggers the PostStack-level rebuild via the
        // `post.*.enabled` subscription in PostStack — no requestRebuild needed.
      }
    });
  }

  build(input: Node, _ctx: PassCtx): Node {
    return input.mul(this.exposureU);
  }

  bindUI(folder: FolderApi, store: ParamStore): void {
    const b = {
      enabled: store.get("post.tonemap.enabled") as boolean,
      mode: store.get("post.tonemap.mode") as number,
      exposure: store.get("post.tonemap.exposure") as number,
    };
    folder
      .addBinding(b, "enabled", { label: "Enabled" })
      .on("change", (e: { value: boolean }) => store.set("post.tonemap.enabled", e.value));
    folder
      .addBinding(b, "mode", {
        label: "Mode",
        options: { None: 0, AgX: 1, ACES: 2, Neutral: 3 },
      })
      .on("change", (e: { value: number }) => store.set("post.tonemap.mode", e.value));
    folder
      .addBinding(b, "exposure", { label: "Exposure", min: 0, max: 4, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.tonemap.exposure", e.value));
  }

  dispose(): void {
    this.renderer.toneMapping = NoToneMapping;
  }

  private applyMode(store: ParamStore): void {
    const enabled = store.get("post.tonemap.enabled") as boolean;
    const modeIdx = store.get("post.tonemap.mode") as number;
    this.renderer.toneMapping = enabled ? (TONEMAP_TABLE[modeIdx] ?? NoToneMapping) : NoToneMapping;
  }
}
```

- [ ] **Step 2: Plumb the renderer into `POST_EFFECTS` (it's now a factory)**

`POST_EFFECTS` was a const list; `TonemapEffect` needs the renderer. Change `src/render/post/index.ts`:

```ts
// src/render/post/index.ts
import type { WebGPURenderer } from "three/webgpu";
import type { PostEffect } from "./PostEffect";
import { AoEffect } from "./effects/AoEffect";
import { TonemapEffect } from "./effects/TonemapEffect";

/** Canonical order. Reorder only by editing this list. */
export function buildPostEffects(renderer: WebGPURenderer): PostEffect[] {
  return [
    new AoEffect(),
    new TonemapEffect(renderer),
  ];
}
```

Update the import in `src/App.ts`:

```ts
import { buildPostEffects } from "./render/post";
```

And the construction call:

```ts
    this.postStack = new PostStack(renderer, scene, camera, paramStore, buildPostEffects(renderer));
```

- [ ] **Step 3: Verify**

Run: `npx tsc --noEmit` → PASS.

`npm run dev`, start test source.
- Default mode is "None" — scene looks identical to Task 7 (AO present, no tonemap change).
- Switch Mode → AgX: shadows tighten, highlights roll off.
- Scrub Exposure: live brightness change.
- Toggle Enabled off → tonemap bypassed (linear pass-through; may look washed but should not crash).
- Reload → mode + exposure + enable persist.

- [ ] **Step 4: Commit**

```bash
git add src/render/post/effects/TonemapEffect.ts src/render/post/index.ts src/App.ts
git commit -m "feat(post): TonemapEffect — exposure multiply + AgX/ACES/Neutral modes"
```

---

### Task 9: `BloomEffect`

Last effect for this round. `bloom()` from Three's TSL addons; positional args become uniforms inside BloomNode.

**Files:**
- Create: `src/render/post/effects/BloomEffect.ts`
- Modify: `src/render/post/index.ts` (insert before Tonemap)

- [ ] **Step 1: Create `src/render/post/effects/BloomEffect.ts`**

```ts
// src/render/post/effects/BloomEffect.ts
import { bloom } from "three/addons/tsl/display/BloomNode.js";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";

/**
 * Wraps three's TSL bloom(node, strength, radius, threshold). The three
 * args become uniforms on the returned BloomNode; hot updates write
 * `bloomNode.<field>.value`. Output is `input + bloom(input)`.
 */
export class BloomEffect implements PostEffect {
  readonly id = "bloom";
  readonly label = "Bloom";
  readonly needs = {} as const;
  enabled = false;

  private bloomNode: { strength: { value: number }; radius: { value: number }; threshold: { value: number } } | null = null;
  private store: ParamStore | null = null;

  registerParams(store: ParamStore): void {
    this.store = store;
    store.subscribe((key, value) => {
      if (typeof value !== "number" || !this.bloomNode) return;
      if (key === "post.bloom.strength")       this.bloomNode.strength.value = value;
      else if (key === "post.bloom.radius")    this.bloomNode.radius.value = value;
      else if (key === "post.bloom.threshold") this.bloomNode.threshold.value = value;
    });
  }

  build(input: Node, _ctx: PassCtx): Node {
    const s = this.store!;
    const strength  = s.get("post.bloom.strength")  as number;
    const radius    = s.get("post.bloom.radius")    as number;
    const threshold = s.get("post.bloom.threshold") as number;
    const node = bloom(input, strength, radius, threshold) as unknown as {
      strength:  { value: number };
      radius:    { value: number };
      threshold: { value: number };
    };
    this.bloomNode = node;
    return input.add(node as unknown as Node);
  }

  bindUI(folder: FolderApi, store: ParamStore): void {
    const b = {
      enabled:   store.get("post.bloom.enabled")   as boolean,
      strength:  store.get("post.bloom.strength")  as number,
      radius:    store.get("post.bloom.radius")    as number,
      threshold: store.get("post.bloom.threshold") as number,
    };
    folder
      .addBinding(b, "enabled", { label: "Enabled" })
      .on("change", (e: { value: boolean }) => store.set("post.bloom.enabled", e.value));
    folder
      .addBinding(b, "strength", { label: "Strength", min: 0, max: 3, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.bloom.strength", e.value));
    folder
      .addBinding(b, "radius", { label: "Radius", min: 0, max: 1, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.bloom.radius", e.value));
    folder
      .addBinding(b, "threshold", { label: "Threshold", min: 0, max: 2, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.bloom.threshold", e.value));
  }

  dispose(): void {
    this.bloomNode = null;
  }
}
```

- [ ] **Step 2: Insert into `POST_EFFECTS` before Tonemap**

```ts
// src/render/post/index.ts
import type { WebGPURenderer } from "three/webgpu";
import type { PostEffect } from "./PostEffect";
import { AoEffect } from "./effects/AoEffect";
import { BloomEffect } from "./effects/BloomEffect";
import { TonemapEffect } from "./effects/TonemapEffect";

export function buildPostEffects(renderer: WebGPURenderer): PostEffect[] {
  return [
    new AoEffect(),
    new BloomEffect(),
    new TonemapEffect(renderer),
  ];
}
```

- [ ] **Step 3: Verify**

Run: `npx tsc --noEmit` → PASS.

`npm run dev`, start test source.
- Default: Bloom disabled — scene unchanged from Task 8.
- Toggle Bloom Enabled on → glow on bright pixels (the bloom appears once the rebuild fires; should be instant).
- Scrub Strength up → visible bloom intensifies; Threshold down → more pixels qualify; Radius up → softer / larger halo.
- Toggle off → bloom disappears.
- With Tonemap set to AgX + Bloom on: bloom accumulates in linear, then tonemap rolls off highlights.
- Reload → all bloom settings persist.

- [ ] **Step 4: Commit**

```bash
git add src/render/post/effects/BloomEffect.ts src/render/post/index.ts
git commit -m "feat(post): BloomEffect (AO → Bloom → Tonemap canonical order)"
```

---

## Done

Final pipeline:

```
sceneColor (pass MRT: output, [normal if AO on])
   │
   ├── AoEffect          (needs normal) — input * mix(1, ao, intensity)
   │
   ├── BloomEffect       — input + bloom(input, strength, radius, threshold)
   │
   └── TonemapEffect     — input * exposure (mode applied by renderer.toneMapping)
   │
   ▼
PostProcessing.outputNode → renderOutput → screen
```

Panel:

```
Analysis  (existing)
Scenes    (existing)
Camera    (new) — FOV, Preset
Post      (new) — AO, Bloom, Tonemap
Reset to defaults
```

All `post.*` and `camera.*` keys persist via `ParamStore`'s existing `localStorage` layer (key `autocorrelation.params.v1`).
