# Postprocessing Stack — Design

Status: Approved 2026-05-24. Implementation plan to follow.

## Goal

Replace the single inline GTAO graph in `App.ts` with a modular postprocessing stack. Ship with three effects (AO, Bloom, Tonemap/HDR). Designed so adding DOF, lens distortion, vignette, glitch, chromatic aberration, halftone, etc. is one new file each. Every effect is toggleable; toggles trigger a graph rebuild so disabled effects also drop their MRT requirements (e.g. disabling AO skips the normals pass).

Also introduces a new top-level **Camera** folder in the panel (FOV + preset dropdown).

## Non-goals

- Reorderable effects in the panel — fixed canonical order in code.
- Presets / profiles ("Cinematic", "CRT", etc.) — could layer on later.
- Automated vitest tests for the post stack — manual verification only for this round.
- Camera folder beyond FOV + preset.

## File layout

```
src/render/post/
  PostStack.ts          # owns ordered list, rebuilds PostProcessing.outputNode
  PostEffect.ts         # interface + PassCtx type
  effects/
    AoEffect.ts         # wraps the existing GTAONode
    TonemapEffect.ts    # exposure + AgX/ACES/Neutral/None
    BloomEffect.ts      # wraps three's TSL bloom()
  index.ts              # POST_EFFECTS = [ao, bloom, tonemap]  (canonical order)
  GTAONode.js           # moved from src/render/

src/params/
  postSchemas.ts        # post.* keys
  cameraSchemas.ts      # camera.* keys
```

## Architecture

### Effect interface

```ts
// src/render/post/PostEffect.ts
import type { Node, TextureNode } from "three/tsl";
import type { PerspectiveCamera, Scene } from "three";
import type { ParamStore } from "../../params/ParamStore";
import type { FolderApi } from "tweakpane";

export interface PassCtx {
  scene: Scene;
  camera: PerspectiveCamera;
  sceneNormal: TextureNode | null;   // null if no enabled effect needs it
  sceneDepth: TextureNode;           // always available from pass
}

export interface PostEffect {
  readonly id: string;               // stable; matches param prefix (e.g. "ao")
  readonly label: string;            // panel folder title
  readonly needs: Readonly<{ normal?: boolean }>;

  enabled: boolean;                  // read by PostStack at build time

  build(input: Node, ctx: PassCtx): Node;
  registerParams(store: ParamStore): void;
  bindUI(folder: FolderApi, store: ParamStore): void;
  dispose(): void;
}
```

Color and depth are always available; only normal needs to be opted-in via `needs.normal`.

### PostStack

Owns the lifecycle of `PostProcessing` plus the ordered list of effects.

```ts
class PostStack {
  constructor(
    renderer: WebGPURenderer,
    scene: Scene,
    camera: PerspectiveCamera,
    store: ParamStore,
    effects: PostEffect[],          // canonical order
  );

  build(): void;                    // (re)build PostProcessing.outputNode
  renderAsync(): Promise<void>;
  dispose(): void;
}
```

`build()`:
1. Read each effect's `enabled` from the store (`post.<id>.enabled`).
2. `needsNormal = enabled.some(e => e.needs.normal)`.
3. Create `pass(scene, camera)`. Set MRT to `{ output }` if no normals needed, `{ output, normal: transformedNormalView }` otherwise. Depth is always available via `pass.getTextureNode("depth")`.
4. Build `PassCtx { scene, camera, sceneDepth, sceneNormal | null }`.
5. Chain: `let node = sceneColorTexture; for (const e of enabledInOrder) node = e.build(node, ctx); post.outputNode = node;`.
6. Dispose previous `PostProcessing` instance.

`PostStack` subscribes to ParamStore in its constructor. Any key matching `/^post\.[^.]+\.enabled$/` schedules `rebuild()` via microtask debounce so toggling multiple enables in one tick → one rebuild. Mode changes that affect graph topology (e.g. `post.tonemap.mode`) also rebuild.

Slider-style params (exposure, bloom strength) feed TSL `uniform()` refs owned by each effect — those are hot and never trigger rebuild.

### Per-effect param subscription

Each effect subscribes its own non-enable, non-topology keys in `registerParams` and writes new values into its uniforms. Keeps effects self-contained; no `PostBridge` class needed.

## Effects (initial set)

### AoEffect
- `needs: { normal: true }`
- Wraps existing `GTAONode.js` (moved into `src/render/post/`).
- Params:
  - `post.ao.enabled` (boolean, default `true`)
  - `post.ao.radius` (continuous, 0.1–2.0, default `0.5`)
  - `post.ao.intensity` (continuous, 0–4, default `1.0`)
- `build(input, ctx)`: `return input.mul(ao(ctx.sceneDepth, ctx.sceneNormal!, ctx.camera))`.

### BloomEffect
- `needs: {}` (samples its own input)
- Uses `bloom()` from `three/addons/tsl/display/BloomNode.js`.
- Params:
  - `post.bloom.enabled` (boolean, default `false` — opt-in)
  - `post.bloom.strength` (continuous, 0–3, default `0.5`)
  - `post.bloom.radius` (continuous, 0–1, default `0.4`)
  - `post.bloom.threshold` (continuous, 0–2, default `0.85`)
- `build(input)`: `return input.add(bloom(input, strength, radius, threshold))`.

### TonemapEffect
- `needs: {}`
- Renderer is set to `LinearToneMapping` so upstream effects stay in linear HDR; tonemapping is applied inside the node graph.
- Params:
  - `post.tonemap.enabled` (boolean, default `true`)
  - `post.tonemap.mode` (discrete `[0,1,2,3]`, default `0` = None, mapping: `0→None, 1→AgX, 2→ACES, 3→Neutral`)
  - `post.tonemap.exposure` (continuous, 0–4, default `1.0`)
- `build(input)`: `mode === 0` → return `input` unchanged; otherwise `return toneMapping(modeConst, exposureUniform, input)`.
- Default `mode: None` preserves current visuals; user can flip to AgX for a more cinematic look.

### Canonical order

```
sceneColor → AO → Bloom → Tonemap → output
```

Tonemap is last so AO and Bloom accumulate in HDR before mapping.

## Panel integration

`ParamPanel` adds two new top-level folders:

```
Analysis   (existing)
Scenes     (existing)
Camera     (new)
Post       (new)
```

The existing `dsp.*`-prefix filter pattern is preserved. `ParamPanel` exposes `this.post: FolderApi` and `this.camera: FolderApi`; `App` wires the stack and camera UI into them in `bindUI`.

### Post folder

`PostStack.bindUI(postFolder, store)` walks effects and creates one sub-folder per effect (matching the Components pattern):

```
Post
├── AO        [☑ enabled] [radius] [intensity]
├── Bloom     [☐ enabled] [strength] [radius] [threshold]
└── Tonemap   [☑ enabled] [mode ▾] [exposure]
```

Each effect's `bindUI(subFolder, store)` adds its own widgets.

### Camera folder

Minimum viable:
- `camera.fov` (continuous, 20–120, default `50`)
- `camera.preset` (discrete dropdown mapping to `["front","side","spectrum","rms","buffer-acf","rms-acf"]`)

Selecting a preset triggers `rig.goTo(name, { duration: 0.8 })`. FOV writes `camera.fov` and calls `camera.updateProjectionMatrix()`.

`App` owns the Camera folder wiring (it already holds the rig and the camera). Adds `App.bindCameraUI(folder)` parallel to existing `bindUI(folder)`. `ParamPanel` exposes `this.camera`; `main.ts` calls `app.bindCameraUI(panel.camera)`.

## Param schemas

New `src/params/postSchemas.ts` and `src/params/cameraSchemas.ts`. Registered in `main.ts` alongside `analysisSchemas`. Each schema needs `label` and `reconfig: false`.

```ts
// postSchemas.ts (sketch, full labels omitted)
export const postSchemas: ParamSchema[] = [
  { key: "post.ao.enabled",      kind: "boolean",  default: true,  ... },
  { key: "post.ao.radius",       kind: "continuous", min: 0.1, max: 2.0, step: 0.05, default: 0.5, ... },
  { key: "post.ao.intensity",    kind: "continuous", min: 0.0, max: 4.0, step: 0.05, default: 1.0, ... },

  { key: "post.bloom.enabled",   kind: "boolean",  default: false, ... },
  { key: "post.bloom.strength",  kind: "continuous", min: 0.0, max: 3.0,  step: 0.01, default: 0.5, ... },
  { key: "post.bloom.radius",    kind: "continuous", min: 0.0, max: 1.0,  step: 0.01, default: 0.4, ... },
  { key: "post.bloom.threshold", kind: "continuous", min: 0.0, max: 2.0,  step: 0.01, default: 0.85, ... },

  { key: "post.tonemap.enabled", kind: "boolean",  default: true,  ... },
  { key: "post.tonemap.mode",    kind: "discrete", options: [0,1,2,3], default: 0, ... },
  { key: "post.tonemap.exposure",kind: "continuous", min: 0.0, max: 4.0, step: 0.01, default: 1.0, ... },
];

// cameraSchemas.ts
export const cameraSchemas: ParamSchema[] = [
  { key: "camera.fov",    kind: "continuous", min: 20, max: 120, step: 1, default: 50, ... },
  { key: "camera.preset", kind: "discrete",   options: [0,1,2,3,4,5], default: 0, ... },
];
```

Discrete-option caveat: `ParamStore`'s `options` field is `number[]` only. `tonemap.mode` and `camera.preset` are integer indices; each effect/owner keeps a local `LABELS` array. Avoids touching `ParamStore`.

## Persistence

Free — `ParamStore.set()` already writes the full value map to `localStorage` (key `autocorrelation.params.v1`) on every change, and `register()` rehydrates on load. Every `post.*` and `camera.*` key registered through `paramStore.register()` is persisted automatically.

## App.ts changes

```ts
// Replaces the current inline graph (App.ts:86-99):
import { PostStack } from "./render/post/PostStack";
import { POST_EFFECTS } from "./render/post";

// in start():
this.postStack = new PostStack(renderer, scene, camera, paramStore, POST_EFFECTS);
this.postStack.build();

// in the RAF loop:
void this.postStack.renderAsync();

// in dispose():
this.postStack.dispose();
```

`App` retains ownership of camera + rig and adds `bindCameraUI(folder)`.

## Migration of GTAONode

Move `src/render/GTAONode.js` → `src/render/post/GTAONode.js`. Update the existing import in `App.ts` to point at the new path (which becomes irrelevant after the inline graph is replaced — `AoEffect.ts` is the only importer).

## Manual verification

User will manually verify in the browser. Spot-checks worth running:
1. Default load — AO on, Tonemap on (mode: None), Bloom off → scene matches current visuals.
2. Toggle Bloom on, raise strength → glow on bright elements.
3. Switch Tonemap mode to AgX, scrub exposure → mapping responds.
4. Toggle AO off → AO contribution gone, scene slightly brighter.
5. Reload page → all enables and slider values persist.
6. Camera FOV slider → camera updates live. Preset dropdown → smooth tween.

## Risks / open questions

- **`bloom()` from `three/addons`** — confirm the import path resolves with our `vite-plugin-wasm` setup. If `three/addons/tsl/display/BloomNode.js` isn't exposed by our Three version, fall back to copying the addon source into `src/render/post/` similar to `GTAONode.js`. Plan should verify with a quick import check before committing the BloomEffect to that path.
- **`toneMapping` TSL function signature** — `three/tsl` exports `toneMapping(mode, exposure, color)`. The first arg is a `ToneMapping` constant from `three`. Plan should verify the import path and signature.
- **Renderer tone mapping** — must set `renderer.toneMapping = LinearToneMapping` so the node graph isn't double-mapped at output. Currently the default `NoToneMapping` would already be correct, but worth being explicit.
