# Vite HMR Design

**Date:** 2026-04-28
**Status:** Design approved, awaiting plan

## Goal

Make most TypeScript edits during `npm run dev` hot-replace the running app instead of triggering a full page reload. The user's pain point is that page reload re-prompts for tab-capture permission (`getDisplayMedia`), turning every save into a multi-click ceremony. After this change, edits to the rendering and panel layers swap in-place while the `AudioContext`, `MediaStream`, and `AudioWorkletNode` survive untouched.

## Architecture

`main.ts` becomes the orchestrator. It owns three lifecycle scopes:

```
┌─ main.ts ──────────────────────────────────────────────────┐
│                                                             │
│  PAGE-LIFETIME (created once, survives every HMR):          │
│    canvas, AudioContext, MediaStream, AudioWorkletNode,     │
│    ParamStore, WebGPURenderer                               │
│                                                             │
│       │ passed in as deps                                   │
│       ▼                                                     │
│                                                             │
│  ┌─ APP-LIFETIME (rebuilt on every HMR replay) ─────────┐   │
│  │   App: scene, camera, camera rig, 9 line renderers,  │   │
│  │        FPS overlay, RAF, keydown handler,            │   │
│  │        camera-side resize handler                    │   │
│  │   ParamPanel: tweakpane bindings                     │   │
│  │   WorkletBridge: ParamStore subscription             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**HMR sequence on file save:**

1. Vite invalidates the changed module(s); invalidation propagates up to `main.ts`'s accept boundary.
2. `main.ts` calls `app.dispose()`, `panel.dispose()`, `bridge.dispose()` — full teardown of the App-lifetime scope.
3. The accept callback returns; Vite re-imports the changed modules.
4. `main.ts` constructs new `App` / `ParamPanel` / `WorkletBridge` instances, passing in the same preserved page-lifetime deps.
5. New App's render loop starts; new tweakpane mounts; new bridge subscribes to ParamStore.

**WebGPURenderer is page-lifetime** because device init takes 10–50 ms async. Doing it once per page (not once per HMR) makes rebuilds essentially instant. The Three.js `Scene` and `PerspectiveCamera` stay App-lifetime — they're cheap JS objects with no GPU state.

## Tech Stack

- **Vite** ≥ 5.4 (already a dependency) — `import.meta.hot.accept(deps, cb)` for the boundary.
- **TypeScript** — existing.
- **Three.js** WebGPU renderer — existing; no new APIs needed (uses `LineRenderer.dispose()` added in v3.1).
- **tweakpane** — existing; uses its built-in `Pane.dispose()` API.

No new runtime dependencies.

## File Structure

**Modified files:**

- `src/main.ts` — major restructure. Becomes the orchestrator: builds page-lifetime deps on first Start click, builds/disposes App-lifetime layer, registers the HMR accept callback. Source-factory imports become dynamic (so edits to `audio/AudioSource.ts` don't invalidate `main.ts`).
- `src/App.ts` — constructor takes an `AppDeps` object; `start()` is now sync and wires up scene/camera/lines/RAF/listeners; new `dispose()` method tears down everything App created. `keydownHandler`, `resizeHandler`, and `rafHandle` lift from anonymous closures to named fields so they can be removed.
- `src/render/Scene.ts` — split `createScene()` into:
  - `createRenderer(canvas)` (page-lifetime, called from `main.ts`) — does WebGPU device init and registers a window resize listener that calls `renderer.setSize(...)`. Listener stays for the page's life.
  - `createSceneAndCamera()` (App-lifetime, called from `App.start()`) — creates `Scene` and `PerspectiveCamera`. App registers its own resize listener for the camera (separate from the renderer's).
- `src/params/ParamPanel.ts` — capture the unsubscribe handle returned by `paramStore.subscribe()`, expose `dispose()` that calls the unsubscribe and `pane.dispose()`.
- `src/params/WorkletBridge.ts` — same pattern: capture unsubscribe handle, expose `dispose()`.

**Unchanged files:**

- `src/audio/dsp-worklet.ts`, `src/audio/AudioSource.ts`, `src/audio/worklet-polyfills.ts` — out of HMR scope (worklet) or dynamic-imported (source factories).
- `src/params/ParamStore.ts`, `src/params/schemas.ts` — imported statically by `main.ts`; full-reload on edit, acceptable.
- `src/render/LineLayouts.ts`, `src/render/LineRenderer.ts`, `src/render/CameraRig.ts` — imported transitively by `App`; covered by the App accept boundary, no per-module accept needed.
- `src/store/FeatureStore.ts`, `src/ui/Stats.ts` — same as above.
- `crates/dsp/**` — separate `npm run wasm` build step; out of HMR scope.

**No new files.**

## Components

### `main.ts`

```ts
import { App } from "./App";
import { ParamStore } from "./params/ParamStore";
import { ParamPanel } from "./params/ParamPanel";
import { WorkletBridge } from "./params/WorkletBridge";
import { analysisSchemas } from "./params/schemas";
import { createRenderer } from "./render/Scene";
import dspWorkletUrl from "./audio/dsp-worklet?worker&url";
import dspWasmUrl from "./wasm-pkg/dsp_bg.wasm?url";
import type { AudioSourceBundle } from "./audio/AudioSource";

interface AppDeps {
  canvas: HTMLCanvasElement;
  renderer: import("three/webgpu").WebGPURenderer;
  audioContext: AudioContext;
  workletNode: AudioWorkletNode;
  paramStore: ParamStore;
}

// Constructor refs are mutable so the HMR accept callback can swap them in.
// (Vite's `accept(deps, cb)` does NOT rewire the file's static imports —
// the cb receives the new module exports and we update these refs.)
let AppCtor: typeof App = App;
let ParamPanelCtor: typeof ParamPanel = ParamPanel;
let WorkletBridgeCtor: typeof WorkletBridge = WorkletBridge;

let pageDeps: AppDeps | null = null;
let app: App | null = null;
let panel: ParamPanel | null = null;
let bridge: WorkletBridge | null = null;
let initialBootstrap = true; // bridge.bootstrap() runs once per page, not on HMR replays

async function buildPageDeps(factory: () => Promise<AudioSourceBundle>): Promise<AppDeps> {
  const { context, source } = await factory();
  const wasmModule = await WebAssembly.compileStreaming(fetch(dspWasmUrl));
  await context.audioWorklet.addModule(dspWorkletUrl);
  const workletNode = new AudioWorkletNode(context, "dsp-processor", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    processorOptions: { wasmModule },
  });
  source.connect(workletNode);

  const canvas = document.getElementById("app") as HTMLCanvasElement;
  const renderer = await createRenderer(canvas);

  const paramStore = new ParamStore();
  for (const s of analysisSchemas) paramStore.register(s);

  return { canvas, renderer, audioContext: context, workletNode, paramStore };
}

function buildAppLayer(deps: AppDeps): void {
  app = new AppCtor(deps);
  panel = new ParamPanelCtor(deps.paramStore);
  bridge = new WorkletBridgeCtor(deps.paramStore, deps.workletNode.port);
  app.start();
  if (initialBootstrap) {
    bridge.bootstrap(); // posts the initial "configure" message — once per page only.
    initialBootstrap = false;
  }
}

function teardownAppLayer(): void {
  app?.dispose();   panel?.dispose();   bridge?.dispose();
  app = null;       panel = null;       bridge = null;
}

const onStart = async (factory: () => Promise<AudioSourceBundle>): Promise<void> => {
  if (pageDeps) return;
  pageDeps = await buildPageDeps(factory);
  buildAppLayer(pageDeps);
};

// Buttons — dynamic-import the source factories so edits to AudioSource.ts
// don't invalidate main.ts and force a full reload.
document.getElementById("start-mic")?.addEventListener("click", async () => {
  const { createMicSource } = await import("./audio/AudioSource");
  await onStart(createMicSource);
});
document.getElementById("start-tab")?.addEventListener("click", async () => {
  const { createTabSource } = await import("./audio/AudioSource");
  await onStart(createTabSource);
});
document.getElementById("start-test")?.addEventListener("click", async () => {
  const { createTestSource } = await import("./audio/AudioSource");
  await onStart(() => createTestSource(440));
});

if (import.meta.hot) {
  import.meta.hot.accept(
    ["./App", "./params/ParamPanel", "./params/WorkletBridge"],
    ([appMod, panelMod, bridgeMod]) => {
      // Each entry is `Module | undefined`; undefined means that module didn't
      // change in this update batch — keep the existing constructor ref.
      if (appMod?.App) AppCtor = appMod.App;
      if (panelMod?.ParamPanel) ParamPanelCtor = panelMod.ParamPanel;
      if (bridgeMod?.WorkletBridge) WorkletBridgeCtor = bridgeMod.WorkletBridge;
      if (!pageDeps) return;
      teardownAppLayer();
      try {
        buildAppLayer(pageDeps);
      } catch (err) {
        console.error("[hmr] failed to rebuild App layer:", err);
        // Leave app/panel/bridge as null. Next successful save retries.
      }
    },
  );
}
```

### `App.ts` (lifecycle skeleton)

```ts
class App {
  private rafHandle: number | null = null;
  private keydownHandler: (e: KeyboardEvent) => void = () => {};
  private resizeHandler: () => void = () => {};

  constructor(private deps: AppDeps) {}

  start(): void {
    const { canvas: _canvas, renderer, workletNode, paramStore: _paramStore } = this.deps;

    // Scene + camera
    const { scene, camera } = createSceneAndCamera();
    this.scene = scene; this.camera = camera;

    // CameraRig + presets (existing logic)
    // FpsOverlay mount
    // (Lines are constructed in rebuildLineRenderers when the worklet posts
    //  the "configured" message — same as v3.1 today.)

    // Worklet message handler
    workletNode.port.onmessage = (e) => { /* dispatch to FeatureStore + rebuildLineRenderers */ };

    // Keydown — named handler so we can remove it
    this.keydownHandler = (e) => { /* preset shortcuts, space toggle */ };
    window.addEventListener("keydown", this.keydownHandler);

    // Camera-side resize (renderer-side stays in main.ts via createRenderer)
    this.resizeHandler = () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", this.resizeHandler);

    // Render loop — capture handle for cancellation
    const loop = (now: number) => {
      // … per-frame work
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
    for (const line of [
      this.waveformLine, this.bufferAcfLine, this.spectrumLine,
      this.lowRmsLine, this.midRmsLine, this.highRmsLine,
      this.rmsLine, this.lowRmsAcfLine, this.rmsAcfLine,
    ]) line?.dispose();
    this.fps.unmount();
    this.deps.workletNode.port.onmessage = null;
  }
}
```

### `render/Scene.ts` split

```ts
// Page-lifetime: WebGPU device + renderer-side resize listener.
export async function createRenderer(canvas: HTMLCanvasElement): Promise<WebGPURenderer> {
  const renderer = new WebGPURenderer({ canvas, antialias: true });
  await renderer.init();
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(new Color(0x0a0a0a), 1);
  // Renderer-side resize: page-lifetime (registered once, never removed).
  window.addEventListener("resize", () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  });
  return renderer;
}

// App-lifetime: Scene + Camera. App registers its own camera resize listener.
export function createSceneAndCamera(): { scene: Scene; camera: PerspectiveCamera } {
  const scene = new Scene();
  const camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
  return { scene, camera };
}
```

### `ParamPanel` and `WorkletBridge` dispose contracts

Both currently subscribe to `ParamStore` in their constructor (per v3.1). The minor refactor:

- Capture the unsubscribe callback returned by `paramStore.subscribe(...)` in a private field.
- Expose `dispose(): void` that:
  - Calls the captured unsubscribe.
  - For `ParamPanel`: also calls `this.pane.dispose()` (tweakpane removes its own DOM and listeners).
  - For `WorkletBridge`: nothing else (the port is page-lifetime).

`bridge.bootstrap()` (which posts the initial `"configure"` message) is only called once per page-lifetime, NOT on HMR replays — the page-lifetime worklet is already configured.

## Data Flow

Unchanged from v3.1 at runtime. The HMR mechanism only intervenes at file save:

```
File save in IDE
   │
   ▼
Vite invalidates modified module(s)
   │
   ▼ propagates up the import graph
main.ts accept boundary catches it
   │
   ▼
teardownAppLayer():
  app.dispose()      ── cancels RAF, removes listeners, disposes lines, unmounts FPS,
                        nulls port.onmessage
  panel.dispose()    ── unsubscribes from paramStore, disposes tweakpane Pane
  bridge.dispose()   ── unsubscribes from paramStore
   │
   ▼
buildAppLayer(pageDeps):
  new App(pageDeps), app.start()
  new ParamPanel(paramStore)
  new WorkletBridge(paramStore, port); bridge.bootstrap() is NOT called on HMR
                                       (only on initial construction; the worklet
                                       is already configured from the page-lifetime
                                       initial bootstrap)
   │
   ▼
First worklet message arrives → existing rebuildLineRenderers path runs
```

## Error Handling

**HMR before user clicks Start.** `pageDeps` is `null`; the accept callback no-ops. Vite still updates module records; next click uses fresh code.

**Broken edit (compile error or runtime throw in new App).** Old App is already disposed when we discover the new one fails. Wrap `buildAppLayer(pageDeps)` in `try/catch`; on failure, leave `app`/`panel`/`bridge` as `null` and log the error. Vite's built-in error overlay surfaces compile errors. Next save retries; recovery is automatic.

**WebGPU device lost.** Out of scope. Pre-existing concern not made worse by this design (renderer's lifecycle is unchanged from a device-loss perspective).

**WASM rebuild (`npm run wasm`).** Out of HMR scope. Reloading WASM requires recreating the worklet, which requires a new `AudioContext`, which loses tab-capture permission. User accepts a manual page reload after Rust DSP edits — same as today.

## Testing

No automated tests. HMR is a dev-time mechanism; Vite's own test suite covers the HMR engine, and the dispose paths are pure-pass-through cleanups whose unit-test value is low.

**Acceptance is via manual smoke tests in `npm run dev`:**

1. **Cold start (regression).** Click Mic; audio + lines work as today.
2. **Edit color in App.ts → save.** Line color updates without page reload; audio still flowing.
3. **Edit Y-offset in LineLayouts.ts → save.** Lines reposition; no reload.
4. **Edit CameraRig.ts (e.g. tween easing) → save.** Camera behavior changes; no reload.
5. **Edit ParamPanel.ts (e.g. add a folder label) → save.** Panel rebuilds; no reload.
6. **Edit `params/schemas.ts` → save.** Full page reload (acceptable per design).
7. **Edit `dsp-worklet.ts` → save.** Full reload (audio worklet API constraint).
8. **Save 20× in a row.** DevTools heap stays flat after GC; `getEventListeners(window)` shows exactly one `keydown` + two `resize` (renderer's + App's) listeners after each save — not N+1.
9. **Tab Audio source + multiple HMR replays.** Permission stays granted; no re-prompt.
10. **Introduce syntax error → save → fix → save.** Vite error overlay; recovery without page reload.
11. **`npm run build`.** Bundle has no `import.meta.hot` references; size delta < 1 KB.

Test (8) is the load-bearing one — it verifies the dispose paths actually balance every `addEventListener` with a `removeEventListener`. If listeners accumulate, dispose is missing something.

## Future Integration

Not in scope, but natural follow-ups once this lands:

- **AudioSource HMR.** Currently dynamic-imported so edits don't break dev. Could later add a "swap source" UI button that lets the dev re-pick a tab without page reload, separate from HMR.
- **Schema HMR.** `params/schemas.ts` currently full-reloads on edit. If devs iterate frequently on schemas, add `accept('./params/schemas')` to the boundary, which would force `ParamPanel` rebuild on schema edits — works as long as the ParamStore can re-register schemas idempotently.
- **Worklet TS HMR.** Out of scope due to the AudioWorkletProcessor registration constraint. A workaround (multiple processor names with versioning) is possible but adds significant complexity for marginal gain.
