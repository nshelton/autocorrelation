# Toggleable Visualizer Components

**Date:** 2026-05-20
**Status:** Approved (pending review)

## Motivation

The project is pivoting from audio-analysis-focused work into a broader visualizer system. Today `App.ts` hardcodes two views (`BoxView`, `DebugView`): instantiated inline, updated in the RAF loop, disposed in `dispose()`. Adding a new view means editing `App.ts` in four places, and there is no way to turn individual views off at runtime.

The goal is a small, uniform component system: each visual subsystem lives in its own file, declares its lifecycle, and is independently toggled on/off via a checkbox in the existing tweakpane panel. `App.ts` shrinks to the things only it can do (scene, camera, post-processing, key bindings, worklet message routing, ParamStore wiring).

## Non-goals

- Multiple scenes. All components share the single `Scene` so the existing GTAO post-processing pipeline keeps working unchanged.
- Per-component camera. The single `CameraRig` is app-level.
- Dynamic registration / hot-loading of components at runtime. The registry is a static array in source.
- Reordering or layering components from the UI.
- Inter-component dependencies. Components do not reference each other.

## Design

### Component contract

`src/render/components/Component.ts` (new) defines:

```ts
import type { Scene } from "three";
import type { FeatureStore } from "../../store/FeatureStore";
import type { ParamStore } from "../../params/ParamStore";

export interface ComponentDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
}

export interface Component {
  update(): void;
  dispose(): void;
}

// A component class IS the registry entry. Static metadata lives on the
// class itself — no separate ComponentEntry interface, no factory wrapper.
// The second constructor arg (the App-owned params bag) is present only
// when the class declares static paramDefaults / paramOpts / paramPrefix.
export interface ComponentClass {
  new (deps: ComponentDeps, params?: Record<string, number>): Component;
  id: string;       // also the ParamStore namespace for this component
  label: string;    // tweakpane folder title
  paramPrefix?: string;          // when present, must equal id
  paramOpts?: Record<string, { min: number; max: number; step?: number }>;
  paramDefaults?: Record<string, number>;
}
```

A component is just a class with `update()` and `dispose()` plus required static `id` / `label`. It may *also* expose static `paramPrefix` / `paramOpts` / `paramDefaults` — that opt-in wires the component's own live-tunable params into ParamStore and tweakpane. When present, `paramPrefix` must equal `id` so one ParamStore namespace covers both the component's own params and its `enabled` toggle.

`update()` takes no arguments. No component currently needs `dt`; the camera rig that does still gets it from App's RAF loop. Add an argument later if a component requires it.

### Registry

`src/render/components/index.ts` (new) is one array:

```ts
import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import type { ComponentClass } from "./Component";

export const COMPONENTS: readonly ComponentClass[] = [DebugView, BoxView];
```

Adding a new component: create its file under `src/render/components/`, import it here, append to the array. `App.ts` does not import concrete component classes.

DebugView stays in `src/render/debug/DebugView.ts` because that folder contains many sub-renderers (TimeSeriesRenderer, BeatGridMarkers, etc.) that are not themselves components. The registry just imports `DebugView` from there.

### ParamStore: boolean support

`ParamStore` is number-only today. Extend it to also hold booleans (more kinds — enums, ints, strings — are anticipated as a future generalization but out of scope here):

```ts
export type ParamValue = number | boolean;

export type ParamSchema = {
  key: string;
  label: string;
  default: ParamValue;
  reconfig: boolean;
} & (
  | { kind: "discrete";   options: number[] }
  | { kind: "continuous"; min: number; max: number; step: number }
  | { kind: "boolean" }
);
```

- `validate()` returns `typeof value === "boolean"` for the `boolean` kind, and rejects non-numbers for the other two kinds.
- `default` must match the kind. (`continuous`/`discrete` defaults stay numeric; `boolean` defaults to a boolean.)
- JSON serialization already supports booleans, so persistence works unchanged.
- `WorkletBridge` only forwards numeric params today; it must skip booleans (component-toggle keys never go to the worklet). Concrete guard: in the subscriber, `if (typeof value !== "number") return;` before any `set_param` / reconfig dispatch.

### ParamPanel: checkbox widget

In `ParamPanel.addWidget()`, branch on `schema.kind === "boolean"` and use tweakpane's default checkbox binding (no `min`/`max`/`options`). The existing change-handler path already forwards values to the store.

### App.ts: component manager

Replace the hardcoded `boxView` / `debugView` fields, `update()` calls, and `dispose()` calls with one component-manager loop.

Per-component state held in App:
```ts
{
  cls: ComponentClass;
  paramsBag: Record<string, number> | null;  // stable across toggle cycles, null when cls has no params
  instance: Component | null;
  uiUnsubs: Array<() => void>;               // tweakpane + store subscriptions specific to this component
}
```

On `start()`:
1. Build the shared `ComponentDeps` once.
2. For each `cls` in `COMPONENTS`:
   - Register a `components.<cls.id>.enabled` schema in `ParamStore` (`kind: "boolean"`, `default: true`).
   - If `cls.paramDefaults` is present, allocate the stable params bag and seed each entry from ParamStore (registering its schema in the `<cls.id>.<key>` namespace if not already), so persisted values are picked up.
3. For each `cls`, if the current persisted `enabled` is `true`, construct: `new cls(deps, paramsBag ?? undefined)`. Add tweakpane bindings for this component (see folder layout below).
4. Subscribe to `ParamStore` for changes to any `components.*.enabled` key. On transition:
   - `true → false`: `instance.dispose()`, drop reference. The folder + checkbox stay; the param-slider bindings stay too (they still target the stable `paramsBag` and let the user pre-tweak).
   - `false → true`: construct a new instance with the existing `paramsBag`.

Per-frame loop iterates live instances and calls `update()`.

On `dispose()`: dispose all live instances, run all `uiUnsubs`.

App still owns: scene + camera + `CameraRig` + presets, `PostProcessing` pipeline (GTAO), keyboard/resize handlers, worklet `features` message routing into the store, FPS overlay, RAF loop. These are not components.

### bindUI / ParamPanel folder layout

For each `cls` in `COMPONENTS`:
1. `App.bindUI` adds a folder titled `cls.label` to the root pane.
2. The first row in the folder is the `components.<id>.enabled` checkbox, bound to the corresponding ParamStore key.
3. If `cls.paramDefaults` is present, App adds slider bindings for each param against the stable params bag, wired to the `<id>.<key>` ParamStore keys (this is the existing `bindViewParams` logic, lightly refactored to read from the bag instead of `view.params`).

The folder is rendered even when the component is disabled. Param sliders are live regardless of enable state — changes write through to the stable bag and to ParamStore. The next time the component is constructed, it picks up the current bag.

The existing "Reset to defaults" button at the root pane resets *all* params including the toggles — components transition off → on (or stay on) per their default of `true`.

#### Why the stable params bag

Tweakpane bindings hold a *reference* to the object they read/write. If each new component instance owned its own `params` object, the bindings created when the first instance was constructed would silently stop working after a toggle dispose→recreate cycle.

Fix: App owns one stable `Record<string, number>` per component-id for the lifetime of the page. The component's constructor receives this object and stores the reference (instead of creating its own). Both the component (each frame) and tweakpane (on slider change) read/write the same object.

Updated `BoxView` shape:

```ts
class BoxView implements Component {
  static id = "boxView";
  static label = "Box View";
  static paramPrefix = "boxView";
  static paramOpts = { pull: {min: 0, max: 1, step: 0.01}, /* … */ };
  static paramDefaults = { pull: 0.3, timestep: 1/30, width: 0.5 };

  private params: Record<string, number>;
  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.params = params;  // keep the reference, never copy
    // …rest of init
  }
}
```

### Files touched

| File | Change |
|---|---|
| `src/render/components/Component.ts` | **new** — `Component` and `ComponentClass` interface definitions |
| `src/render/components/index.ts` | **new** — `COMPONENTS` array of component classes |
| `src/render/components/BoxView.ts` | move `params` / `paramOpts` / `paramPrefix` from instance fields to static class properties (rename to `paramDefaults`/`paramOpts`/`paramPrefix`); add static `id` / `label`; accept the params bag as the second constructor arg and store the reference |
| `src/render/debug/DebugView.ts` | add static `id = "debugView"` and `label = "Debug View"`; no params metadata needed (it has no live-tunable params today) |
| `src/params/ParamStore.ts` | extend `ParamValue` to `number \| boolean`, add `boolean` schema kind, update `validate()` accordingly |
| `src/params/ParamPanel.ts` | branch on `boolean` kind to use a checkbox widget |
| `src/params/WorkletBridge.ts` | skip non-numeric params (component toggles never go to the worklet) |
| `src/App.ts` | drop fixed `boxView`/`debugView` fields + per-component lifecycle calls; add registry-driven component manager (stable params bags, enable-subscription loop); `bindUI` iterates `COMPONENTS` and binds folder+checkbox+params per class |

### What does NOT change

- Worklet / DSP plumbing
- FeatureStore
- The existing render primitives (`TimeSeriesLineRenderer`, `BeatGridMarkers`, etc.)
- HMR wiring in `main.ts`
- `CameraRig`, key bindings, FPS overlay
- DSP param schemas in `src/params/schemas.ts`

## Rationale

**Full dispose / recreate on toggle (vs. visibility flag):** Cleanest mental model — "off" means *gone*, no hidden cost. BoxView's 1024-body RAPIER world stops stepping. DebugView's ~14 sub-renderers leave the scene and free their GPU resources. The re-init cost on toggle (BoxView's `await RAPIER.init()` is the slowest, and is idempotent so the second time is fast) is acceptable for an interactive toggle.

**ParamStore-persisted toggle state (vs. App-local):** The user's preferred component selection is exactly the kind of thing that should survive a reload. ParamStore already does localStorage, so booleans are a one-line extension — and the same machinery generalizes to future enum / int / string param kinds.

**Class IS the registry entry (vs. ComponentEntry wrapper objects):** Removes a layer. Metadata can't drift from the class because it lives on the class. Adding a component is an import + one array entry.

**Stable params bag owned by App (vs. component owns its own `params` object):** Tweakpane bindings hold a reference; dispose/recreate would silently desync the widgets. Putting the bag in App means widgets stay live across toggles and the user can pre-tweak params before enabling the component.

**Each component owns its own folder (vs. a separate "Components" folder of toggles):** Per-component cohesion — the enable checkbox and the component's params live in one place. Matches the mental model: "this folder is everything about this component."

## Open questions

None at design time.
