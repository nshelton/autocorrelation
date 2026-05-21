# Toggleable Visualizer Components

**Date:** 2026-05-20
**Status:** Approved (pending review)

## Motivation

The project is pivoting from audio-analysis-focused work into a broader visualizer system. Today `App.ts` hardcodes two views (`BoxView`, `DebugView`): instantiated inline, updated in the RAF loop, disposed in `dispose()`. Adding a new view means editing `App.ts` in four places, and there is no way to turn individual views off at runtime.

The goal is a small, uniform component system: each visual subsystem lives in its own file, declares its lifecycle, and is independently toggled on/off via a checkbox in the existing tweakpane panel. `App.ts` shrinks to the things only it can do (scene, camera, post-processing, key bindings, worklet message routing, ParamStore wiring).

## Non-goals

- Multiple scenes. All components continue to share the single `Scene` so the existing GTAO post-processing pipeline keeps working unchanged.
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
  update(dt: number): void;
  dispose(): void;
}

export interface ComponentEntry {
  id: string;       // also the ParamStore namespace for this component
  label: string;    // tweakpane folder title
  // Second arg is the App-owned params bag — present only if the component
  // declares static paramDefaults / paramOpts / paramPrefix.
  factory: (deps: ComponentDeps, params?: Record<string, number>) => Component;
  paramPrefix?: string;
  paramOpts?: Record<string, { min: number; max: number; step?: number }>;
  paramDefaults?: Record<string, number>;
}
```

A component is just a class with `update(dt)` and `dispose()`. It may *also* expose static `ViewWithParams` metadata (`paramPrefix`, `paramOpts`, `paramDefaults`) — that opt-in wires the component's own live-tunable params into ParamStore and tweakpane. Static rather than per-instance for a concrete reason: see "Stable param object across toggle cycles" below.

When a component exposes that metadata, its `paramPrefix` must equal its registry `id`. This lets one ParamStore namespace cover both the component's own params and its `enabled` toggle.

### Registry

`src/render/components/index.ts` (new) exports an ordered array:

```ts
import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import type { ComponentEntry } from "./Component";

export const COMPONENTS: ComponentEntry[] = [
  {
    id: "debugView",
    label: "Debug View",
    factory: (d) => new DebugView(d),
  },
  {
    id: "boxView",
    label: "Box View",
    factory: (d, params) => new BoxView(d, params!),
    paramPrefix: "boxView",
    paramOpts: BoxView.paramOpts,
    paramDefaults: BoxView.paramDefaults,
  },
];
```

Adding a new component: create its file under `src/render/components/`, import it here, append one line. `App.ts` does not import concrete component classes.

DebugView stays in `src/render/debug/DebugView.ts` because that folder also contains many sub-renderers (TimeSeriesRenderer, BeatGridMarkers, etc.) that are not themselves components. The registry just imports `DebugView` from there.

### ParamStore: boolean support

`ParamStore` is number-only today. Extend it to also hold booleans:

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

State per component: `{ entry: ComponentEntry; instance: Component | null }`.

On `start()`:
1. Build the shared `ComponentDeps` once.
2. For each `entry` in `COMPONENTS`, register a `components.<id>.enabled` schema in `ParamStore` (`kind: "boolean"`, `default: true`).
3. Read the current persisted value of each `enabled` key. If `true`, construct the component now and call `bindViewParams` if it implements `ViewWithParams`.
4. Subscribe to `ParamStore` for changes to any `components.*.enabled` key. On transition:
   - `true → false`: `instance.dispose()`, drop reference, also unbind that component's `ViewWithParams` subscriptions if any (so its UI updates stop firing).
   - `false → true`: construct via `factory`, call `bindViewParams` if applicable.
5. The per-frame loop iterates live instances and calls `update(dt)`.

On `dispose()`: dispose all live instances, unsubscribe.

App still owns: scene + camera + `CameraRig` + presets, `PostProcessing` pipeline (GTAO), keyboard/resize handlers, worklet `features` message routing into the store, FPS overlay, RAF loop. These are not components.

### bindUI / ParamPanel folder layout

For each `entry` in `COMPONENTS`:
1. `ParamPanel` adds a folder titled `entry.label`.
2. The first row in the folder is the `components.<id>.enabled` checkbox.
3. If the component implements `ViewWithParams`, the existing `bindViewParams` adds its params as subsequent rows in the same folder.

The folder is rendered even when the component is disabled. Params can be tweaked while disabled; they will take effect on next enable.

#### Stable param object across toggle cycles

Tweakpane bindings hold a *reference* to the object they read/write. Today `BoxView.params = { pull, timestep, width }` is created inside the component instance, so after a dispose→recreate the new instance owns a different object and the existing tweakpane widgets would silently stop working.

Fix: App owns one stable params bag per component-id, kept in a map alongside the instance state. On first registration, the bag is populated from the component's declared defaults (read once via a throwaway factory call, or by separating param metadata from the instance — see below). On every re-construction, App passes the bag into the factory, and the component mutates it in place each frame instead of holding its own `params` object.

Concretely, `ViewWithParams` evolves to:

```ts
export interface ViewWithParams {
  paramPrefix: string;
  paramOpts: Record<string, { min: number; max: number; step?: number }>;
  paramDefaults: Record<string, number>;
}
```

…and the registry entry for a component with params exposes those statically, e.g. as a property of the component class:

```ts
class BoxView implements Component {
  static paramPrefix = "boxView";
  static paramOpts = { pull: { min: 0, max: 1, step: 0.01 }, /* … */ };
  static paramDefaults = { pull: 0.3, timestep: 1/30, width: 0.5 };
  constructor(deps: ComponentDeps, params: Record<string, number>) { /* keep ref */ }
}
```

App reads the static metadata once at start to register schemas and build the bag, then injects the bag into the factory call each time the component is constructed.

Currently the DSP folder is added directly to the root pane by `ParamPanel`. With this change, each component gets its own folder added to the same root pane by `App.bindUI`. The existing "Reset to defaults" button at the root resets *all* params including the toggles — components transition off → on (or stay on) per their default of `true`.

### Files touched

| File | Change |
|---|---|
| `src/render/components/Component.ts` | **new** — interface definitions |
| `src/render/components/index.ts` | **new** — `COMPONENTS` registry |
| `src/render/components/BoxView.ts` | move `params` / `paramOpts` / `paramPrefix` from instance fields to static properties; rename to `paramDefaults`/`paramOpts`/`paramPrefix`; accept the params bag in the constructor and store the reference |
| `src/render/debug/DebugView.ts` | optional: add `paramPrefix = "debugView"` and empty `params` / `paramOpts` for symmetry; **not required** since `ViewWithParams` is opt-in |
| `src/params/ParamStore.ts` | extend `ParamValue` to `number \| boolean`, add `boolean` schema kind, validate accordingly |
| `src/params/ParamPanel.ts` | branch on `boolean` kind to use a checkbox widget |
| `src/params/WorkletBridge.ts` | skip non-numeric params (component toggles never go to the worklet) |
| `src/App.ts` | replace fixed `boxView`/`debugView` fields + lifecycle calls with a registry-driven component manager; `bindUI` iterates `COMPONENTS` |

### What does NOT change

- Worklet / DSP plumbing
- FeatureStore
- The existing render primitives (`TimeSeriesLineRenderer`, `BeatGridMarkers`, etc.)
- HMR wiring in `main.ts`
- `CameraRig`, key bindings, FPS overlay
- DSP param schemas in `src/params/schemas.ts`

## Rationale

**Why full dispose / recreate on toggle (vs. visibility flag):** Cleanest mental model — "off" means *gone*, no hidden cost. BoxView's 1024-body RAPIER world stops stepping. DebugView's ~14 sub-renderers leave the scene and free their GPU resources. The re-init cost on toggle (BoxView's `await RAPIER.init()` is the slowest, and is idempotent so the second time is fast) is acceptable for an interactive toggle.

**Why ParamStore-persisted toggle state (vs. App-local):** The user's preferred component selection is exactly the kind of thing that should survive a reload. ParamStore already does localStorage, so booleans are a one-line extension.

**Why a static registry array (vs. self-registering modules):** Static is easier to read, easier to reorder, and avoids import-order side effects. The codebase already favors direct imports over magic.

**Why each component owns its own folder (vs. a separate "Components" folder of toggles):** Per-component cohesion — the enable checkbox and the component's params live in one place. Matches the user's mental model: "this folder is everything about this component."

## Open questions

None at design time.
