# Param Modulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every continuous render/post/component/camera/light param modulatable at runtime by an audio source (rmsLow/Mid/High or beatPulses[0..3]), with depth 0..1 controlling how much the audio takes over. Slider remains the user's resting base; modulation rides on top.

**Architecture:** A `Modulator` engine runs each RAF frame, reads the latest sample of the bound source buffer from `FeatureStore`, computes `effective = lerp(base, lerp(min, max, src), depth)`, and pushes it through a new `ParamStore.notify()` that fires existing subscribers without persisting. Subscribers receive an extra `source: "user" | "modulator"` argument so the two UI-mirror sites can gate themselves; everything else (uniforms, camera.fov, etc.) treats both notifications identically. A `bindParam()` helper attaches an inline "↳ mod" sub-folder (source dropdown + depth slider) to every modulatable schema's tweakpane widget.

**Tech Stack:** TypeScript strict, tweakpane v4, vitest + happy-dom for tests.

**Spec reference:** `docs/superpowers/specs/2026-05-24-param-modulation-design.md`

**Plan refinement vs spec:** The spec stated "no consumer migration" via `notify()`. In practice, two UI-mirror sites (`ParamPanel` and `App.bindCameraUI`) would visibly track modulation if we didn't disambiguate. We add a third positional arg `source` to the subscriber signature; only those two sites gate on it. Other consumers (uniform setters, camera.fov assignment, `ComponentManager` paramsBag mirror) treat both identically and naturally pick up modulation. Adding a positional arg is type-compatible with existing `(key, value) =>` callbacks — none break.

---

## File Structure

**New files:**
- `src/params/Modulator.ts` — the runtime engine, plus the `MOD_SOURCES` table.
- `src/params/bindParam.ts` — the tweakpane helper.
- `tests/params/Modulator.test.ts` — unit tests for the engine.
- `tests/params/ParamStore.test.ts` — unit tests for `notify()` + `schemaFor()` (new test file; ParamStore had no dedicated test).

**Modified:**
- `src/params/ParamStore.ts` — add `notify()`, `schemaFor()`, and `source` arg.
- `src/params/ParamPanel.ts` — refactor to use `bindParam`; add "Reset modulation" button; gate `pane.refresh()` on `source === "user"`.
- `src/App.ts` — construct `Modulator`, call `tick()` in RAF loop, pass to `components.bindUI` and `postStack.bindUI`, gate camera-UI mirrors on `source === "user"`, migrate camera/light bindings to `bindParam`, dispose on teardown.
- `src/render/post/PostStack.ts` — `bindUI(folder, modulator)` signature; pass modulator down.
- `src/render/post/PostEffect.ts` — `bindUI(folder, store, modulator)` signature.
- `src/render/post/effects/BloomEffect.ts` — refactor `bindUI` to use `bindParam`.
- `src/render/post/effects/AoEffect.ts` — same.
- `src/render/post/effects/TonemapEffect.ts` — same.
- `src/render/components/ComponentManager.ts` — `bindUI(parent, modulator)` signature; replace per-param `addBinding` calls with `bindParam`.

**Worktree note:** This work touches enough files across `src/params/`, `src/App.ts`, `src/render/post/`, and `src/render/components/` that the executing skill should spin up an isolated worktree via `superpowers:using-git-worktrees`.

---

## Task 1: ParamStore — add `notify()`, `schemaFor()`, and `source` arg

**Files:**
- Modify: `src/params/ParamStore.ts`
- Test: `tests/params/ParamStore.test.ts` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/params/ParamStore.test.ts`:

```ts
import { describe, it, expect, beforeEach } from "vitest";
import { ParamStore, type ParamSchema } from "../../src/params/ParamStore";

const FOV_SCHEMA: ParamSchema = {
  key: "camera.fov",
  label: "FOV",
  kind: "continuous",
  min: 20,
  max: 120,
  step: 1,
  default: 60,
  reconfig: false,
};

describe("ParamStore", () => {
  let store: ParamStore;
  beforeEach(() => {
    localStorage.clear();
    store = new ParamStore();
    store.register(FOV_SCHEMA);
  });

  it("set() fires subscribers with source='user'", () => {
    const calls: Array<[string, unknown, string]> = [];
    store.subscribe((k, v, s) => calls.push([k, v, s]));
    store.set("camera.fov", 75);
    expect(calls).toEqual([["camera.fov", 75, "user"]]);
  });

  it("notify() fires subscribers with source='modulator' without mutating value or persisting", () => {
    store.set("camera.fov", 75);
    const calls: Array<[string, unknown, string]> = [];
    store.subscribe((k, v, s) => calls.push([k, v, s]));
    store.notify("camera.fov", 90);
    expect(calls).toEqual([["camera.fov", 90, "modulator"]]);
    expect(store.get("camera.fov")).toBe(75);
    expect(JSON.parse(localStorage.getItem("autocorrelation.params.v1")!))
      .toEqual({ "camera.fov": 75 });
  });

  it("schemaFor() returns the schema by key", () => {
    expect(store.schemaFor("camera.fov")).toBe(FOV_SCHEMA);
    expect(store.schemaFor("does.not.exist")).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run tests/params/ParamStore.test.ts`
Expected: 3 failures — `notify` and `schemaFor` undefined; subscriber callback receives only 2 args.

- [ ] **Step 3: Implement**

In `src/params/ParamStore.ts`:

Update the `Subscriber` type:
```ts
type Subscriber = (key: string, value: ParamValue, source: "user" | "modulator") => void;
```

Update `set()` to pass source:
```ts
set(key: string, value: ParamValue): void {
  const schema = this.schemas.get(key);
  if (!schema) throw new Error(`ParamStore: unknown key ${key}`);
  if (!this.validate(schema, value)) {
    console.warn(`ParamStore: rejected ${key}=${value} (out of range or wrong type)`);
    return;
  }
  this.values.set(key, value);
  this.writePersisted();
  for (const fn of this.subscribers) fn(key, value, "user");
}
```

Update `reset()` to pass source:
```ts
for (const [key, value] of changed) {
  for (const fn of this.subscribers) fn(key, value, "user");
}
```

Add the new methods (place after `subscribe`):
```ts
notify(key: string, value: ParamValue): void {
  for (const fn of this.subscribers) fn(key, value, "modulator");
}

schemaFor(key: string): ParamSchema | undefined {
  return this.schemas.get(key);
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `npx vitest run tests/params/ParamStore.test.ts`
Expected: 3 passing.

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `npm test`
Expected: all green. Existing 2-arg subscribers compile because the 3rd arg is positional and optional from the caller's POV (TS function-type contravariance allows narrower-arity functions to satisfy wider-arity callbacks).

- [ ] **Step 6: Commit**

```bash
git add src/params/ParamStore.ts tests/params/ParamStore.test.ts
git commit -m "feat(params): ParamStore.notify() + schemaFor() + source arg on subscribers"
```

---

## Task 2: Modulator core + persistence

**Files:**
- Create: `src/params/Modulator.ts`
- Test: `tests/params/Modulator.test.ts` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/params/Modulator.test.ts`:

```ts
import { describe, it, expect, beforeEach, vi } from "vitest";
import { ParamStore, type ParamSchema } from "../../src/params/ParamStore";
import { FeatureStore } from "../../src/store/FeatureStore";
import { Modulator } from "../../src/params/Modulator";

const FOV: ParamSchema = {
  key: "camera.fov",
  label: "FOV",
  kind: "continuous",
  min: 20,
  max: 120,
  step: 1,
  default: 60,
  reconfig: false,
};

const STR: ParamSchema = {
  key: "post.bloom.strength",
  label: "Strength",
  kind: "continuous",
  min: 0,
  max: 3,
  step: 0.01,
  default: 0.5,
  reconfig: false,
};

function setup() {
  localStorage.clear();
  const store = new ParamStore();
  store.register(FOV);
  store.register(STR);
  const features = new FeatureStore();
  const mod = new Modulator(store, features);
  return { store, features, mod };
}

describe("Modulator", () => {
  beforeEach(() => localStorage.clear());

  it("tick() with no bindings fires nothing", () => {
    const { store, mod } = setup();
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    expect(spy).not.toHaveBeenCalled();
  });

  it("tick() with depth=0 fires notify(key, base)", () => {
    const { store, features, mod } = setup();
    features.set("rmsLow", new Float32Array([0, 0.8]));
    mod.setBinding("camera.fov", { source: "rms.low", depth: 0 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    expect(spy).toHaveBeenCalledWith("camera.fov", 60, "modulator");
  });

  it("tick() with depth=1 fires notify(key, lerp(min,max,src))", () => {
    const { store, features, mod } = setup();
    features.set("rmsLow", new Float32Array([0, 0.25]));
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    // lerp(20, 120, 0.25) = 45
    expect(spy).toHaveBeenCalledWith("camera.fov", 45, "modulator");
  });

  it("tick() with NaN source fires notify(key, base)", () => {
    const { store, features, mod } = setup();
    features.set("beatPulses", new Float32Array([NaN, NaN, NaN, NaN]));
    mod.setBinding("post.bloom.strength", { source: "beat.1x", depth: 1 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    expect(spy).toHaveBeenCalledWith("post.bloom.strength", 0.5, "modulator");
  });

  it("tick() with empty source buffer fires notify(key, base)", () => {
    const { store, mod } = setup();
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    expect(spy).toHaveBeenCalledWith("camera.fov", 60, "modulator");
  });

  it("beat sources read indexed slot of beatPulses", () => {
    const { store, features, mod } = setup();
    features.set("beatPulses", new Float32Array([0.1, 0.4, 0.7, 1.0]));
    mod.setBinding("camera.fov", { source: "beat.4x", depth: 1 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    // lerp(20,120,0.7) = 90
    expect(spy).toHaveBeenCalledWith("camera.fov", 90, "modulator");
  });

  it("setBinding(key, null) removes binding and fires one notify(key, base)", () => {
    const { store, features, mod } = setup();
    features.set("rmsLow", new Float32Array([1.0]));
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.setBinding("camera.fov", null);
    expect(spy).toHaveBeenCalledWith("camera.fov", 60, "modulator");
    spy.mockClear();
    mod.tick();
    expect(spy).not.toHaveBeenCalled();
  });

  it("persists bindings across instances", () => {
    const { store, features, mod } = setup();
    mod.setBinding("camera.fov", { source: "rms.high", depth: 0.42 });
    const mod2 = new Modulator(store, features);
    expect(mod2.getBinding("camera.fov")).toEqual({ source: "rms.high", depth: 0.42 });
  });

  it("drops persisted binding with unknown source on load", () => {
    localStorage.setItem(
      "autocorrelation.modulation.v1",
      JSON.stringify({ "camera.fov": { source: "totally.fake", depth: 1 } }),
    );
    const store = new ParamStore();
    store.register(FOV);
    const features = new FeatureStore();
    const mod = new Modulator(store, features);
    expect(mod.getBinding("camera.fov")).toBeNull();
  });

  it("drops persisted binding with unknown paramKey on load", () => {
    localStorage.setItem(
      "autocorrelation.modulation.v1",
      JSON.stringify({ "nonexistent.key": { source: "rms.low", depth: 1 } }),
    );
    const store = new ParamStore();
    store.register(FOV);
    const features = new FeatureStore();
    const mod = new Modulator(store, features);
    expect(mod.getBinding("nonexistent.key")).toBeNull();
  });

  it("subscribe() fires on setBinding changes", () => {
    const { mod } = setup();
    const spy = vi.fn();
    mod.subscribe(spy);
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1 });
    expect(spy).toHaveBeenCalledWith("camera.fov");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run tests/params/Modulator.test.ts`
Expected: module-not-found errors (Modulator.ts doesn't exist yet).

- [ ] **Step 3: Implement Modulator**

Create `src/params/Modulator.ts`:

```ts
import type { ParamStore } from "./ParamStore";
import type { FeatureStore } from "../store/FeatureStore";

const STORAGE_KEY = "autocorrelation.modulation.v1";

export interface ModBinding {
  source: string;
  depth: number;
}

type SourceDescriptor = {
  buffer: string;
  read: (b: Float32Array) => number;
};

function latest(b: Float32Array): number {
  return b.length === 0 ? 0 : b[b.length - 1];
}

// Curated audio sources. UI dropdown + persistence use these string keys.
// Beat saws read fixed indices of beatPulses (1x/2x/4x/8x phase, 0..1 saw).
export const MOD_SOURCES: Record<string, SourceDescriptor> = {
  "rms.low":  { buffer: "rmsLow",     read: latest },
  "rms.mid":  { buffer: "rmsMid",     read: latest },
  "rms.high": { buffer: "rmsHigh",    read: latest },
  "beat.1x":  { buffer: "beatPulses", read: (b) => (b.length > 0 ? b[0] : 0) },
  "beat.2x":  { buffer: "beatPulses", read: (b) => (b.length > 1 ? b[1] : 0) },
  "beat.4x":  { buffer: "beatPulses", read: (b) => (b.length > 2 ? b[2] : 0) },
  "beat.8x":  { buffer: "beatPulses", read: (b) => (b.length > 3 ? b[3] : 0) },
};

export const MOD_SOURCE_KEYS = Object.keys(MOD_SOURCES);

type UiSubscriber = (key: string) => void;

export class Modulator {
  private bindings = new Map<string, ModBinding>();
  private uiSubs = new Set<UiSubscriber>();

  constructor(
    private store: ParamStore,
    private features: FeatureStore,
  ) {
    this.load();
  }

  setBinding(key: string, binding: ModBinding | null): void {
    if (binding === null) {
      if (!this.bindings.delete(key)) return;
      this.persist();
      // Snap consumer state back to the slider value.
      const base = this.store.get(key);
      if (typeof base === "number") this.store.notify(key, base);
      for (const fn of this.uiSubs) fn(key);
      return;
    }
    this.bindings.set(key, { ...binding });
    this.persist();
    for (const fn of this.uiSubs) fn(key);
  }

  getBinding(key: string): ModBinding | null {
    const b = this.bindings.get(key);
    return b ? { ...b } : null;
  }

  tick(): void {
    for (const [key, b] of this.bindings) {
      const schema = this.store.schemaFor(key);
      if (!schema || schema.kind !== "continuous") continue;
      const src = MOD_SOURCES[b.source];
      if (!src) continue;
      const buf = this.features.get(src.buffer);
      let v = src.read(buf);
      if (!Number.isFinite(v)) v = 0;
      const base = this.store.get(key) as number;
      const target = schema.min + (schema.max - schema.min) * v;
      const eff = base + (target - base) * b.depth;
      this.store.notify(key, eff);
    }
  }

  subscribe(fn: UiSubscriber): () => void {
    this.uiSubs.add(fn);
    return () => this.uiSubs.delete(fn);
  }

  dispose(): void {
    this.uiSubs.clear();
  }

  private load(): void {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(STORAGE_KEY);
    } catch {
      return;
    }
    if (!raw) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return;
    }
    if (!parsed || typeof parsed !== "object") return;
    for (const [key, val] of Object.entries(parsed as Record<string, unknown>)) {
      if (!val || typeof val !== "object") continue;
      const candidate = val as { source?: unknown; depth?: unknown };
      if (typeof candidate.source !== "string") continue;
      if (typeof candidate.depth !== "number") continue;
      if (!(candidate.source in MOD_SOURCES)) continue;
      if (!this.store.schemaFor(key)) continue;
      this.bindings.set(key, { source: candidate.source, depth: candidate.depth });
    }
  }

  private persist(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.bindings)));
    } catch {
      // localStorage unavailable; no-op
    }
  }
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `npx vitest run tests/params/Modulator.test.ts`
Expected: 10 passing.

- [ ] **Step 5: Commit**

```bash
git add src/params/Modulator.ts tests/params/Modulator.test.ts
git commit -m "feat(params): Modulator engine + persistence + MOD_SOURCES table"
```

---

## Task 3: Gate UI-mirror sites + WorkletBridge on source==="user"

**Files:**
- Modify: `src/params/ParamPanel.ts` (subscribe block, ~lines 29-35)
- Modify: `src/App.ts` (cameraUnsub block, ~lines 229-243)
- Modify: `src/params/WorkletBridge.ts` (handleChange call site)

ParamPanel and App.bindCameraUI are the only two places that call `pane.refresh()` or mutate UI-proxy fields in a way that would visibly track modulation. Other subscribers (uniform setters, ComponentManager paramsBag mirror) want both base and modulated values and need no change.

`WorkletBridge` doesn't touch UI, but if a hand-crafted localStorage modulation binding ever targeted a `dsp.*` key (no UI path creates one; persistence load drops unknown keys; defense in depth), `notify()` would fire `WorkletBridge.handleChange` per frame and flood the worklet with param messages. Cheap to gate on `source === "user"`.

- [ ] **Step 1: Update ParamPanel.subscribe to ignore modulator source**

In `src/params/ParamPanel.ts`, replace the existing subscribe block (around lines 29-35):

```ts
this.unsubscribe = store.subscribe((key, value, source) => {
  if (source !== "user") return;
  if (!key.startsWith("dsp.")) return;
  if (this.bindings[key] !== value) {
    this.bindings[key] = value;
    this.pane.refresh();
  }
});
```

- [ ] **Step 2: Update App.bindCameraUI subscriber to ignore modulator source**

In `src/App.ts`, replace the existing `this.cameraUnsub = store.subscribe(...)` block (around lines 229-243):

```ts
this.cameraUnsub = store.subscribe((key, value, source) => {
  if (key === "camera.fov" && typeof value === "number") {
    camera.fov = value;
    camera.updateProjectionMatrix();
    if (source === "user") fovBinding.fov = value;
  } else if (key === "camera.preset" && typeof value === "number") {
    if (source === "user") {
      const name = CAMERA_PRESET_NAMES[value];
      if (name) void this.rig.goTo(name, { duration: 0.8 });
      presetBinding.preset = value;
    }
  } else if (key === "light.directional.enabled" && typeof value === "boolean") {
    if (value) this.scene.add(this.directionalLight);
    else this.scene.remove(this.directionalLight);
    if (source === "user") lightBinding.enabled = value;
  }
});
```

Note: `camera.fov` is the only continuous key here that will ever be modulated, so the camera-uniform write (`camera.fov = value; camera.updateProjectionMatrix()`) runs for both sources. The discrete preset + boolean light cases get fully gated since their consumer-side state (rig goto, scene add/remove) is also gated — they aren't modulatable and would never fire from the modulator, but defensive gating is cheap.

- [ ] **Step 3: Gate WorkletBridge on source==="user"**

In `src/params/WorkletBridge.ts`, update the constructor's subscribe call (around line 22):
```ts
this.unsubscribe = store.subscribe((key, _value, source) => {
  if (source !== "user") return;
  this.handleChange(key);
});
```

- [ ] **Step 4: Run full test suite + typecheck via build**

Run: `npm test && npm run build`
Expected: green tests, clean build.

- [ ] **Step 5: Commit**

```bash
git add src/params/ParamPanel.ts src/App.ts src/params/WorkletBridge.ts
git commit -m "feat(params): gate UI mirror sites + WorkletBridge on source=user"
```

---

## Task 4: bindParam helper

**Files:**
- Create: `src/params/bindParam.ts`

This is the tweakpane helper. It wraps the recurring `addBinding(...).on("change", store.set)` pattern, attaches an inline mod sub-folder for continuous non-`dsp.*` schemas, and subscribes to the modulator for two-way sync.

- [ ] **Step 1: Implement bindParam**

Create `src/params/bindParam.ts`:

```ts
import type { FolderApi } from "tweakpane";
import { ParamStore, type ParamSchema, type ParamValue } from "./ParamStore";
import { Modulator, MOD_SOURCE_KEYS } from "./Modulator";

// Sentinel value for "no modulation" in the source dropdown.
const NONE = "none";

// Optional map a caller can pass to collect "re-pull from store into proxy"
// callbacks per param key. Only ParamPanel uses this — see Task 6. Other
// call sites omit it (matches current behavior: their sliders don't
// visually refresh on external writes like Reset).
export type ParamProxyRegistry = Map<string, () => void>;

// Wraps the recurring `folder.addBinding(proxy, ...).on("change", store.set)`
// pattern. For continuous, non-dsp.* schemas, also appends a collapsed
// `↳ mod` sub-folder with a source dropdown + depth slider that drive
// Modulator.setBinding.
//
// The visible slider/dropdown/checkbox is bound to a LOCAL proxy object
// owned by this helper's closure. Tweakpane writes the user's drag into
// the proxy and fires `change`; we forward to store.set.
//
// External writes to the store (e.g. Reset to defaults) do NOT auto-refresh
// the proxy by default. Callers that need that — currently only ParamPanel
// for dsp.* sliders — pass a `ParamProxyRegistry` Map and bindParam
// registers a refresh callback for the key. The caller then calls the
// callback + `pane.refresh()` from its own subscriber.
//
// Modulator `notify()` calls DO NOT touch the proxy — see ParamStore.notify
// docs and Task 3 — so the slider stays anchored to the base value when
// modulation is active.
//
// Modulator subscription is added below; it lives until the Modulator is
// disposed (which happens in App.dispose, before panel teardown). No HMR
// leak because the modulator is recreated each cycle.
export function bindParam(
  folder: FolderApi,
  store: ParamStore,
  modulator: Modulator,
  schema: ParamSchema,
  refreshRegistry?: ParamProxyRegistry,
): void {
  const proxy: { value: ParamValue } = { value: store.get(schema.key) };
  const binding = makeWidget(folder, proxy, schema);
  binding.on("change", (e: { value: ParamValue }) => store.set(schema.key, e.value));

  refreshRegistry?.set(schema.key, () => {
    proxy.value = store.get(schema.key);
  });

  const modulatable =
    schema.kind === "continuous" && !schema.key.startsWith("dsp.");
  if (!modulatable) return;

  const sub = folder.addFolder({ title: "↳ mod", expanded: false });

  const existing = modulator.getBinding(schema.key);
  const modProxy = {
    source: existing?.source ?? NONE,
    depth: existing?.depth ?? 0,
  };

  const sourceOptions: Record<string, string> = { [NONE]: NONE };
  for (const k of MOD_SOURCE_KEYS) sourceOptions[k] = k;

  const sourceBinding = sub.addBinding(modProxy, "source", {
    label: "source",
    options: sourceOptions,
  });
  const depthBinding = sub.addBinding(modProxy, "depth", {
    label: "depth",
    min: 0,
    max: 1,
    step: 0.01,
  });

  const writeBinding = () => {
    if (modProxy.source === NONE) {
      modulator.setBinding(schema.key, null);
    } else {
      modulator.setBinding(schema.key, {
        source: modProxy.source,
        depth: modProxy.depth,
      });
    }
  };
  sourceBinding.on("change", writeBinding);
  depthBinding.on("change", writeBinding);

  // Two-way sync from modulator changes (e.g. persisted-on-load).
  modulator.subscribe((key) => {
    if (key !== schema.key) return;
    const current = modulator.getBinding(schema.key);
    modProxy.source = current?.source ?? NONE;
    modProxy.depth = current?.depth ?? 0;
    sub.refresh();
  });
}

function makeWidget(
  folder: FolderApi,
  proxy: { value: ParamValue },
  schema: ParamSchema,
) {
  if (schema.kind === "boolean") {
    return folder.addBinding(proxy, "value", { label: schema.label });
  }
  if (schema.kind === "discrete") {
    const labels = schema.optionLabels ?? schema.options.map(String);
    return folder.addBinding(proxy, "value", {
      label: schema.label,
      options: Object.fromEntries(schema.options.map((v, i) => [labels[i], v])),
    });
  }
  return folder.addBinding(proxy, "value", {
    label: schema.label,
    min: schema.min,
    max: schema.max,
    step: schema.step,
  });
}
```

- [ ] **Step 2: Confirm it typechecks**

Run: `npm run build`
Expected: clean. (The helper isn't imported anywhere yet — this step just catches TS errors in the new file.)

- [ ] **Step 3: Commit**

```bash
git add src/params/bindParam.ts
git commit -m "feat(params): bindParam helper with inline mod sub-folder"
```

---

## Task 5: Wire Modulator into App lifecycle

**Files:**
- Modify: `src/App.ts`

- [ ] **Step 1: Add modulator field and construction**

In `src/App.ts`:

Add import near the existing `ParamStore` type import:
```ts
import { Modulator } from "./params/Modulator";
```

Add field declaration near the other private fields (around line 67). Use `public` since `main.ts` will need to read it in Task 6 to wire up the ParamPanel — matches existing public-field style on `App`:
```ts
public modulator!: Modulator;
```

In `start()`, immediately after the `this.components = new ComponentManager(...)` block but BEFORE `this.components.start();`, construct the modulator:
```ts
this.modulator = new Modulator(paramStore, this.store);
```

(`this.store` is the `FeatureStore` already on the class.)

- [ ] **Step 2: Call modulator.tick() in the RAF loop**

In the `loop` function inside `start()`, add `this.modulator.tick();` immediately before `this.components.update();`:

```ts
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
```

- [ ] **Step 3: Dispose modulator on teardown**

In `dispose()`, add `this.modulator?.dispose();` after `this.components?.dispose();`:

```ts
this.components?.dispose();
this.modulator?.dispose();
this.postStack?.dispose();
```

- [ ] **Step 4: Run full test suite + build**

Run: `npm test && npm run build`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/App.ts
git commit -m "feat(app): construct + tick + dispose Modulator in App lifecycle"
```

---

## Task 6: Migrate ParamPanel widgets to bindParam + add Reset modulation button

**Files:**
- Modify: `src/params/ParamPanel.ts`

ParamPanel currently renders only `dsp.*` params, all non-modulatable. We still route through `bindParam` to unify the path (and the helper correctly skips the mod sub-folder for `dsp.*`). The "Reset modulation" button is global, so it lives on ParamPanel.

- [ ] **Step 1: Refactor to use bindParam**

Replace the entire body of `src/params/ParamPanel.ts` with:

```ts
import { FolderApi, Pane } from "tweakpane";
import { ParamStore } from "./ParamStore";
import { Modulator } from "./Modulator";
import { bindParam, type ParamProxyRegistry } from "./bindParam";

export class ParamPanel {
  public pane: Pane;
  public scenes: FolderApi;
  public camera: FolderApi;
  public post: FolderApi;
  private unsubscribe: () => void;
  private proxies: ParamProxyRegistry = new Map();

  constructor(store: ParamStore, modulator: Modulator, container?: HTMLElement) {
    this.pane = new Pane({ container });
    const folder = this.pane.addFolder({ title: "Analysis", expanded: false });

    // ParamPanel owns the DSP folder only. Component-toggle and
    // component-param schemas are rendered by ComponentManager.bindUI()
    // into their own per-component folders. We pass `this.proxies` so
    // bindParam registers a "re-pull proxy from store" callback per key;
    // the subscriber below calls those + pane.refresh() to restore the
    // existing Reset-snaps-sliders behavior for dsp.* params.
    for (const schema of store.schemasInOrder()) {
      if (!schema.key.startsWith("dsp.")) continue;
      bindParam(folder, store, modulator, schema, this.proxies);
    }

    // Gated on source==='user' so per-frame modulator notifies don't
    // jitter the UI.
    this.unsubscribe = store.subscribe((key, _value, source) => {
      if (source !== "user") return;
      const refresh = this.proxies.get(key);
      if (!refresh) return;
      refresh();
      this.pane.refresh();
    });

    this.scenes = this.pane.addFolder({ title: "Scenes" });
    this.camera = this.pane.addFolder({ title: "Camera", expanded: false });
    this.post = this.pane.addFolder({ title: "Post", expanded: false });
    this.pane.addButton({ title: "Reset to defaults" }).on("click", () => store.reset());
    this.pane.addButton({ title: "Reset modulation" }).on("click", () => {
      for (const schema of store.schemasInOrder()) {
        if (modulator.getBinding(schema.key)) modulator.setBinding(schema.key, null);
      }
    });
  }

  dispose(): void {
    this.unsubscribe();
    this.proxies.clear();
    this.pane.dispose();
  }
}
```

No HMR leak: only the single `unsubscribe` lives in `ParamStore.subscribers`, removed on dispose. The proxies are held by the registry (owned by the panel) and by bindParam closures (held by the disposed tweakpane bindings) — both released after `pane.dispose()`.

- [ ] **Step 2: Reorder main.ts construction so the panel is built after the modulator**

`main.ts` currently constructs `ParamPanel` BEFORE `app.start()` (which is where Task 5 puts the Modulator construction). Task 5 declared `App.modulator` as a public field; now reorder `main.ts` so the panel is built after `app.start()` and can read `app.modulator`.

**Edit `src/main.ts`:** update `buildAppLayer` (around lines 126-138). Replace:
```ts
function buildAppLayer(deps: AppDeps): void {
  app = new AppCtor(deps);
  panel = new ParamPanelCtor(deps.paramStore);
  bridge = new WorkletBridgeCtor(deps.paramStore, deps.workletNode.port);
  app.start();
  app.bindUI(panel.scenes);
  app.bindCameraUI(panel.camera);
  app.bindPostUI(panel.post);
  if (initialBootstrap) {
    bridge.bootstrap();
    initialBootstrap = false;
  }
}
```

with:
```ts
function buildAppLayer(deps: AppDeps): void {
  app = new AppCtor(deps);
  app.start();                                          // constructs modulator
  panel = new ParamPanelCtor(deps.paramStore, app.modulator);
  bridge = new WorkletBridgeCtor(deps.paramStore, deps.workletNode.port);
  app.bindUI(panel.scenes);
  app.bindCameraUI(panel.camera);
  app.bindPostUI(panel.post);
  if (initialBootstrap) {
    bridge.bootstrap();
    initialBootstrap = false;
  }
}
```

Teardown ordering in `teardownAppLayer` is unchanged — `app.dispose()` clears the modulator's UI subscribers via `Modulator.dispose()`, then `panel.dispose()` tears down tweakpane. Both safe.

- [ ] **Step 3: Run build + dev server smoke**

Run: `npm run build`
Expected: clean.

Run: `npm run dev` (background). Open http://localhost:5173. Verify:
- Analysis folder still shows the 11 DSP params.
- No "↳ mod" sub-folder under any DSP param (because `dsp.*` is filtered).
- Reset to defaults still works.
- Reset modulation button exists (does nothing visible yet).

- [ ] **Step 4: Commit**

```bash
git add src/params/ParamPanel.ts src/main.ts src/App.ts
git commit -m "feat(panel): ParamPanel uses bindParam; add Reset modulation button"
```

---

## Task 7: Migrate App.bindCameraUI to bindParam

**Files:**
- Modify: `src/App.ts` (bindCameraUI, ~lines 201-248)

- [ ] **Step 1: Replace the three hand-rolled bindings with bindParam**

In `src/App.ts`, replace the body of `bindCameraUI()` with:

```ts
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

  // Side-effects subscriber (kept separate from UI proxy, which bindParam
  // owns). `camera.fov` is the only key here that may be modulated, so its
  // camera-uniform write runs on every notify (no source gate). The other
  // two keys (preset, light) are not modulatable; their side-effects also
  // run on both sources defensively, and are idempotent.
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
```

Add import at the top of `src/App.ts`:
```ts
import { bindParam } from "./params/bindParam";
```

- [ ] **Step 2: Manual smoke**

Run: `npm run dev` and verify:
- Camera folder shows FOV slider, with a collapsed `↳ mod` sub-folder underneath.
- Expanding the sub-folder shows `source` (defaults to `none`) and `depth` (default 0).
- Setting source to `rms.low` and depth to 0.5 — start audio (press T for test source), then play something — FOV should now wobble with the rms.low signal.
- Preset dropdown still works.
- Light toggle still works.
- No `↳ mod` sub-folder under preset or light (discrete/boolean).

- [ ] **Step 3: Commit**

```bash
git add src/App.ts
git commit -m "feat(camera-ui): migrate bindCameraUI to bindParam (FOV is now modulatable)"
```

---

## Task 8: Migrate post effects to bindParam

**Files:**
- Modify: `src/render/post/PostEffect.ts` (interface signature)
- Modify: `src/render/post/PostStack.ts` (bindUI signature)
- Modify: `src/render/post/effects/BloomEffect.ts`
- Modify: `src/render/post/effects/AoEffect.ts`
- Modify: `src/render/post/effects/TonemapEffect.ts`
- Modify: `src/App.ts` (bindPostUI call site)

- [ ] **Step 1: Widen the PostEffect.bindUI interface**

In `src/render/post/PostEffect.ts`, change the signature:
```ts
// Add import at the top:
import type { Modulator } from "../../params/Modulator";

// In the interface:
bindUI(folder: FolderApi, store: ParamStore, modulator: Modulator): void;
```

- [ ] **Step 2: Update PostStack.bindUI**

In `src/render/post/PostStack.ts`, change the signature to accept a modulator and pass it through:
```ts
// Add import:
import type { Modulator } from "../../params/Modulator";

// Replace bindUI:
bindUI(folder: FolderApi, modulator: Modulator): void {
  for (const effect of this.effects) {
    const sub = folder.addFolder({ title: effect.label, expanded: false });
    effect.bindUI(sub, this.store, modulator);
  }
}
```

- [ ] **Step 3: Refactor BloomEffect.bindUI**

In `src/render/post/effects/BloomEffect.ts`, replace `bindUI`:
```ts
// Add imports at the top:
import type { Modulator } from "../../../params/Modulator";
import { bindParam } from "../../../params/bindParam";

// Replace bindUI:
bindUI(folder: FolderApi, store: ParamStore, modulator: Modulator): void {
  for (const key of [
    "post.bloom.enabled",
    "post.bloom.strength",
    "post.bloom.radius",
    "post.bloom.threshold",
  ]) {
    const schema = store.schemaFor(key);
    if (!schema) throw new Error(`BloomEffect.bindUI: schema ${key} missing`);
    bindParam(folder, store, modulator, schema);
  }
}
```

- [ ] **Step 4: Refactor AoEffect.bindUI and TonemapEffect.bindUI**

Read each file first, then apply the same pattern as BloomEffect: import `Modulator` and `bindParam`, replace the `bindUI` body with a loop over the effect's param keys, calling `bindParam` for each. Tonemap has a discrete `mode` key (gets no mod sub-folder) and a boolean `enabled` (no mod sub-folder either) — bindParam handles those correctly via the kind/dsp-prefix check.

Tonemap's keys: `post.tonemap.enabled`, `post.tonemap.mode`, `post.tonemap.exposure`.
AO's keys: `post.ao.enabled`, `post.ao.radius`, `post.ao.intensity`.

- [ ] **Step 5: Update App.bindPostUI**

In `src/App.ts`, update the method:
```ts
bindPostUI(folder: FolderApi): void {
  this.postStack.bindUI(folder, this.modulator);
}
```

- [ ] **Step 6: Smoke test**

Run: `npm test && npm run build && npm run dev`

In the browser:
- Open the Post folder. Each effect (AO, Bloom, Tonemap) opens to show the same widgets as before.
- Continuous params (AO radius/intensity, Bloom strength/radius/threshold, Tonemap exposure) each have a `↳ mod` sub-folder.
- Booleans (enabled toggles) and the Tonemap mode dropdown have no `↳ mod` sub-folder.
- Bind Bloom strength to `beat.1x` with depth 0.7. Play a beat-heavy track. Bloom strength should pulse on each beat.

- [ ] **Step 7: Commit**

```bash
git add src/render/post/PostEffect.ts src/render/post/PostStack.ts src/render/post/effects/*.ts src/App.ts
git commit -m "feat(post): migrate post effects to bindParam (all continuous post params modulatable)"
```

---

## Task 9: Migrate ComponentManager.bindUI to bindParam

**Files:**
- Modify: `src/render/components/ComponentManager.ts`
- Modify: `src/App.ts` (the call to `this.components.bindUI(folder)`)

- [ ] **Step 1: Update ComponentManager.bindUI signature**

In `src/render/components/ComponentManager.ts`:

Add import at the top:
```ts
import { Modulator } from "../../params/Modulator";
import { bindParam } from "../../params/bindParam";
```

Change the signature:
```ts
bindUI(parent: FolderApi, modulator: Modulator): void {
```

- [ ] **Step 2: Replace per-param addBinding calls with bindParam**

Inside `bindUI`, replace the for-loop body that iterates `allKeys` (around lines 136-163). The new body resolves each key's schema (registered in `start()` via `paramStore.register`), then delegates to `bindParam`. The existing `paramStore.subscribe` in `start()` already mirrors notifications into `slot.paramsBag`, so the instance picks up modulated values automatically.

Replace lines 136-163 with:
```ts
const allKeys = new Set<string>([
  ...Object.keys(slot.cls.paramOpts ?? {}),
  ...Object.keys(slot.cls.paramDefaults ?? {}),
]);
for (const k of allKeys) {
  const fullKey = `${slot.cls.paramPrefix ?? slot.cls.id}.${k}`;
  const schema = paramStore.schemaFor(fullKey);
  if (!schema) continue;
  bindParam(folder, paramStore, modulator, schema);
}
```

The enable-checkbox path (lines 83-101), per-component reset button (lines 112-119), and paramButtons block (lines 121-129) all stay as-is. They are independent of the per-param binding migration.

- [ ] **Step 3: Update App.bindUI to pass the modulator**

In `src/App.ts`:
```ts
bindUI(parent: import("tweakpane").FolderApi): void {
  this.components.bindUI(parent, this.modulator);
}
```

- [ ] **Step 4: Verify the existing ComponentManager paramsBag mirror still works correctly under modulation**

This is a code-inspection check, not a code change. In `ComponentManager.start()` (lines 51-69), the existing subscriber mirrors store notifications into `slot.paramsBag`. The Modulator fires `store.notify(key, eff)` per frame → the mirror writes `slot.paramsBag[localKey] = eff` → the live instance (which reads from the bag in its `update()`) picks up the modulated value. The bag is not the tweakpane binding target anymore (bindParam owns its own proxy), so the slider stays anchored to the user-set base. Correct behavior.

The mirror's source-gating: it currently mirrors for both `set` and `notify` (no source check). This is what we want — the instance should see effective values.

- [ ] **Step 5: Smoke test**

Run: `npm test && npm run build && npm run dev`

In the browser:
- Each component folder (OrbitalCloud, ParticleView, BoxView) shows its enable checkbox, reset button, and per-param sliders as before.
- Each continuous param has a `↳ mod` sub-folder. (Discrete params, e.g. OrbitalCloud's mode if it has one, skip the sub-folder.)
- Bind a visual param (e.g. OrbitalCloud's size) to `rms.high` with depth 1.0. Play audio with high-frequency content. The visual param should respond.
- Per-component "Reset to defaults" still resets the underlying params (sliders snap to their default values).

- [ ] **Step 6: Commit**

```bash
git add src/render/components/ComponentManager.ts src/App.ts
git commit -m "feat(components): migrate ComponentManager.bindUI to bindParam"
```

---

## Task 10: Final integration verification

- [ ] **Step 1: Full test suite**

Run: `npm test`
Expected: all green, including new ParamStore + Modulator tests.

- [ ] **Step 2: Type-check via build**

Run: `npm run build`
Expected: clean.

- [ ] **Step 3: End-to-end manual verification in browser**

Run: `npm run dev`. Press `T` to start the test oscillator source.

Verify each:
- [ ] Each continuous, non-`dsp.*` param has a collapsed `↳ mod` sub-folder.
- [ ] `dsp.*` params have NO mod sub-folder.
- [ ] Boolean toggles (`*.enabled`) have NO mod sub-folder.
- [ ] Discrete params (Tonemap mode, camera preset, any component discrete params) have NO mod sub-folder.
- [ ] Setting source=`rms.low`, depth=1, base=mid-range, on FOV — FOV swings with the rms.low signal; slider stays at the user-set value.
- [ ] Setting depth=0 with any source — param sits exactly at base.
- [ ] Setting source back to `none` — param snaps back to base immediately (one notify fired on unbind).
- [ ] `Reset modulation` button clears all bindings; every modulated param snaps back to base.
- [ ] Reload the page — modulation bindings persist (depth + source restored).
- [ ] `Reset to defaults` resets base values; modulation bindings unaffected.
- [ ] HMR (edit a renderer file with dev server running): Modulator survives, bindings still active after rebuild.

- [ ] **Step 4: Commit any tidy-ups**

If the smoke test reveals tweaks (e.g. label wording, folder-default-expanded), apply and commit. If everything checks out, no extra commit needed.

- [ ] **Step 5: Verification-before-completion**

Per the project's verification practice, before declaring done: re-run `npm test`, `npm run build`, and confirm the dev server starts cleanly with the manual checks above passing. Only then is the work complete.

---

## Summary of public API additions

For reference / future readers:

```ts
// src/params/ParamStore.ts
class ParamStore {
  // existing...
  notify(key: string, value: ParamValue): void;
  schemaFor(key: string): ParamSchema | undefined;
}
type Subscriber = (key: string, value: ParamValue, source: "user" | "modulator") => void;

// src/params/Modulator.ts
const MOD_SOURCES: Record<string, { buffer: string; read: (b: Float32Array) => number }>;
const MOD_SOURCE_KEYS: string[];
interface ModBinding { source: string; depth: number; }
class Modulator {
  constructor(store: ParamStore, features: FeatureStore);
  setBinding(key: string, binding: ModBinding | null): void;
  getBinding(key: string): ModBinding | null;
  tick(): void;
  subscribe(fn: (key: string) => void): () => void;
  dispose(): void;
}

// src/params/bindParam.ts
function bindParam(
  folder: FolderApi,
  store: ParamStore,
  modulator: Modulator,
  schema: ParamSchema,
): void;
```

LocalStorage keys used:
- `autocorrelation.params.v1` (existing, base values)
- `autocorrelation.modulation.v1` (new, modulation bindings)
