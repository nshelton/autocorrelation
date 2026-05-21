# Toggleable Visualizer Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace App.ts's hardcoded view instantiation with a uniform component system. Each visual subsystem (BoxView, DebugView, future additions) registers as a class in one array, gets its own tweakpane folder with an enable checkbox, and is constructed/disposed in response to that checkbox via persisted state in ParamStore.

**Architecture:** A `Component` is a class with `update()` + `dispose()` and required static `id` / `label`. Optional static `paramPrefix` / `paramOpts` / `paramDefaults` opt the component into ParamStore-backed live-tunable params. A `ComponentManager` owns per-component state (stable params bag, instance, tweakpane subscriptions), subscribes to `components.<id>.enabled` keys in ParamStore, and constructs/disposes instances on transitions. App.ts holds one ComponentManager and three call sites (`start`/`update`/`dispose`).

**Tech Stack:** TypeScript 5 (strict), Three.js (webgpu subpath), tweakpane, vitest + happy-dom for tests, RAPIER (rapier3d-compat) inside BoxView only.

**Source spec:** `docs/superpowers/specs/2026-05-20-toggleable-components-design.md`

**Conventions to honor:**
- Comments explain *why*, not *what*. Match the existing tone in `App.ts` and `crates/dsp/src/lib.rs`.
- TypeScript strict — no unused locals, no unused params, no fallthrough.
- One canonical string per buffer/key; don't reintroduce drift.
- Frequent commits — one per task minimum.

---

## File Map

**New files:**
- `src/render/components/Component.ts` — `Component` and `ComponentClass` interfaces, `ComponentDeps` type
- `src/render/components/ComponentManager.ts` — owns per-component state, lifecycle, tweakpane wiring
- `src/render/components/index.ts` — exports the `COMPONENTS` array
- `tests/render/ComponentManager.test.ts` — unit tests for the manager with a fake component

**Modified files:**
- `src/params/ParamStore.ts` — extend `ParamValue` to `number | boolean`, add `boolean` schema kind, update `validate()`
- `src/params/ParamPanel.ts` — branch on `boolean` schema kind for checkbox widget
- `src/params/WorkletBridge.ts` — type-guard against non-numeric param values
- `src/render/components/BoxView.ts` — move metadata to static class properties, accept params bag in constructor
- `src/render/debug/DebugView.ts` — add static `id` / `label`
- `src/App.ts` — drop hardcoded view fields and lifecycle calls; delegate to ComponentManager
- `tests/params/ParamStore.test.ts` — boolean kind coverage
- `tests/params/WorkletBridge.test.ts` — boolean-skip coverage

---

## Task 1: ParamStore boolean schema kind

**Files:**
- Modify: `src/params/ParamStore.ts`
- Test: `tests/params/ParamStore.test.ts`

- [ ] **Step 1: Add failing tests for boolean kind**

Append to `tests/params/ParamStore.test.ts` after the existing tests (before the closing `});` of the `describe` block):

```ts
  const booleanSchema: ParamSchema = {
    key: "test.flag",
    label: "Flag",
    kind: "boolean",
    default: true,
    reconfig: false,
  };

  it("boolean kind: register loads default when no persisted entry", () => {
    const store = new ParamStore();
    store.register(booleanSchema);
    expect(store.get("test.flag")).toBe(true);
  });

  it("boolean kind: restores persisted boolean from localStorage", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ "test.flag": false }));
    const store = new ParamStore();
    store.register(booleanSchema);
    expect(store.get("test.flag")).toBe(false);
  });

  it("boolean kind: set accepts a boolean and persists it", () => {
    const store = new ParamStore();
    store.register(booleanSchema);
    store.set("test.flag", false);
    expect(store.get("test.flag")).toBe(false);
    const persisted = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
    expect(persisted["test.flag"]).toBe(false);
  });

  it("boolean kind: set rejects non-boolean values", () => {
    const store = new ParamStore();
    store.register(booleanSchema);
    const fn = vi.fn();
    store.subscribe(fn);
    store.set("test.flag", 1 as unknown as boolean);
    expect(store.get("test.flag")).toBe(true);
    expect(fn).not.toHaveBeenCalled();
  });

  it("continuous kind: set rejects a boolean value", () => {
    const store = new ParamStore();
    store.register(continuousSchema);
    store.set("test.alpha", true as unknown as number);
    expect(store.get("test.alpha")).toBe(0.2);
  });
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run tests/params/ParamStore.test.ts
```

Expected: 5 new tests fail. Other reasons may include TypeScript errors about `kind: "boolean"` not being assignable — that's also a valid failure mode.

- [ ] **Step 3: Update ParamStore types and validate**

Replace the contents of `src/params/ParamStore.ts` with:

```ts
export type ParamValue = number | boolean;

export type ParamSchema = {
  key: string;
  label: string;
  default: ParamValue;
  reconfig: boolean;
} & (
  | { kind: "discrete"; options: number[] }
  | { kind: "continuous"; min: number; max: number; step: number }
  | { kind: "boolean" }
);

type Subscriber = (key: string, value: ParamValue) => void;

const STORAGE_KEY = "autocorrelation.params.v1";

export class ParamStore {
  private values = new Map<string, ParamValue>();
  private schemas = new Map<string, ParamSchema>();
  private subscribers = new Set<Subscriber>();
  private persisted: Record<string, ParamValue>;

  constructor() {
    this.persisted = this.readPersisted();
  }

  register(schema: ParamSchema): void {
    this.schemas.set(schema.key, schema);
    // Idempotent: keep in-memory value across HMR re-registers. Only seed
    // from persisted/default the first time a key shows up.
    if (this.values.has(schema.key)) return;
    const initial = this.persisted[schema.key];
    // Drop a persisted value whose type no longer matches the schema (e.g.
    // a key was repurposed). Fall back to default rather than throwing.
    const usable =
      initial !== undefined && this.matchesKind(schema, initial)
        ? initial
        : schema.default;
    this.values.set(schema.key, usable);
  }

  get(key: string): ParamValue {
    if (!this.values.has(key)) throw new Error(`ParamStore: unknown key ${key}`);
    return this.values.get(key)!;
  }

  set(key: string, value: ParamValue): void {
    const schema = this.schemas.get(key);
    if (!schema) throw new Error(`ParamStore: unknown key ${key}`);
    if (!this.validate(schema, value)) {
      console.warn(`ParamStore: rejected ${key}=${value} (out of range or wrong type)`);
      return;
    }
    this.values.set(key, value);
    this.writePersisted();
    for (const fn of this.subscribers) fn(key, value);
  }

  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  reset(): void {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // localStorage unavailable; values still reset in-memory
    }
    this.persisted = {};
    const changed: Array<[string, ParamValue]> = [];
    for (const [key, schema] of this.schemas) {
      const oldValue = this.values.get(key);
      if (oldValue !== schema.default) {
        this.values.set(key, schema.default);
        changed.push([key, schema.default]);
      }
    }
    for (const [key, value] of changed) {
      for (const fn of this.subscribers) fn(key, value);
    }
  }

  getAll(): Record<string, ParamValue> {
    return Object.fromEntries(this.values);
  }

  schemasInOrder(): ParamSchema[] {
    return Array.from(this.schemas.values());
  }

  private matchesKind(schema: ParamSchema, value: ParamValue): boolean {
    if (schema.kind === "boolean") return typeof value === "boolean";
    return typeof value === "number";
  }

  private validate(schema: ParamSchema, value: ParamValue): boolean {
    if (schema.kind === "boolean") return typeof value === "boolean";
    if (typeof value !== "number") return false;
    if (schema.kind === "discrete") return schema.options.includes(value);
    return value >= schema.min && value <= schema.max;
  }

  private readPersisted(): Record<string, ParamValue> {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return typeof parsed === "object" && parsed !== null ? parsed : {};
    } catch {
      return {};
    }
  }

  private writePersisted(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.values)));
    } catch {
      // localStorage unavailable; no-op
    }
  }
}
```

- [ ] **Step 4: Run all ParamStore tests, verify pass**

```bash
npx vitest run tests/params/ParamStore.test.ts
```

Expected: all tests pass (the existing 6 + the new 5 = 11 total).

- [ ] **Step 5: Commit**

```bash
git add src/params/ParamStore.ts tests/params/ParamStore.test.ts
git commit -m "feat(params): add boolean schema kind to ParamStore"
```

---

## Task 2: WorkletBridge skips non-numeric params

**Files:**
- Modify: `src/params/WorkletBridge.ts`
- Test: `tests/params/WorkletBridge.test.ts`

- [ ] **Step 1: Add invariant test for component toggle keys**

Append to `tests/params/WorkletBridge.test.ts` after the existing tests, before the closing `});` of the `describe` block. This documents the invariant: component toggle keys never produce worklet messages. The current `handleChange` happens to already satisfy this (it returns silently for keys that don't start with `dsp.`), so the test should pass on the existing code — but Task 1's `ParamValue = number | boolean` change forces type-system updates in `bootstrap()` and `resolveHotValue()` anyway. We commit them together with the type guard for clarity.

```ts
  it("ignores component toggle keys (they never produce worklet messages)", () => {
    const store = makeStore();
    store.register({
      key: "components.boxView.enabled",
      label: "Box View enabled",
      kind: "boolean",
      default: true,
      reconfig: false,
    });
    const port = makePort();
    new WorkletBridge(store, port);
    (port.postMessage as ReturnType<typeof vi.fn>).mockClear();
    store.set("components.boxView.enabled", false);
    expect(port.postMessage).not.toHaveBeenCalled();
  });
```

- [ ] **Step 2: Run test, confirm it already passes**

```bash
npx vitest run tests/params/WorkletBridge.test.ts -t "ignores component toggle"
```

Expected: PASS. The current bridge code happens to already satisfy this invariant. The test now locks it in so future edits can't regress it.

- [ ] **Step 3: Update WorkletBridge for the widened ParamValue type and add defensive type guard**

In `src/params/WorkletBridge.ts`, modify `handleChange()`:

```ts
  private handleChange(key: string): void {
    const value = this.store.get(key);
    if (typeof value !== "number") return;
    if (key === "dsp.windowSize" || key === "dsp.rmsHistoryLen") {
      this.port.postMessage({
        type: "configure",
        windowSize: this.store.get("dsp.windowSize") as number,
        rmsHistoryLen: this.store.get("dsp.rmsHistoryLen") as number,
      });
      return;
    }
    if (key.startsWith("dsp.")) {
      const suffix = key.slice("dsp.".length);
      if (!(HOT_KEYS as readonly string[]).includes(suffix)) return;
      const hotKey = suffix as (typeof HOT_KEYS)[number];
      this.port.postMessage({
        type: "param",
        key: hotKey,
        value: this.resolveHotValue(hotKey),
      });
    }
  }

  private resolveHotValue(key: (typeof HOT_KEYS)[number]): number {
    const value = this.store.get(`dsp.${key}`) as number;
    if (key === "hopSize") {
      return Math.min(value, this.store.get("dsp.windowSize") as number);
    }
    return value;
  }
```

Also fix `bootstrap()` for the new `ParamValue` type:

```ts
  bootstrap(): void {
    this.port.postMessage({
      type: "configure",
      windowSize: this.store.get("dsp.windowSize") as number,
      rmsHistoryLen: this.store.get("dsp.rmsHistoryLen") as number,
    });
    for (const k of HOT_KEYS) {
      this.port.postMessage({
        type: "param",
        key: k,
        value: this.resolveHotValue(k),
      });
    }
  }
```

- [ ] **Step 4: Run all WorkletBridge tests, verify pass**

```bash
npx vitest run tests/params/WorkletBridge.test.ts
```

Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/params/WorkletBridge.ts
git commit -m "feat(params): guard WorkletBridge against non-numeric param values"
```

---

## Task 3: ParamPanel checkbox widget for boolean kind

**Files:**
- Modify: `src/params/ParamPanel.ts`

No unit test — tweakpane is exercised in the browser. The change is a small branch in `addWidget()`.

- [ ] **Step 1: Update ParamPanel.addWidget to handle boolean**

Replace `src/params/ParamPanel.ts` with:

```ts
import { Pane } from "tweakpane";
import { ParamStore, type ParamSchema, type ParamValue } from "./ParamStore";

export class ParamPanel {
  public pane: Pane;
  private bindings: Record<string, ParamValue> = {};
  private unsubscribe: () => void;

  constructor(store: ParamStore, container?: HTMLElement) {
    this.pane = new Pane({ title: "Analysis", container });
    const folder = this.pane.addFolder({ title: "DSP" });

    for (const schema of store.schemasInOrder()) {
      this.bindings[schema.key] = store.get(schema.key);
      const widget = this.addWidget(folder, schema);
      widget.on("change", (e: { value: ParamValue }) => store.set(schema.key, e.value));
    }

    this.unsubscribe = store.subscribe((key, value) => {
      if (this.bindings[key] !== value) {
        this.bindings[key] = value;
        this.pane.refresh();
      }
    });

    this.pane.addButton({ title: "Reset to defaults" }).on("click", () => store.reset());
  }

  dispose(): void {
    this.unsubscribe();
    this.pane.dispose();
  }

  private addWidget(folder: ReturnType<Pane["addFolder"]>, schema: ParamSchema) {
    if (schema.kind === "boolean") {
      return folder.addBinding(this.bindings, schema.key, { label: schema.label });
    }
    if (schema.kind === "discrete") {
      return folder.addBinding(this.bindings, schema.key, {
        label: schema.label,
        options: Object.fromEntries(schema.options.map((v) => [String(v), v])),
      });
    }
    return folder.addBinding(this.bindings, schema.key, {
      label: schema.label,
      min: schema.min,
      max: schema.max,
      step: schema.step,
    });
  }
}
```

- [ ] **Step 2: Type-check**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/params/ParamPanel.ts
git commit -m "feat(params): render checkbox widget for boolean schema kind"
```

---

## Task 4: Component contract and ComponentDeps

**Files:**
- Create: `src/render/components/Component.ts`

- [ ] **Step 1: Create the contract file**

Write `src/render/components/Component.ts`:

```ts
import type { Scene } from "three";
import type { FeatureStore } from "../../store/FeatureStore";
import type { ParamStore } from "../../params/ParamStore";

// Shared dependencies passed to every component constructor. App builds this
// once at start and reuses it for every component instance.
export interface ComponentDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
}

// A component is a class with update() + dispose(). update() takes no args;
// if a future component needs dt, App's RAF loop already tracks it and we add
// the argument then.
export interface Component {
  update(): void;
  dispose(): void;
}

// A component CLASS is the registry entry. Static metadata replaces a separate
// entry-wrapper struct. id/label are required; the param trio is optional and
// opts the component into ParamStore-backed live-tunable params (App owns a
// stable params bag and passes it as the second constructor arg).
export interface ComponentClass {
  new (deps: ComponentDeps, params?: Record<string, number>): Component;
  id: string;
  label: string;
  paramPrefix?: string;
  paramOpts?: Record<string, { min: number; max: number; step?: number }>;
  paramDefaults?: Record<string, number>;
}
```

- [ ] **Step 2: Type-check**

```bash
npx tsc --noEmit
```

Expected: no errors. (The file isn't imported anywhere yet, but it must still parse and resolve its imports.)

- [ ] **Step 3: Commit**

```bash
git add src/render/components/Component.ts
git commit -m "feat(components): add Component and ComponentClass contracts"
```

---

## Task 5: ComponentManager (TDD)

**Files:**
- Create: `src/render/components/ComponentManager.ts`
- Test: `tests/render/ComponentManager.test.ts`

The manager owns all per-component state (instance + stable params bag), registers ParamStore schemas, subscribes to `components.<id>.enabled` keys, and constructs/disposes instances on transitions. App.ts will call `start(deps)`, `bindUI(pane)`, `update()`, and `dispose()`. Tweakpane wiring lives here too — see the per-component-folder layout in the spec.

- [ ] **Step 1: Create the test directory**

```bash
mkdir -p /Users/nshelton/autocorrelation/tests/render
```

- [ ] **Step 2: Write failing tests with a fake component**

Write `tests/render/ComponentManager.test.ts`:

```ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Scene } from "three";
import { ParamStore } from "../../src/params/ParamStore";
import { FeatureStore } from "../../src/store/FeatureStore";
import { ComponentManager } from "../../src/render/components/ComponentManager";
import type {
  Component,
  ComponentClass,
  ComponentDeps,
} from "../../src/render/components/Component";

// Fake component records its lifecycle calls and (optionally) reads from its
// injected params bag. No three.js / RAPIER baggage.
class FakeNoParams implements Component {
  static id = "fakeA";
  static label = "Fake A";
  static instances: FakeNoParams[] = [];
  public updateCount = 0;
  public disposed = false;
  constructor(public deps: ComponentDeps) {
    FakeNoParams.instances.push(this);
  }
  update(): void {
    this.updateCount += 1;
  }
  dispose(): void {
    this.disposed = true;
  }
}

class FakeWithParams implements Component {
  static id = "fakeB";
  static label = "Fake B";
  static paramPrefix = "fakeB";
  static paramOpts = { gain: { min: 0, max: 1, step: 0.01 } };
  static paramDefaults = { gain: 0.5 };
  static instances: FakeWithParams[] = [];
  constructor(
    public deps: ComponentDeps,
    public params: Record<string, number>,
  ) {
    FakeWithParams.instances.push(this);
  }
  update(): void {}
  dispose(): void {}
}

function makeDeps(): ComponentDeps {
  // happy-dom doesn't ship a WebAudio API; cast through `unknown` since the
  // manager passes it through to constructors without using it.
  const audioContext = {} as unknown as AudioContext;
  return {
    scene: new Scene(),
    store: new FeatureStore(),
    paramStore: new ParamStore(),
    audioContext,
  };
}

describe("ComponentManager", () => {
  beforeEach(() => {
    localStorage.clear();
    FakeNoParams.instances = [];
    FakeWithParams.instances = [];
  });

  it("constructs enabled components on start (default enabled = true)", () => {
    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeNoParams as ComponentClass]);
    mgr.start();
    expect(FakeNoParams.instances).toHaveLength(1);
  });

  it("does not construct disabled components on start", () => {
    const deps = makeDeps();
    // Pre-persist the enabled flag as false.
    localStorage.setItem(
      "autocorrelation.params.v1",
      JSON.stringify({ "components.fakeA.enabled": false }),
    );
    const mgr = new ComponentManager(deps, [FakeNoParams as ComponentClass]);
    mgr.start();
    expect(FakeNoParams.instances).toHaveLength(0);
  });

  it("disposes the live instance on enabled true -> false", () => {
    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeNoParams as ComponentClass]);
    mgr.start();
    const inst = FakeNoParams.instances[0];
    deps.paramStore.set("components.fakeA.enabled", false);
    expect(inst.disposed).toBe(true);
  });

  it("constructs a fresh instance on enabled false -> true", () => {
    const deps = makeDeps();
    localStorage.setItem(
      "autocorrelation.params.v1",
      JSON.stringify({ "components.fakeA.enabled": false }),
    );
    const mgr = new ComponentManager(deps, [FakeNoParams as ComponentClass]);
    mgr.start();
    expect(FakeNoParams.instances).toHaveLength(0);
    deps.paramStore.set("components.fakeA.enabled", true);
    expect(FakeNoParams.instances).toHaveLength(1);
  });

  it("update() calls update on every live instance", () => {
    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeNoParams as ComponentClass]);
    mgr.start();
    mgr.update();
    mgr.update();
    expect(FakeNoParams.instances[0].updateCount).toBe(2);
  });

  it("update() skips disposed components", () => {
    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeNoParams as ComponentClass]);
    mgr.start();
    const inst = FakeNoParams.instances[0];
    deps.paramStore.set("components.fakeA.enabled", false);
    mgr.update();
    expect(inst.updateCount).toBe(0);
  });

  it("dispose() tears down live instances and stops responding to store changes", () => {
    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeNoParams as ComponentClass]);
    mgr.start();
    const inst = FakeNoParams.instances[0];
    mgr.dispose();
    expect(inst.disposed).toBe(true);
    // Subsequent store mutations must not construct a new instance.
    deps.paramStore.set("components.fakeA.enabled", false);
    deps.paramStore.set("components.fakeA.enabled", true);
    expect(FakeNoParams.instances).toHaveLength(1);
  });

  it("registers param schemas for a component with paramDefaults and seeds the bag", () => {
    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeWithParams as ComponentClass]);
    mgr.start();
    expect(deps.paramStore.get("fakeB.gain")).toBe(0.5);
    expect(FakeWithParams.instances[0].params).toEqual({ gain: 0.5 });
  });

  it("seeds the bag from persisted ParamStore values (not the static default)", () => {
    const deps = makeDeps();
    localStorage.setItem(
      "autocorrelation.params.v1",
      JSON.stringify({ "fakeB.gain": 0.9 }),
    );
    // Build a fresh paramStore so it picks up the new persisted blob.
    deps.paramStore = new ParamStore();
    const mgr = new ComponentManager(deps, [FakeWithParams as ComponentClass]);
    mgr.start();
    expect(FakeWithParams.instances[0].params).toEqual({ gain: 0.9 });
  });

  it("the params bag is shared across toggle cycles (same reference)", () => {
    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeWithParams as ComponentClass]);
    mgr.start();
    const firstBag = FakeWithParams.instances[0].params;
    deps.paramStore.set("components.fakeB.enabled", false);
    deps.paramStore.set("components.fakeB.enabled", true);
    const secondBag = FakeWithParams.instances[1].params;
    expect(secondBag).toBe(firstBag);
  });

  it("external paramStore writes update the bag (so live instance picks them up)", () => {
    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeWithParams as ComponentClass]);
    mgr.start();
    deps.paramStore.set("fakeB.gain", 0.25);
    expect(FakeWithParams.instances[0].params.gain).toBe(0.25);
  });
});
```

- [ ] **Step 3: Run tests, verify failure**

```bash
npx vitest run tests/render/ComponentManager.test.ts
```

Expected: all tests fail (module not found).

- [ ] **Step 4: Implement ComponentManager**

Write `src/render/components/ComponentManager.ts`:

```ts
import type { Pane } from "tweakpane";
import type { Component, ComponentClass, ComponentDeps } from "./Component";

// Per-component runtime state. paramsBag is null for components that don't
// declare static paramDefaults (DebugView today). The bag is allocated once
// per page lifetime and kept across toggle cycles so tweakpane bindings
// (which hold a *reference*) stay live.
interface Slot {
  cls: ComponentClass;
  paramsBag: Record<string, number> | null;
  instance: Component | null;
  enabledKey: string;
}

export class ComponentManager {
  private slots: Slot[] = [];
  private storeUnsub: (() => void) | null = null;
  private paneFolders: Array<{ dispose: () => void }> = [];

  constructor(
    private deps: ComponentDeps,
    private classes: readonly ComponentClass[],
  ) {}

  // Register schemas, allocate bags, and construct components whose enabled
  // flag is true. Must be called once before update() or bindUI().
  start(): void {
    const { paramStore } = this.deps;

    for (const cls of this.classes) {
      const enabledKey = `components.${cls.id}.enabled`;
      paramStore.register({
        key: enabledKey,
        label: `${cls.label} enabled`,
        kind: "boolean",
        default: true,
        reconfig: false,
      });

      const paramsBag = this.allocateBag(cls);
      const slot: Slot = { cls, paramsBag, instance: null, enabledKey };
      this.slots.push(slot);

      if (paramStore.get(enabledKey) === true) {
        this.construct(slot);
      }
    }

    this.storeUnsub = paramStore.subscribe((key, value) => {
      const slot = this.slots.find((s) => s.enabledKey === key);
      if (slot) {
        if (value === true && !slot.instance) this.construct(slot);
        else if (value === false && slot.instance) this.destroy(slot);
        return;
      }
      // Mirror external param changes into the stable bag so the live
      // instance picks them up next frame (it reads from the same object).
      for (const s of this.slots) {
        if (!s.paramsBag) continue;
        const prefix = `${s.cls.paramPrefix ?? s.cls.id}.`;
        if (!key.startsWith(prefix)) continue;
        const localKey = key.slice(prefix.length);
        if (localKey in s.paramsBag && typeof value === "number") {
          s.paramsBag[localKey] = value;
        }
      }
    });
  }

  // Add one tweakpane folder per component: enable checkbox first, then
  // (if applicable) one slider per param bound to the stable bag.
  bindUI(pane: Pane): void {
    const { paramStore } = this.deps;

    for (const slot of this.slots) {
      const folder = pane.addFolder({ title: slot.cls.label });
      this.paneFolders.push(folder);

      const enabledProxy: { enabled: boolean } = {
        enabled: paramStore.get(slot.enabledKey) === true,
      };
      const enabledBinding = folder.addBinding(enabledProxy, "enabled", {
        label: "enabled",
      });
      enabledBinding.on("change", (e: { value: boolean }) => {
        paramStore.set(slot.enabledKey, e.value);
      });
      // Mirror external enable changes back into the checkbox.
      const unsub = paramStore.subscribe((key, value) => {
        if (key === slot.enabledKey && typeof value === "boolean") {
          if (enabledProxy.enabled !== value) {
            enabledProxy.enabled = value;
            pane.refresh();
          }
        }
      });
      this.paneFolders.push({ dispose: unsub });

      if (!slot.paramsBag || !slot.cls.paramOpts) continue;
      for (const [k, opts] of Object.entries(slot.cls.paramOpts)) {
        const fullKey = `${slot.cls.paramPrefix ?? slot.cls.id}.${k}`;
        const slider = folder.addBinding(slot.paramsBag, k, {
          ...opts,
          step: opts.step ?? (opts.max - opts.min) / 100,
        });
        slider.on("change", (e: { value: number }) => {
          paramStore.set(fullKey, e.value);
        });
      }
    }
  }

  update(): void {
    for (const slot of this.slots) {
      slot.instance?.update();
    }
  }

  dispose(): void {
    this.storeUnsub?.();
    this.storeUnsub = null;
    for (const slot of this.slots) {
      if (slot.instance) {
        slot.instance.dispose();
        slot.instance = null;
      }
    }
    for (const f of this.paneFolders) {
      try {
        f.dispose();
      } catch {
        // Some pane entries are subscriptions, not real folders; both
        // expose .dispose() but neither should throw.
      }
    }
    this.paneFolders = [];
    this.slots = [];
  }

  // Allocate (or null) the params bag. Must be called BEFORE construct() —
  // the bag is the second constructor arg.
  private allocateBag(cls: ComponentClass): Record<string, number> | null {
    if (!cls.paramDefaults) return null;
    const { paramStore } = this.deps;
    const bag: Record<string, number> = {};
    const prefix = cls.paramPrefix ?? cls.id;
    for (const [k, def] of Object.entries(cls.paramDefaults)) {
      const fullKey = `${prefix}.${k}`;
      const opts = cls.paramOpts?.[k];
      paramStore.register({
        key: fullKey,
        label: k,
        kind: "continuous",
        reconfig: false,
        default: def,
        min: opts?.min ?? 0,
        max: opts?.max ?? 1,
        step: opts?.step ?? 0.01,
      });
      const v = paramStore.get(fullKey);
      bag[k] = typeof v === "number" ? v : def;
    }
    return bag;
  }

  private construct(slot: Slot): void {
    slot.instance = slot.paramsBag
      ? new slot.cls(this.deps, slot.paramsBag)
      : new slot.cls(this.deps);
  }

  private destroy(slot: Slot): void {
    slot.instance?.dispose();
    slot.instance = null;
  }
}
```

- [ ] **Step 5: Run ComponentManager tests, verify pass**

```bash
npx vitest run tests/render/ComponentManager.test.ts
```

Expected: all 11 tests pass.

- [ ] **Step 6: Run full test suite**

```bash
npm test
```

Expected: all tests pass (no regressions in ParamStore / WorkletBridge / FeatureStore).

- [ ] **Step 7: Commit**

```bash
git add src/render/components/ComponentManager.ts tests/render/ComponentManager.test.ts
git commit -m "feat(components): add ComponentManager with lifecycle + bag wiring"
```

---

## Task 6: Refactor BoxView to the new contract

**Files:**
- Modify: `src/render/components/BoxView.ts`

- [ ] **Step 1: Adapt BoxView**

Rewrite `src/render/components/BoxView.ts`. Replace the existing class so that:
- Metadata moves to static class properties (`id`, `label`, `paramPrefix`, `paramOpts`, `paramDefaults`)
- The constructor accepts `(deps: ComponentDeps, params: Record<string, number>)` and stores the bag reference (instead of the old `BoxViewDeps`)
- Old `paramOpts` / `params` / `paramPrefix` instance fields are removed

Full new file content:

```ts
import {
  InstancedMesh,
  InstancedBufferAttribute,
  BoxGeometry,
  Object3D,
  Color,
} from "three";
import { MeshBasicNodeMaterial } from "three/webgpu";
import {
  vec3,
  vec4,
  float,
  dot,
  max,
  normalWorld,
  instancedBufferAttribute,
} from "three/tsl";
import RAPIER from "@dimforge/rapier3d-compat";
import type { Component, ComponentDeps } from "./Component";

const BOX_COUNT = 1024;
const CONTAINER_HALF = 1.5;
const BASE_SIZE = 0.12;

export class BoxView implements Component {
  static id = "boxView";
  static label = "Box View";
  static paramPrefix = "boxView";
  static paramOpts = {
    pull: { min: 0, max: 1, step: 0.01 },
    timestep: { min: 0.005, max: 0.1, step: 0.001 },
    width: { min: 0, max: 2, step: 0.01 },
  };
  static paramDefaults = {
    pull: 0.3,
    timestep: 1 / 30,
    width: 0.5,
  };

  // Reference to App-owned stable bag — read each frame, mutated by tweakpane.
  // Never reassigned; tweakpane bindings depend on the object identity.
  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private store: ComponentDeps["store"];
  private mesh: InstancedMesh | null = null;
  private world: RAPIER.World | null = null;
  private bodies: RAPIER.RigidBody[] = [];
  private colliders: RAPIER.Collider[] = [];
  private dummy = new Object3D();
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.store = deps.store;
    this.params = params;
    void this.init();
  }

  private async init(): Promise<void> {
    await RAPIER.init();
    if (this.disposed) return;

    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    world.timestep = this.params.timestep;

    const c = CONTAINER_HALF;

    // Per-instance HSL colors via our own InstancedBufferAttribute. setColorAt would route
    // through NodeMaterial's vInstanceColor varying, which is broken for our setup in r170.
    const colorArr = new Float32Array(BOX_COUNT * 3);
    const tmpColor = new Color();
    for (let i = 0; i < BOX_COUNT; i++) {
      tmpColor.setHSL(1, 1, 1);
      tmpColor.toArray(colorArr, i * 3);
    }
    const colorAttr = new InstancedBufferAttribute(colorArr, 3);

    // MeshStandardNodeMaterial with custom colorNode silently drops lights in r170 + WebGPU +
    // InstancedMesh. Hand-rolled lambert on MeshBasicNodeMaterial is what works.
    const mat = new MeshBasicNodeMaterial();
    const instColor = vec3(instancedBufferAttribute(colorAttr, "vec3", 3, 0));
    const lightDir = vec3(0.408, 0.866, 0.306);
    const ndotl = max(dot(normalWorld, lightDir), float(0.0));
    const lit = ndotl.mul(0.7).add(0.3);
    mat.colorNode = vec4(instColor.mul(lit), 1.0);

    const geom = new BoxGeometry(BASE_SIZE, BASE_SIZE, BASE_SIZE);
    const mesh = new InstancedMesh(geom, mat, BOX_COUNT);

    const half = BASE_SIZE / 2;
    for (let i = 0; i < BOX_COUNT; i++) {
      const x = (Math.random() - 0.5) * 2 * c * 0.7;
      const y = (Math.random() - 0.5) * 2 * c * 0.7;
      const z = (Math.random() - 0.5) * 2 * c * 0.7;
      const body = world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic()
          .setTranslation(x, y, z)
          .setLinvel(
            (Math.random() - 0.5) * 1.5,
            (Math.random() - 0.5) * 1.5,
            (Math.random() - 0.5) * 1.5,
          )
          .setAngvel({
            x: (Math.random() - 0.5) * 2,
            y: (Math.random() - 0.5) * 2,
            z: (Math.random() - 0.5) * 2,
          })
          .setLinearDamping(0.1)
          .setAngularDamping(0.1),
      );
      const collider = world.createCollider(
        RAPIER.ColliderDesc.cuboid(half, half, half).setRestitution(0.9),
        body,
      );
      this.bodies.push(body);
      this.colliders.push(collider);
    }

    this.world = world;
    this.mesh = mesh;
    this.scene.add(mesh);
  }

  update(): void {
    if (!this.world || !this.mesh) return;
    this.world.timestep = this.params.timestep;
    this.world.step();
    const PULL = this.params.pull;

    const spec = this.store.get("spectrum");
    const specLen = spec.length;

    const baseHalf = BASE_SIZE / 2;
    const halfCount = (BOX_COUNT - 1) / 2;
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i];
      const t = b.translation();
      const r = b.rotation();

      const restX = ((i - halfCount) * this.params.width) / BOX_COUNT;
      const vel = b.linvel();
      b.setLinvel(
        {
          x: vel.x + (restX - t.x) * PULL,
          y: vel.y - t.y * PULL,
          z: vel.z - t.z * PULL,
        },
        true,
      );

      let s = 1.0;
      if (specLen > 0) {
        const bin = Math.min(
          specLen - 1,
          Math.floor((i / BOX_COUNT) * specLen * 0.25),
        );
        s = 0.1 + spec[bin] * 3.0;
      }

      const h = baseHalf * s;
      this.colliders[i].setHalfExtents({ x: h, y: h, z: h });

      this.dummy.position.set(t.x, t.y, t.z);
      this.dummy.quaternion.set(r.x, r.y, r.z, r.w);
      this.dummy.scale.set(s, s, s);
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this.disposed = true;
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as MeshBasicNodeMaterial).dispose();
      this.mesh.dispose();
      this.mesh = null;
    }
    if (this.world) {
      this.world.free();
      this.world = null;
    }
    this.bodies = [];
    this.colliders = [];
  }
}
```

- [ ] **Step 2: Type-check**

```bash
npx tsc --noEmit
```

Expected: errors only in `src/App.ts` (it still references the old `BoxViewDeps`-shaped constructor). Those are fixed in Task 8.

- [ ] **Step 3: Commit**

```bash
git add src/render/components/BoxView.ts
git commit -m "refactor(BoxView): adapt to Component contract with static metadata"
```

---

## Task 7: Add static metadata to DebugView

**Files:**
- Modify: `src/render/debug/DebugView.ts`

DebugView keeps its existing constructor signature `(deps: DebugViewDeps)` — DebugViewDeps already has the same shape as ComponentDeps (scene + store + paramStore + audioContext), so it's structurally compatible. We only add the two static fields for the registry.

- [ ] **Step 1: Add static id/label**

In `src/render/debug/DebugView.ts`, change the `export class DebugView` line and the block above it. Find:

```ts
export class DebugView {
  private lines = new Map<LineStoreKey, TimeSeriesRenderer>();
```

Replace with:

```ts
export class DebugView {
  static id = "debugView";
  static label = "Debug View";

  private lines = new Map<LineStoreKey, TimeSeriesRenderer>();
```

- [ ] **Step 2: Type-check**

```bash
npx tsc --noEmit
```

Expected: errors only in `src/App.ts` (still fine — fixed in Task 8). No new errors in `DebugView.ts`.

- [ ] **Step 3: Commit**

```bash
git add src/render/debug/DebugView.ts
git commit -m "refactor(DebugView): expose static id/label for component registry"
```

---

## Task 8: Create the COMPONENTS registry

**Files:**
- Create: `src/render/components/index.ts`

- [ ] **Step 1: Write the registry**

Write `src/render/components/index.ts`:

```ts
import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import type { ComponentClass } from "./Component";

// Order = render order in the scene (insertion order). Also drives the
// order of folders in the tweakpane panel. Add a new component: import it
// here and append to this array.
export const COMPONENTS: readonly ComponentClass[] = [
  DebugView as unknown as ComponentClass,
  BoxView,
];
```

The `DebugView as unknown as ComponentClass` cast is necessary because DebugView's constructor takes its own `DebugViewDeps` (a structural superset of `ComponentDeps`) rather than `ComponentDeps` literally — TypeScript's invariant function parameters reject the direct assignment even though it's safe.

- [ ] **Step 2: Type-check**

```bash
npx tsc --noEmit
```

Expected: errors only in `src/App.ts`. No errors in `index.ts`.

- [ ] **Step 3: Commit**

```bash
git add src/render/components/index.ts
git commit -m "feat(components): add COMPONENTS registry array"
```

---

## Task 9: Wire App.ts to use ComponentManager

**Files:**
- Modify: `src/App.ts`

This drops the hardcoded `boxView`/`debugView` fields, the `bindViewParams` helper, and the `ViewWithParams` interface. ComponentManager replaces all of it.

- [ ] **Step 1: Rewrite App.ts**

Replace the entire contents of `src/App.ts` with:

```ts
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

    this.rig = new CameraRig(camera);
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
    void this.rig.goTo("front", { duration: 0 });

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
    this.keydownHandler = (e) => {
      const preset = presetKeys[e.key];
      if (preset) {
        this.rig.goTo(preset, { duration: 0.8 });
        return;
      }
      if (e.key === " ") {
        toggled = !toggled;
        this.rig.goTo(toggled ? "side" : "front", { duration: 0.8 });
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

  bindUI(pane: import("tweakpane").Pane): void {
    this.components.bindUI(pane);
  }

  dispose(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    window.removeEventListener("keydown", this.keydownHandler);
    window.removeEventListener("resize", this.resizeHandler);
    this.components?.dispose();
    this.fps.unmount();
    this.deps.workletNode.port.onmessage = null;
  }
}
```

- [ ] **Step 2: Type-check**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Run full test suite**

```bash
npm test
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/App.ts
git commit -m "refactor(App): delegate component lifecycle to ComponentManager"
```

---

## Task 10: End-to-end manual verification

**Files:** none (verification only)

- [ ] **Step 1: Clear persisted state (clean baseline)**

In the browser DevTools console (run the dev server first):
```js
localStorage.removeItem("autocorrelation.params.v1")
```

Or just open an incognito tab.

- [ ] **Step 2: Start the dev server**

```bash
npm run dev
```

Open the URL it prints (default http://localhost:5173). Press any key or click "Mic" to start the audio source (test source with `T` if you don't want to grant mic permission).

- [ ] **Step 3: Verify default behavior**

- The tweakpane panel on the right shows: an "Analysis" title, a "DSP" folder (existing), and below it two new folders: **"Debug View"** and **"Box View"**.
- Each new folder has an "enabled" checkbox at the top, both checked by default.
- The Box View folder also shows three sliders: `pull`, `timestep`, `width`.
- The Debug View renders its lines/markers; the Box View renders 1024 instanced cubes.

- [ ] **Step 4: Toggle Debug View off**

- Uncheck Debug View's "enabled" checkbox.
- The lines and beat markers should disappear from the scene.
- The cubes remain.
- The folder stays visible; the checkbox is unchecked.

- [ ] **Step 5: Toggle Debug View back on**

- Check the box again.
- The lines reappear (with fresh scrolling history — they were disposed, not paused).

- [ ] **Step 6: Toggle Box View off**

- Uncheck Box View's checkbox.
- The cubes disappear.
- The pull/timestep/width sliders remain visible and adjustable (pre-tweak).

- [ ] **Step 7: Pre-tweak then re-enable Box View**

- With Box View disabled, drag `width` to ~1.5.
- Check the enabled box.
- Cubes reappear, arranged along a wider rest line (the new instance read the tweaked width from the bag).

- [ ] **Step 8: Persistence check**

- Uncheck both components.
- Reload the page (`Cmd-R`).
- Restart the audio source.
- Both components should still be disabled after the reload (state is persisted in localStorage).
- Re-enable both before continuing.

- [ ] **Step 9: Reset behavior**

- Click "Reset to defaults" at the bottom of the panel.
- Both `components.*.enabled` keys reset to `true`; if anything was disabled, it gets re-constructed.
- BoxView's pull/timestep/width return to their defaults.

- [ ] **Step 10: HMR sanity**

- With the dev server running, edit `src/render/components/BoxView.ts` (e.g. change `BOX_COUNT` to 512).
- Save.
- Vite's HMR should swap App; components get re-constructed by the new ComponentManager. The cube count visibly changes without a full page reload.
- Revert the edit.

- [ ] **Step 11: Final commit (if any fixes were made during this task)**

Otherwise skip. If you fixed something:

```bash
git add -p  # review hunks
git commit -m "fix(components): <describe>"
```

---

## Done

When all 10 tasks check out:
- `npm test` passes
- `npx tsc --noEmit` clean
- Manual verification all 10 steps pass
- Git log shows ~9 commits (one per implementation task)

The end state matches the spec: App.ts is ~150 lines (was ~225), each visualizer is a self-contained class, adding a new one is a single import + array append, and toggle state persists per-user across reloads.
