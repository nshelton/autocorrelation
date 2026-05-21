import { beforeEach, describe, expect, it } from "vitest";
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
    // Build a fresh ParamStore so it reads the new persisted blob.
    deps.paramStore = new ParamStore();
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
    deps.paramStore = new ParamStore();
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

  it("registers discrete schemas for keys declared in paramKinds", () => {
    class FakeDiscrete {
      static id = "fakeDisc";
      static label = "Fake Discrete";
      static paramPrefix = "fakeDisc";
      static paramOpts = { count: { min: 0, max: 0, step: 0 } }; // ignored for discrete
      static paramDefaults = { count: 1000 };
      static paramKinds = { count: "discrete" as const };
      static paramDiscreteOptions = { count: [500, 1000, 2000, 5000] };
      public params: Record<string, number>;
      constructor(_deps: ComponentDeps, params: Record<string, number>) {
        this.params = params;
      }
      update(): void {}
      dispose(): void {}
    }

    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeDiscrete as unknown as ComponentClass]);
    mgr.start();
    const schema = deps.paramStore.schemasInOrder().find((s) => s.key === "fakeDisc.count");
    expect(schema).toBeDefined();
    expect(schema!.kind).toBe("discrete");
    if (schema!.kind === "discrete") {
      expect(schema!.options).toEqual([500, 1000, 2000, 5000]);
    }
    expect(deps.paramStore.get("fakeDisc.count")).toBe(1000);
  });

  it("rejects a discrete value not in the allowed options", () => {
    class FakeDiscrete {
      static id = "fakeDisc2";
      static label = "Fake Discrete 2";
      static paramPrefix = "fakeDisc2";
      static paramOpts = { count: { min: 0, max: 0, step: 0 } };
      static paramDefaults = { count: 1000 };
      static paramKinds = { count: "discrete" as const };
      static paramDiscreteOptions = { count: [500, 1000, 2000] };
      constructor(_deps: ComponentDeps, _params: Record<string, number>) {}
      update(): void {}
      dispose(): void {}
    }

    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeDiscrete as unknown as ComponentClass]);
    mgr.start();
    deps.paramStore.set("fakeDisc2.count", 1234);  // not in [500, 1000, 2000]
    expect(deps.paramStore.get("fakeDisc2.count")).toBe(1000);
  });
});
