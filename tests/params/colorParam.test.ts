import { beforeEach, describe, expect, it } from "vitest";
import { Scene } from "three";
import { Pane } from "tweakpane";
import { ParamStore, COLOR_MAX } from "../../src/params/ParamStore";
import { FeatureStore } from "../../src/store/FeatureStore";
import { Modulator } from "../../src/params/Modulator";
import { bindParam } from "../../src/params/bindParam";
import { ComponentManager } from "../../src/render/components/ComponentManager";
import type { Component, ComponentClass, ComponentDeps } from "../../src/render/components/Component";

class FakeColored implements Component {
  static id = "fakeC";
  static label = "Fake C";
  static paramPrefix = "fakeC";
  static paramOpts = { gain: { min: 0, max: 1, step: 0.01 } };
  static paramDefaults = { gain: 0.5, tint: 0xf9b389 };
  static paramKinds = { tint: "color" as const };
  constructor(
    public deps: ComponentDeps,
    public params: Record<string, number>,
  ) {}
  update(): void {}
  dispose(): void {}
}

function makeDeps(paramStore: ParamStore): ComponentDeps {
  return {
    scene: new Scene(),
    store: new FeatureStore(),
    paramStore,
    audioContext: {} as unknown as AudioContext,
    renderer: {} as unknown as import("three/webgpu").WebGPURenderer,
    camera: {} as unknown as import("three").PerspectiveCamera,
  };
}

describe("color params", () => {
  beforeEach(() => localStorage.clear());

  it("accepts the full 24-bit range and rejects anything outside it", () => {
    const store = new ParamStore();
    store.register({ key: "x.tint", label: "tint", kind: "color", default: 0x102030, reconfig: false });

    store.set("x.tint", 0x000000);
    expect(store.get("x.tint")).toBe(0x000000);
    store.set("x.tint", COLOR_MAX);
    expect(store.get("x.tint")).toBe(COLOR_MAX);

    store.set("x.tint", COLOR_MAX + 1);
    expect(store.get("x.tint")).toBe(COLOR_MAX); // rejected, previous kept
    store.set("x.tint", -1);
    expect(store.get("x.tint")).toBe(COLOR_MAX);
  });

  it("registers a color schema from paramKinds and seeds the bag", () => {
    const paramStore = new ParamStore();
    const mgr = new ComponentManager(makeDeps(paramStore), [FakeColored as unknown as ComponentClass]);
    mgr.start();

    expect(paramStore.schemaFor("fakeC.tint")?.kind).toBe("color");
    expect(paramStore.get("fakeC.tint")).toBe(0xf9b389);
    // Other params in the same component are unaffected.
    expect(paramStore.schemaFor("fakeC.gain")?.kind).toBe("continuous");
    mgr.dispose();
  });

  it("renders a color widget and no modulation button", () => {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const pane = new Pane({ container });
    const store = new ParamStore();
    store.register({ key: "x.tint", label: "tint", kind: "color", default: 0x4db3f2, reconfig: false });
    const mod = new Modulator(store, new FeatureStore());

    bindParam(pane as unknown as import("tweakpane").FolderApi, store, mod, store.schemaFor("x.tint")!);

    // Tweakpane's color view renders a swatch button plus a text input holding
    // the packed value — this is what the `view: "color"` cast buys us.
    expect(container.querySelector("input")?.value).toBe("0x4db3f2");
    expect(container.querySelector(".mod-btn")).toBeNull();

    pane.dispose();
    container.remove();
  });
});
