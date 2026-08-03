import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Scene } from "three";
import { Pane } from "tweakpane";
import { ParamStore } from "../../src/params/ParamStore";
import { FeatureStore } from "../../src/store/FeatureStore";
import { Modulator } from "../../src/params/Modulator";
import { PresetStore } from "../../src/params/PresetStore";
import {
  ComponentManager,
  type SceneColumnHost,
} from "../../src/render/components/ComponentManager";
import type { Component, ComponentClass, ComponentDeps } from "../../src/render/components/Component";

class FakeA implements Component {
  static id = "fakeA";
  static label = "Fake A";
  static paramPrefix = "fakeA";
  static paramOpts = { gain: { min: 0, max: 1, step: 0.01 } };
  static paramDefaults = { gain: 0.5 };
  constructor(public deps: ComponentDeps, public params: Record<string, number>) {}
  update(): void {}
  dispose(): void {}
}
class FakeB implements Component {
  static id = "fakeB";
  static label = "Fake B";
  static paramPrefix = "fakeB";
  static paramOpts = { size: { min: 0, max: 1, step: 0.01 } };
  static paramDefaults = { size: 0.25 };
  constructor(public deps: ComponentDeps, public params: Record<string, number>) {}
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

describe("per-scene columns", () => {
  let root: HTMLElement;
  let togglePane: Pane;
  // Stand-in for ParamPanel: records live columns and their order.
  let columns: Array<{ el: HTMLElement; pane: Pane; id: string }>;
  let host: SceneColumnHost;

  beforeEach(() => {
    localStorage.clear();
    root = document.createElement("div");
    document.body.appendChild(root);
    togglePane = new Pane({ container: root });
    columns = [];
    host = {
      addScenePanel(id: string) {
        const el = document.createElement("div");
        el.dataset.panelId = id;
        root.appendChild(el);
        const pane = new Pane({ container: el });
        const entry = { el, pane, id };
        columns.push(entry);
        return {
          pane,
          dispose: () => {
            columns.splice(columns.indexOf(entry), 1);
            pane.dispose();
            el.remove();
          },
        };
      },
    };
  });

  afterEach(() => {
    togglePane.dispose();
    root.remove();
  });

  function setup(aOn: boolean, bOn: boolean) {
    const paramStore = new ParamStore();
    const mgr = new ComponentManager(makeDeps(paramStore), [
      FakeA as unknown as ComponentClass,
      FakeB as unknown as ComponentClass,
    ]);
    mgr.start();
    paramStore.set("components.fakeA.enabled", aOn);
    paramStore.set("components.fakeB.enabled", bOn);
    const mod = new Modulator(paramStore, new FeatureStore());
    const presets = new PresetStore(paramStore, mod);
    const toggles = togglePane.addFolder({ title: "Scenes" });
    mgr.bindUI(toggles, host, mod, presets);
    return { paramStore, mgr };
  }

  it("gives each enabled scene its own panel", () => {
    const { mgr } = setup(true, true);
    expect(columns.map((c) => c.id)).toEqual(["fakeA", "fakeB"]);
    mgr.dispose();
  });

  it("builds no panel for a disabled scene", () => {
    const { mgr } = setup(true, false);
    expect(columns.map((c) => c.id)).toEqual(["fakeA"]);
    mgr.dispose();
  });

  it("adds and removes a panel as the toggle flips", () => {
    const { paramStore, mgr } = setup(false, false);
    expect(columns).toHaveLength(0);

    paramStore.set("components.fakeB.enabled", true);
    expect(columns.map((c) => c.id)).toEqual(["fakeB"]);

    paramStore.set("components.fakeA.enabled", true);
    expect(columns.map((c) => c.id).sort()).toEqual(["fakeA", "fakeB"]);

    paramStore.set("components.fakeB.enabled", false);
    expect(columns.map((c) => c.id)).toEqual(["fakeA"]);
    mgr.dispose();
  });

  it("renders that scene's params in its own panel, not the toggle column", () => {
    const { mgr } = setup(true, false);
    const inColumn = columns[0].el.textContent ?? "";
    expect(inColumn).toContain("gain");
    // The toggle column carries labels only.
    const toggleText = root.querySelector(".tp-rotv")?.textContent ?? "";
    expect(toggleText).toContain("Fake A");
    mgr.dispose();
  });

  it("dispose tears every panel back out", () => {
    const { mgr } = setup(true, true);
    expect(columns).toHaveLength(2);
    mgr.dispose();
    expect(columns).toHaveLength(0);
  });
});
