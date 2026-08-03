import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Pane } from "tweakpane";
import { ParamStore, type ParamSchema } from "../../src/params/ParamStore";
import { FeatureStore } from "../../src/store/FeatureStore";
import { Modulator } from "../../src/params/Modulator";
import { PresetStore, type PresetScope } from "../../src/params/PresetStore";
import { addSystemPresets } from "../../src/params/SystemPresets";

const SYSTEM: PresetScope = { id: "system", prefixes: [] };

function num(key: string, def: number): ParamSchema {
  return { key, label: key, kind: "continuous", min: 0, max: 10, step: 0.01, default: def, reconfig: false };
}

function setup() {
  localStorage.clear();
  const store = new ParamStore();
  store.register(num("spawner.rate", 1));
  store.register(num("camera.fov", 2));
  store.register(num("post.bloom.strength", 3));
  store.register({
    key: "components.boxView.enabled",
    label: "Box View enabled",
    kind: "boolean",
    default: true,
    reconfig: false,
  });
  const mod = new Modulator(store, new FeatureStore());
  return { store, mod, presets: new PresetStore(store, mod) };
}

describe("system presets", () => {
  let container: HTMLElement;
  let pane: Pane;

  beforeEach(() => {
    localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    pane = new Pane({ container });
  });

  afterEach(() => {
    pane.dispose();
    container.remove();
  });

  it("captures every scope at once, enable flags included", () => {
    const { store, mod, presets } = setup();
    store.set("spawner.rate", 5);
    store.set("camera.fov", 6);
    store.set("components.boxView.enabled", false);
    mod.setBinding("post.bloom.strength", { source: "rms.low" });
    presets.save(SYSTEM, "full");

    const p = presets.list(SYSTEM)[0];
    expect(Object.keys(p.params).sort()).toEqual([
      "camera.fov",
      "components.boxView.enabled",
      "post.bloom.strength",
      "spawner.rate",
    ]);
    expect(Object.keys(p.mods)).toEqual(["post.bloom.strength"]);
  });

  it("restores every scope and clears modulations added since the save", () => {
    const { store, mod, presets } = setup();
    presets.save(SYSTEM, "base");
    store.set("spawner.rate", 9);
    store.set("components.boxView.enabled", false);
    mod.setBinding("camera.fov", { source: "rms.high" });

    presets.apply(SYSTEM, "base");
    expect(store.get("spawner.rate")).toBe(1);
    expect(store.get("components.boxView.enabled")).toBe(true);
    expect(mod.getBinding("camera.fov")).toBeNull();
  });

  it("goes dirty on a change anywhere in the system", () => {
    const { store, presets } = setup();
    presets.save(SYSTEM, "base");
    expect(presets.isDirty(SYSTEM)).toBe(false);
    store.set("post.bloom.strength", 1.5);
    expect(presets.isDirty(SYSTEM)).toBe(true);
  });

  it("stores the captured thumbnail and keeps it when overwriting without one", () => {
    const { presets } = setup();
    presets.save(SYSTEM, "a", "data:image/jpeg;base64,AAAA");
    expect(presets.list(SYSTEM)[0].thumb).toBe("data:image/jpeg;base64,AAAA");
    presets.save(SYSTEM, "a");
    expect(presets.list(SYSTEM)[0].thumb).toBe("data:image/jpeg;base64,AAAA");
  });

  it("renders one square per preset, marks the current, and loads on click", async () => {
    const { store, presets } = setup();
    const folder = pane.addFolder({ title: "System" });
    addSystemPresets(folder, presets, () => Promise.resolve("data:image/jpeg;base64,AAAA"));

    presets.save(SYSTEM, "one");
    store.set("spawner.rate", 8);
    presets.save(SYSTEM, "two");
    await Promise.resolve();

    const squares = [...container.querySelectorAll<HTMLElement>(".sys-thumb")];
    expect(squares.map((s) => s.title)).toEqual(["one", "two"]);
    expect(squares[1].classList.contains("active")).toBe(true);

    squares[0].click();
    expect(store.get("spawner.rate")).toBe(1);
    expect(presets.current(SYSTEM)).toBe("one");
  });

  it("deletes from the square's × without loading the preset", async () => {
    const { store, presets } = setup();
    const folder = pane.addFolder({ title: "System" });
    addSystemPresets(folder, presets, () => Promise.resolve(""));
    presets.save(SYSTEM, "one");
    store.set("spawner.rate", 7);
    await Promise.resolve();

    container.querySelector<HTMLElement>(".sys-thumb .sys-del")!.click();
    expect(presets.list(SYSTEM)).toHaveLength(0);
    // Click did not fall through to the square's load handler.
    expect(store.get("spawner.rate")).toBe(7);
  });

  it("captures a thumbnail when saving through the button", async () => {
    const { presets } = setup();
    const folder = pane.addFolder({ title: "System" });
    addSystemPresets(folder, presets, () => Promise.resolve("data:image/jpeg;base64,ZZZ"));

    const save = [...container.querySelectorAll("button")].find(
      (b) => b.textContent?.trim() === "+ new system preset",
    );
    save!.click();
    const input = document.querySelector<HTMLInputElement>(".preset-prompt input")!;
    input.value = "snap";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    await Promise.resolve();
    await Promise.resolve();

    expect(presets.list(SYSTEM)[0]?.thumb).toBe("data:image/jpeg;base64,ZZZ");
  });
});
