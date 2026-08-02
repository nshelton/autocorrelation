import { describe, it, expect, beforeEach } from "vitest";
import { ParamStore, type ParamSchema } from "../../src/params/ParamStore";
import { FeatureStore } from "../../src/store/FeatureStore";
import { Modulator } from "../../src/params/Modulator";
import { PresetStore, type PresetScope } from "../../src/params/PresetStore";

const SPAWNER: PresetScope = { id: "spawner", prefixes: ["spawner"] };
// Camera-shaped scope: two prefixes under one preset id.
const CAMERA: PresetScope = { id: "camera", prefixes: ["camera", "light"] };

function schema(key: string, def: number): ParamSchema {
  return { key, label: key, kind: "continuous", min: 0, max: 10, step: 0.01, default: def, reconfig: false };
}

function setup() {
  localStorage.clear();
  const store = new ParamStore();
  store.register(schema("spawner.rate", 1));
  store.register(schema("spawner.size", 2));
  store.register(schema("particleView.damping", 0.2));
  store.register(schema("camera.fov", 6));
  store.register(schema("light.intensity", 1));
  const mod = new Modulator(store, new FeatureStore());
  return { store, mod, presets: new PresetStore(store, mod) };
}

describe("PresetStore", () => {
  beforeEach(() => localStorage.clear());

  it("captures params + mods for its scope only", () => {
    const { store, mod, presets } = setup();
    store.set("spawner.rate", 5);
    mod.setBinding("spawner.size", { source: "rms.low", lo: 0, hi: 3 });
    mod.setBinding("particleView.damping", { source: "rms.high" });
    presets.save(SPAWNER, "fast");

    const p = presets.list(SPAWNER)[0];
    expect(p.params).toEqual({ "spawner.rate": 5, "spawner.size": 2 });
    expect(Object.keys(p.mods)).toEqual(["spawner.size"]);
    expect(presets.current(SPAWNER)).toBe("fast");
  });

  it("apply() restores params and clears mods added after the save", () => {
    const { store, mod, presets } = setup();
    presets.save(SPAWNER, "clean");
    store.set("spawner.rate", 9);
    mod.setBinding("spawner.rate", { source: "rms.low" });

    presets.apply(SPAWNER, "clean");
    expect(store.get("spawner.rate")).toBe(1);
    expect(mod.getBinding("spawner.rate")).toBeNull();
  });

  it("apply() leaves other scopes alone", () => {
    const { store, mod, presets } = setup();
    presets.save(SPAWNER, "a");
    store.set("particleView.damping", 0.9);
    mod.setBinding("particleView.damping", { source: "rms.low" });

    presets.apply(SPAWNER, "a");
    expect(store.get("particleView.damping")).toBe(0.9);
    expect(mod.getBinding("particleView.damping")).not.toBeNull();
  });

  it("tracks dirty on param and modulation edits, clean after re-save", () => {
    const { store, mod, presets } = setup();
    presets.save(SPAWNER, "a");
    expect(presets.isDirty(SPAWNER)).toBe(false);

    store.set("spawner.rate", 7);
    expect(presets.isDirty(SPAWNER)).toBe(true);
    presets.save(SPAWNER, "a");
    expect(presets.isDirty(SPAWNER)).toBe(false);

    mod.setBinding("spawner.rate", { source: "rms.low" });
    expect(presets.isDirty(SPAWNER)).toBe(true);
  });

  it("save() overwrites same-name, appends new names", () => {
    const { store, presets } = setup();
    presets.save(SPAWNER, "a");
    store.set("spawner.rate", 4);
    presets.save(SPAWNER, "a");
    expect(presets.list(SPAWNER)).toHaveLength(1);
    expect(presets.list(SPAWNER)[0].params["spawner.rate"]).toBe(4);

    presets.save(SPAWNER, "b");
    expect(presets.list(SPAWNER).map((p) => p.name)).toEqual(["a", "b"]);
    expect(presets.current(SPAWNER)).toBe("b");
  });

  it("remove() drops the preset and clears current when it was selected", () => {
    const { presets } = setup();
    presets.save(SPAWNER, "a");
    presets.remove(SPAWNER, "a");
    expect(presets.list(SPAWNER)).toHaveLength(0);
    expect(presets.current(SPAWNER)).toBeNull();
    expect(presets.isDirty(SPAWNER)).toBe(false);
  });

  it("captures every prefix of a multi-prefix scope", () => {
    const { store, mod, presets } = setup();
    store.set("camera.fov", 3);
    store.set("light.intensity", 4);
    mod.setBinding("light.intensity", { source: "rms.low" });
    presets.save(CAMERA, "wide");

    const p = presets.list(CAMERA)[0];
    expect(Object.keys(p.params).sort()).toEqual(["camera.fov", "light.intensity"]);
    expect(Object.keys(p.mods)).toEqual(["light.intensity"]);

    store.set("light.intensity", 9);
    expect(presets.isDirty(CAMERA)).toBe(true);
    presets.apply(CAMERA, "wide");
    expect(store.get("light.intensity")).toBe(4);
    expect(presets.isDirty(CAMERA)).toBe(false);
  });

  it("keeps scopes in separate storage slots", () => {
    const { presets } = setup();
    presets.save(SPAWNER, "a");
    presets.save(CAMERA, "b");
    expect(presets.list(SPAWNER).map((p) => p.name)).toEqual(["a"]);
    expect(presets.list(CAMERA).map((p) => p.name)).toEqual(["b"]);
  });

  it("round-trips through localStorage", () => {
    const { store, mod, presets } = setup();
    store.set("spawner.rate", 3);
    mod.setBinding("spawner.rate", { source: "rms.low", lo: 1, hi: 2 });
    presets.save(SPAWNER, "a");

    const reloaded = new PresetStore(store, mod);
    expect(reloaded.current(SPAWNER)).toBe("a");
    expect(reloaded.list(SPAWNER)[0].mods["spawner.rate"].source).toBe("rms.low");
    // Insertion order differs after a JSON round-trip; dirty compares content.
    expect(reloaded.isDirty(SPAWNER)).toBe(false);
  });
});
