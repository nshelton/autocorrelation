import { beforeEach, describe, expect, it } from "vitest";
import { ParamStore, type ParamSchema } from "../../src/params/ParamStore";
import { FeatureStore } from "../../src/store/FeatureStore";
import { Modulator } from "../../src/params/Modulator";
import { PresetStore, type PresetScope } from "../../src/params/PresetStore";
import { presetTween } from "../../src/params/PresetTween";

const SPAWNER: PresetScope = { id: "spawner", prefixes: ["spawner"] };

function num(key: string, def: number): ParamSchema {
  return { key, label: key, kind: "continuous", min: 0, max: 100, step: 0.01, default: def, reconfig: false };
}

function setup(tweenSecs = 1) {
  localStorage.clear();
  presetTween.cancel();
  const store = new ParamStore();
  store.register(num("spawner.rate", 0));
  store.register(num("spawner.size", 0));
  store.register({
    key: "spawner.tint",
    label: "tint",
    kind: "color",
    default: 0x000000,
    reconfig: false,
  });
  store.register({
    key: "spawner.mode",
    label: "mode",
    kind: "discrete",
    options: [0, 1],
    default: 0,
    reconfig: false,
  });
  store.register({
    key: "components.spawner.enabled",
    label: "enabled",
    kind: "boolean",
    default: true,
    reconfig: false,
  });
  store.register({
    key: "system.presetTweenSecs",
    label: "preset tween",
    kind: "continuous",
    min: 0,
    max: 5,
    step: 0.05,
    default: tweenSecs,
    reconfig: false,
  });
  const mod = new Modulator(store, new FeatureStore());
  return { store, mod, presets: new PresetStore(store, mod) };
}

describe("PresetTween", () => {
  beforeEach(() => {
    localStorage.clear();
    presetTween.cancel();
  });

  it("glides continuous params instead of clobbering them", () => {
    const { store, presets } = setup(1);
    store.set("spawner.rate", 0);
    presets.save(SPAWNER, "zero");
    store.set("spawner.rate", 100);
    presets.save(SPAWNER, "hundred");

    presets.apply(SPAWNER, "zero");
    // Nothing moves on the apply itself.
    expect(store.get("spawner.rate")).toBe(100);

    presetTween.tick(0.5, store);
    const mid = store.get("spawner.rate") as number;
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(100);

    presetTween.tick(0.5, store);
    expect(store.get("spawner.rate")).toBe(0);
    expect(presetTween.active).toBe(false);
  });

  it("lands exactly on the target so the preset reads clean", () => {
    const { store, presets } = setup(1);
    store.set("spawner.rate", 33.33);
    presets.save(SPAWNER, "a");
    store.set("spawner.rate", 7);

    presets.apply(SPAWNER, "a");
    for (let i = 0; i < 100; i++) presetTween.tick(1 / 60, store);
    expect(store.get("spawner.rate")).toBe(33.33);
    expect(presets.isDirty(SPAWNER)).toBe(false);
  });

  it("snaps booleans and discrete params immediately", () => {
    const { store, presets } = setup(1);
    store.set("spawner.mode", 1);
    store.set("components.spawner.enabled", false);
    presets.save(SPAWNER, "a");
    // Scope is spawner.*, so the enable flag is not in this preset; drive the
    // discrete one back and check apply() lands it without a tick.
    store.set("spawner.mode", 0);
    presets.apply(SPAWNER, "a");
    expect(store.get("spawner.mode")).toBe(1);
  });

  it("interpolates colors per channel", () => {
    const { store, presets } = setup(1);
    store.set("spawner.tint", 0x000000);
    presets.save(SPAWNER, "black");
    store.set("spawner.tint", 0xffffff);

    presets.apply(SPAWNER, "black");
    presetTween.tick(0.5, store);
    const mid = store.get("spawner.tint") as number;
    // Halfway between white and black is mid-gray on every channel.
    expect((mid >> 16) & 0xff).toBeGreaterThan(0x40);
    expect((mid >> 16) & 0xff).toBeLessThan(0xc0);
    expect((mid >> 16) & 0xff).toBe(mid & 0xff);
  });

  it("snaps when the tween length is 0", () => {
    const { store, presets } = setup(0);
    store.set("spawner.rate", 10);
    presets.save(SPAWNER, "a");
    store.set("spawner.rate", 90);

    presets.apply(SPAWNER, "a");
    expect(store.get("spawner.rate")).toBe(10);
    expect(presetTween.active).toBe(false);
  });

  it("re-targets from the mid-flight value when a second preset is loaded", () => {
    const { store, presets } = setup(1);
    store.set("spawner.rate", 0);
    presets.save(SPAWNER, "zero");
    store.set("spawner.rate", 100);
    presets.save(SPAWNER, "hundred");

    presets.apply(SPAWNER, "zero");
    presetTween.tick(0.5, store);
    const mid = store.get("spawner.rate") as number;

    presets.apply(SPAWNER, "hundred");
    presetTween.tick(0.01, store);
    // Continues from where it was, not from 0.
    const after = store.get("spawner.rate") as number;
    expect(after).toBeGreaterThanOrEqual(Math.min(mid, 100) - 1);
    presetTween.tick(1, store);
    expect(store.get("spawner.rate")).toBe(100);
  });

  it("skips params already at the target", () => {
    const { store, presets } = setup(1);
    store.set("spawner.rate", 5);
    store.set("spawner.size", 5);
    presets.save(SPAWNER, "a");
    store.set("spawner.rate", 50);

    presets.apply(SPAWNER, "a");
    const writes: string[] = [];
    store.subscribe((k) => writes.push(k));
    presetTween.tick(0.1, store);
    expect(writes).toEqual(["spawner.rate"]);
  });
});
