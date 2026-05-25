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
