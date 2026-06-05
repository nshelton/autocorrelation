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
    mod.setBinding("post.bloom.strength", { source: "beat.1x saw", depth: 1 });
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

  it("beat saw sources read indexed slot of beatPulses", () => {
    const { store, features, mod } = setup();
    const buf = new Float32Array([0.1, 0.4, 0.7, 1.0]);
    features.set("beatPulses", buf);
    mod.setBinding("camera.fov", { source: "beat.4x saw", depth: 1 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    // lerp(20, 120, buf[2]) — use the Float32 value directly to match runtime precision
    const expected = 20 + (120 - 20) * buf[2];
    expect(spy).toHaveBeenCalledWith("camera.fov", expected, "modulator");
  });

  it("beat sin sources map beat phase through a full sine cycle (0..1)", () => {
    const { store, features, mod } = setup();
    // phase 0.25 → sin(π/2)=1 → mapped to 1.0 (peak of the cycle)
    const buf = new Float32Array([0.1, 0.25, 0.7, 1.0]);
    features.set("beatPulses", buf);
    mod.setBinding("camera.fov", { source: "beat.2x sin", depth: 1 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    const v = 0.5 + 0.5 * Math.sin(2 * Math.PI * buf[1]);
    const expected = 20 + (120 - 20) * v;
    expect(spy).toHaveBeenCalledWith("camera.fov", expected, "modulator");
  });

  it("power curve shapes the source before the depth lerp", () => {
    const { store, features, mod } = setup();
    features.set("rmsLow", new Float32Array([0, 0.5]));
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1, power: 2 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    // 0.5^2 = 0.25 → lerp(20, 120, 0.25) = 45
    expect(spy).toHaveBeenCalledWith("camera.fov", 45, "modulator");
  });

  it("missing power defaults to 1 (linear)", () => {
    const { store, features, mod } = setup();
    features.set("rmsLow", new Float32Array([0, 0.5]));
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1 });
    const spy = vi.fn();
    store.subscribe(spy);
    mod.tick();
    // 0.5^1 = 0.5 → lerp(20, 120, 0.5) = 70
    expect(spy).toHaveBeenCalledWith("camera.fov", 70, "modulator");
  });

  it("legacy beat.Nx bindings migrate to beat.Nx saw on load", () => {
    localStorage.clear();
    const store = new ParamStore();
    store.register(FOV);
    // Seed persisted state BEFORE the Modulator is constructed (load() runs in
    // the ctor); can't use setup() because it clears localStorage first.
    localStorage.setItem(
      "autocorrelation.modulation.v1",
      JSON.stringify({ "camera.fov": { source: "beat.1x", depth: 0.5 } }),
    );
    const mod = new Modulator(store, new FeatureStore());
    expect(mod.getBinding("camera.fov")).toEqual({ source: "beat.1x saw", depth: 0.5 });
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

  it("subscribeValue broadcasts the live modulated value each tick", () => {
    const { features, mod } = setup();
    features.set("rmsLow", new Float32Array([0, 0.25]));
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1 });
    const spy = vi.fn();
    mod.subscribeValue(spy);
    mod.tick();
    // lerp(20, 120, 0.25) = 45
    expect(spy).toHaveBeenCalledWith("camera.fov", 45);
  });

  it("subscribeValue gets the base value back when modulation is removed", () => {
    const { features, mod } = setup();
    features.set("rmsLow", new Float32Array([1.0]));
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1 });
    const spy = vi.fn();
    mod.subscribeValue(spy);
    mod.setBinding("camera.fov", null);
    expect(spy).toHaveBeenCalledWith("camera.fov", 60);   // base
  });

  it("smoothing applies a cheap EMA to the source", () => {
    const { features, mod } = setup();
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1, smoothing: 0.5 });
    const spy = vi.fn();
    mod.subscribeValue(spy);
    features.set("rmsLow", new Float32Array([0]));
    mod.tick();                       // seed EMA at 0
    features.set("rmsLow", new Float32Array([1]));
    mod.tick();                       // alpha=0.5 → sm=0.5 → lerp(20,120,0.5)=70
    expect(spy).toHaveBeenLastCalledWith("camera.fov", 70);
  });

  it("processedValue exposes the smoothed+power signal in 0..1", () => {
    const { features, mod } = setup();
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1, power: 2 });
    features.set("rmsLow", new Float32Array([0.5]));
    mod.tick();
    // smoothing default 0 → sm=0.5; power 2 → 0.25
    expect(mod.processedValue("camera.fov")).toBeCloseTo(0.25);
  });

  it("processedValue is 0 for an unmodulated key", () => {
    const { mod } = setup();
    expect(mod.processedValue("camera.fov")).toBe(0);
  });

  it("missing smoothing defaults to none (raw passes through)", () => {
    const { features, mod } = setup();
    features.set("rmsLow", new Float32Array([0.25]));
    mod.setBinding("camera.fov", { source: "rms.low", depth: 1 });
    const spy = vi.fn();
    mod.subscribeValue(spy);
    mod.tick();
    // no smoothing → sm=0.25 → lerp(20,120,0.25)=45 on the very first tick
    expect(spy).toHaveBeenCalledWith("camera.fov", 45);
  });

  // ---- triggers ----

  const TRIG = "orbitalCloud.button.RandomizeSH";

  it("readSource returns finite source value, NaN/unknown → 0", () => {
    const { features, mod } = setup();
    features.set("rmsLow", new Float32Array([0.1, 0.7]));
    features.set("beatPulses", new Float32Array([NaN, NaN, NaN, NaN]));
    expect(mod.readSource("rms.low")).toBeCloseTo(0.7);
    expect(mod.readSource("beat.1x saw")).toBe(0);   // NaN → 0
    expect(mod.readSource("none")).toBe(0);          // unknown → 0
  });

  it("trigger fires once on a rising edge across threshold", () => {
    const { features, mod } = setup();
    const fire = vi.fn();
    mod.registerTriggerCallback(TRIG, fire);
    mod.setTrigger(TRIG, { source: "rms.low", threshold: 0.5 });

    features.set("rmsLow", new Float32Array([0.2]));
    mod.tick();                       // first sample: arms (0.2 < 0.5), no fire
    expect(fire).not.toHaveBeenCalled();

    features.set("rmsLow", new Float32Array([0.8]));
    mod.tick();                       // rising edge → fire
    expect(fire).toHaveBeenCalledTimes(1);

    mod.tick();                       // still above → no re-fire (disarmed)
    expect(fire).toHaveBeenCalledTimes(1);
  });

  it("trigger re-arms after the source drops back below threshold", () => {
    const { features, mod } = setup();
    const fire = vi.fn();
    mod.registerTriggerCallback(TRIG, fire);
    mod.setTrigger(TRIG, { source: "rms.low", threshold: 0.5 });

    features.set("rmsLow", new Float32Array([0.2]));
    mod.tick();                       // arm
    features.set("rmsLow", new Float32Array([0.8]));
    mod.tick();                       // fire (1)
    features.set("rmsLow", new Float32Array([0.1]));
    mod.tick();                       // drop below → re-arm
    features.set("rmsLow", new Float32Array([0.9]));
    mod.tick();                       // rising edge again → fire (2)
    expect(fire).toHaveBeenCalledTimes(2);
  });

  it("trigger does not fire on the first sample when already above threshold", () => {
    const { features, mod } = setup();
    const fire = vi.fn();
    mod.registerTriggerCallback(TRIG, fire);
    mod.setTrigger(TRIG, { source: "rms.low", threshold: 0.5 });

    features.set("rmsLow", new Float32Array([0.9]));
    mod.tick();                       // first sample above → initial state only
    expect(fire).not.toHaveBeenCalled();
  });

  it("trigger does nothing without a registered callback", () => {
    const { features, mod } = setup();
    mod.setTrigger(TRIG, { source: "rms.low", threshold: 0.5 });
    features.set("rmsLow", new Float32Array([0.2]));
    mod.tick();
    features.set("rmsLow", new Float32Array([0.9]));
    expect(() => mod.tick()).not.toThrow();   // no callback → no-op, no crash
  });

  it("triggers persist across instances under their own key", () => {
    const { features, mod } = setup();
    mod.setTrigger(TRIG, { source: "beat.1x saw", threshold: 0.3 });
    const mod2 = new Modulator(new ParamStore(), features);
    expect(mod2.getTrigger(TRIG)).toEqual({ source: "beat.1x saw", threshold: 0.3 });
  });
});
