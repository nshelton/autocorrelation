import { describe, it, expect } from "vitest";
import { FeatureStore } from "../../src/store/FeatureStore";

describe("FeatureStore", () => {
  it("returns an empty Float32Array for an unknown key", () => {
    const store = new FeatureStore();
    expect(store.get("nope").length).toBe(0);
  });

  it("stores and returns the latest buffer for a key", () => {
    const store = new FeatureStore();
    const a = new Float32Array([1, 2, 3]);
    store.set("waveform", a);
    expect(store.get("waveform")).toBe(a);
  });

  it("set replaces the previous buffer", () => {
    const store = new FeatureStore();
    store.set("waveform", new Float32Array([1]));
    const second = new Float32Array([9, 9]);
    store.set("waveform", second);
    expect(store.get("waveform")).toBe(second);
  });
});
