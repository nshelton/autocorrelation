import { beforeEach, describe, expect, it, vi } from "vitest";
import { ParamStore, type ParamSchema } from "../../src/params/ParamStore";

const STORAGE_KEY = "autocorrelation.params.v1";

const continuousSchema: ParamSchema = {
  key: "test.alpha",
  label: "Alpha",
  kind: "continuous",
  min: 0,
  max: 1,
  step: 0.01,
  default: 0.2,
  reconfig: false,
};

const fovSchema: ParamSchema = {
  key: "camera.fov",
  label: "FOV",
  kind: "continuous",
  min: 10,
  max: 120,
  step: 1,
  default: 60,
  reconfig: false,
};

const discreteSchema: ParamSchema = {
  key: "test.size",
  label: "Size",
  kind: "discrete",
  options: [256, 512, 1024],
  default: 512,
  reconfig: true,
};

describe("ParamStore", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("register loads default when no localStorage entry exists", () => {
    const store = new ParamStore();
    store.register(continuousSchema);
    expect(store.get("test.alpha")).toBe(0.2);
  });

  it("register restores persisted value from localStorage", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ "test.alpha": 0.7 }));
    const store = new ParamStore();
    store.register(continuousSchema);
    expect(store.get("test.alpha")).toBe(0.7);
  });

  it("set persists the full values map to localStorage", () => {
    const store = new ParamStore();
    store.register(continuousSchema);
    store.register(discreteSchema);
    store.set("test.alpha", 0.5);
    const persisted = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
    expect(persisted).toEqual({ "test.alpha": 0.5, "test.size": 512 });
  });

  it("set notifies subscribers synchronously with key + value", () => {
    const store = new ParamStore();
    store.register(continuousSchema);
    const fn = vi.fn();
    store.subscribe(fn);
    store.set("test.alpha", 0.3);
    expect(fn).toHaveBeenCalledWith("test.alpha", 0.3, "user");
  });

  it("set rejects out-of-range continuous values", () => {
    const store = new ParamStore();
    store.register(continuousSchema);
    const fn = vi.fn();
    store.subscribe(fn);
    store.set("test.alpha", 1.5);
    expect(store.get("test.alpha")).toBe(0.2);
    expect(fn).not.toHaveBeenCalled();
  });

  it("reset wipes localStorage and restores defaults, notifying for each changed key", () => {
    const store = new ParamStore();
    store.register(continuousSchema);
    store.register(discreteSchema);
    store.set("test.alpha", 0.9);
    store.set("test.size", 1024);
    const fn = vi.fn();
    store.subscribe(fn);
    store.reset();
    expect(store.get("test.alpha")).toBe(0.2);
    expect(store.get("test.size")).toBe(512);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(fn).toHaveBeenCalledWith("test.alpha", 0.2, "user");
    expect(fn).toHaveBeenCalledWith("test.size", 512, "user");
  });

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

  describe("notify / schemaFor / source arg", () => {
    let store: ParamStore;
    beforeEach(() => {
      store = new ParamStore();
      store.register(fovSchema);
    });

    it("notify() fires subscribers with source='modulator' without mutating value or persisting", () => {
      store.set("camera.fov", 75);
      const calls: Array<[string, unknown, string]> = [];
      store.subscribe((k, v, s) => calls.push([k, v, s]));
      store.notify("camera.fov", 90);
      expect(calls).toEqual([["camera.fov", 90, "modulator"]]);
      expect(store.get("camera.fov")).toBe(75);
      expect(JSON.parse(localStorage.getItem("autocorrelation.params.v1")!))
        .toEqual({ "camera.fov": 75 });
    });

    it("schemaFor() returns the schema by key", () => {
      expect(store.schemaFor("camera.fov")).toBe(fovSchema);
      expect(store.schemaFor("does.not.exist")).toBeUndefined();
    });
  });
});
