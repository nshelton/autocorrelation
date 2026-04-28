import { beforeEach, describe, expect, it, vi } from "vitest";
import { ParamStore } from "../../src/params/ParamStore";
import { analysisSchemas } from "../../src/params/schemas";
import { WorkletBridge } from "../../src/params/WorkletBridge";

function makeStore(): ParamStore {
  const store = new ParamStore();
  for (const s of analysisSchemas) store.register(s);
  return store;
}

function makePort() {
  return { postMessage: vi.fn() } as unknown as MessagePort;
}

function makeMockPort() {
  const posted: unknown[] = [];
  const port = {
    postMessage: (msg: unknown) => {
      posted.push(msg);
    },
  } as unknown as MessagePort;
  return { port, posted };
}

describe("WorkletBridge", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("bootstrap posts one configure + three param messages with current store values", () => {
    const store = makeStore();
    const port = makePort();
    const bridge = new WorkletBridge(store, port);
    bridge.bootstrap();
    const calls = (port.postMessage as ReturnType<typeof vi.fn>).mock.calls.map((c) => c[0]);
    expect(calls).toContainEqual({ type: "configure", windowSize: 2048, rmsHistoryLen: 512 });
    expect(calls).toContainEqual({ type: "param", key: "hopSize", value: 1024 });
    expect(calls).toContainEqual({ type: "param", key: "smoothingTauSecs", value: 0.0956 });
    expect(calls).toContainEqual({ type: "param", key: "dbFloor", value: -100 });
    expect(calls.length).toBe(4);
  });

  it("windowSize change posts a configure message with both reconfig params", () => {
    const store = makeStore();
    const port = makePort();
    new WorkletBridge(store, port);
    (port.postMessage as ReturnType<typeof vi.fn>).mockClear();
    store.set("dsp.windowSize", 1024);
    expect(port.postMessage).toHaveBeenCalledWith({
      type: "configure",
      windowSize: 1024,
      rmsHistoryLen: 512,
    });
  });

  it("smoothingTauSecs change posts a param message with the hot key (no dsp prefix)", () => {
    const store = makeStore();
    const port = makePort();
    new WorkletBridge(store, port);
    (port.postMessage as ReturnType<typeof vi.fn>).mockClear();
    store.set("dsp.smoothingTauSecs", 0.5);
    expect(port.postMessage).toHaveBeenCalledWith({
      type: "param",
      key: "smoothingTauSecs",
      value: 0.5,
    });
  });

  it("hopSize > windowSize is clamped to windowSize before sending", () => {
    const store = makeStore();
    const port = makePort();
    new WorkletBridge(store, port);
    store.set("dsp.windowSize", 512);
    (port.postMessage as ReturnType<typeof vi.fn>).mockClear();
    store.set("dsp.hopSize", 1024);
    expect(port.postMessage).toHaveBeenCalledWith({
      type: "param",
      key: "hopSize",
      value: 512,
    });
  });

  it("dispose() unsubscribes from the store", () => {
    const store = new ParamStore();
    for (const s of analysisSchemas) store.register(s);
    const port = makeMockPort();
    const bridge = new WorkletBridge(store, port.port);
    port.posted.length = 0; // ignore subscription-time messages, if any

    bridge.dispose();

    // Mutate the store; after dispose, no further messages should be posted.
    store.set("dsp.smoothingTauSecs", 0.05);
    expect(port.posted).toHaveLength(0);
  });
});
