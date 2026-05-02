import type { ParamStore } from "./ParamStore";

const HOT_KEYS = [
  "hopSize",
  "smoothingTauSecs",
  "onsetSmoothingTauSecs",
  "dbFloor",
  "teaTauSecs",
  "teaSigma",
  "acfSmoothingSigma",
  "acfDecay",
  "phaseLock",
  "autoGain",
] as const;

export class WorkletBridge {
  private unsubscribe: () => void;

  constructor(
    private store: ParamStore,
    private port: MessagePort,
  ) {
    this.unsubscribe = store.subscribe((key) => this.handleChange(key));
  }

  dispose(): void {
    this.unsubscribe();
  }

  /**
   * Send the worklet the current store state as one configure message
   * + one param message per hot key. Must be called by the App after
   * constructing the worklet node and wiring `onmessage`. The class
   * subscribes to the store at construction, so any `store.set()` calls
   * after construction but before `bootstrap()` will post messages to a
   * worklet that may not yet be ready (the worklet's pre-boot pending
   * state covers this for `configure`, but `param` messages received
   * before `Dsp` is ready are silently dropped).
   */
  bootstrap(): void {
    this.port.postMessage({
      type: "configure",
      windowSize: this.store.get("dsp.windowSize"),
      rmsHistoryLen: this.store.get("dsp.rmsHistoryLen"),
    });
    for (const k of HOT_KEYS) {
      this.port.postMessage({
        type: "param",
        key: k,
        value: this.resolveHotValue(k),
      });
    }
  }

  private handleChange(key: string): void {
    if (key === "dsp.windowSize" || key === "dsp.rmsHistoryLen") {
      this.port.postMessage({
        type: "configure",
        windowSize: this.store.get("dsp.windowSize"),
        rmsHistoryLen: this.store.get("dsp.rmsHistoryLen"),
      });
      return;
    }
    if (key.startsWith("dsp.")) {
      const suffix = key.slice("dsp.".length);
      if (!(HOT_KEYS as readonly string[]).includes(suffix)) return;
      const hotKey = suffix as (typeof HOT_KEYS)[number];
      this.port.postMessage({
        type: "param",
        key: hotKey,
        value: this.resolveHotValue(hotKey),
      });
    }
  }

  private resolveHotValue(key: (typeof HOT_KEYS)[number]): number {
    const value = this.store.get(`dsp.${key}`);
    if (key === "hopSize") {
      return Math.min(value, this.store.get("dsp.windowSize"));
    }
    return value;
  }
}
