import type { ParamStore } from "./ParamStore";

const HOT_KEYS = ["hopSize", "smoothingAlpha", "dbFloor"] as const;

export class WorkletBridge {
  constructor(private store: ParamStore, private port: MessagePort) {
    store.subscribe((key) => this.handleChange(key));
  }

  bootstrap(): void {
    this.port.postMessage({
      type: "configure",
      windowSize: this.store.get("dsp.windowSize"),
      rmsHistoryLen: this.store.get("dsp.rmsHistoryLen"),
    });
    for (const k of HOT_KEYS) {
      this.port.postMessage({ type: "param", key: k, value: this.resolveHotValue(k) });
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
      const hotKey = key.slice("dsp.".length) as (typeof HOT_KEYS)[number];
      if (!HOT_KEYS.includes(hotKey)) return;
      this.port.postMessage({ type: "param", key: hotKey, value: this.resolveHotValue(hotKey) });
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
