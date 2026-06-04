import type { ParamStore } from "./ParamStore";
import type { FeatureStore } from "../store/FeatureStore";

const STORAGE_KEY = "autocorrelation.modulation.v1";

export interface ModBinding {
  source: string;
  depth: number;
}

type SourceDescriptor = {
  buffer: string;
  read: (b: Float32Array) => number;
};

function latest(b: Float32Array): number {
  return b.length === 0 ? 0 : b[b.length - 1];
}

// Curated audio sources. UI dropdown + persistence use these string keys.
// Beat saws read fixed indices of beatPulses (1x/2x/4x/8x phase, 0..1 saw).
export const MOD_SOURCES: Record<string, SourceDescriptor> = {
  "rms.low":  { buffer: "rmsLow",     read: latest },
  "rms.mid":  { buffer: "rmsMid",     read: latest },
  "rms.high": { buffer: "rmsHigh",    read: latest },
  "beat.1x":  { buffer: "beatPulses", read: (b) => (b.length > 0 ? b[0] : 0) },
  "beat.2x":  { buffer: "beatPulses", read: (b) => (b.length > 1 ? b[1] : 0) },
  "beat.4x":  { buffer: "beatPulses", read: (b) => (b.length > 2 ? b[2] : 0) },
  "beat.8x":  { buffer: "beatPulses", read: (b) => (b.length > 3 ? b[3] : 0) },
};

export const MOD_SOURCE_KEYS = Object.keys(MOD_SOURCES);

type UiSubscriber = (key: string) => void;

export class Modulator {
  private bindings = new Map<string, ModBinding>();
  private uiSubs = new Set<UiSubscriber>();

  constructor(
    private store: ParamStore,
    private features: FeatureStore,
  ) {
    this.load();
  }

  setBinding(key: string, binding: ModBinding | null): void {
    if (binding === null) {
      if (!this.bindings.delete(key)) return;
      this.persist();
      // Snap consumer state back to the slider value.
      const schema = this.store.schemaFor(key);
      if (schema) {
        const base = this.store.get(key);
        if (typeof base === "number") this.store.notify(key, base);
      }
      for (const fn of this.uiSubs) fn(key);
      return;
    }
    this.bindings.set(key, { ...binding });
    this.persist();
    for (const fn of this.uiSubs) fn(key);
  }

  getBinding(key: string): ModBinding | null {
    const b = this.bindings.get(key);
    return b ? { ...b } : null;
  }

  tick(): void {
    for (const [key, b] of this.bindings) {
      const schema = this.store.schemaFor(key);
      if (!schema || schema.kind !== "continuous") continue;
      const src = MOD_SOURCES[b.source];
      if (!src) continue;
      const buf = this.features.get(src.buffer);
      const v = src.read(buf);
      const base = this.store.get(key) as number;
      if (buf.length === 0 || !Number.isFinite(v)) {
        this.store.notify(key, base);
        continue;
      }
      const target = schema.min + (schema.max - schema.min) * v;
      this.store.notify(key, base + (target - base) * b.depth);
    }
  }

  subscribe(fn: UiSubscriber): () => void {
    this.uiSubs.add(fn);
    return () => this.uiSubs.delete(fn);
  }

  // Bindings intentionally kept — they persist in localStorage and reload
  // on the next constructor call (used by HMR teardown/reconstruct).
  dispose(): void {
    this.uiSubs.clear();
  }

  private load(): void {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(STORAGE_KEY);
    } catch {
      return;
    }
    if (!raw) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return;
    }
    if (!parsed || typeof parsed !== "object") return;
    for (const [key, val] of Object.entries(parsed as Record<string, unknown>)) {
      if (!val || typeof val !== "object") continue;
      const candidate = val as { source?: unknown; depth?: unknown };
      if (typeof candidate.source !== "string") continue;
      if (typeof candidate.depth !== "number") continue;
      if (!(candidate.source in MOD_SOURCES)) continue;
      if (!this.store.schemaFor(key)) continue;
      this.bindings.set(key, { source: candidate.source, depth: candidate.depth });
    }
  }

  private persist(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.bindings)));
    } catch {
      // localStorage unavailable; no-op
    }
  }
}
