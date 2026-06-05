import type { ParamStore } from "./ParamStore";
import type { FeatureStore } from "../store/FeatureStore";

const STORAGE_KEY = "autocorrelation.modulation.v1";
// Triggers persist under their own key so the continuous-binding format above
// needs no migration.
const TRIGGER_STORAGE_KEY = "autocorrelation.triggers.v1";

export interface ModBinding {
  source: string;
  depth: number;
  // Power curve applied to the 0..1 source value before the depth lerp:
  // v^power. >1 emphasizes peaks (crushes lows), <1 lifts lows. Optional for
  // back-compat with bindings persisted before this existed; treated as 1.
  power?: number;
  // Cheap per-tick EMA on the source before the power curve: 0 = none,
  // →1 = heavy. alpha = 1 - smoothing (clamped > 0). Optional; treated as 0.
  smoothing?: number;
}

// A button trigger: fire the button's action once when `source` rises across
// `threshold` (rising edge), re-arming only after it drops back below.
export interface TriggerBinding {
  source: string;
  threshold: number;
}

type SourceDescriptor = {
  buffer: string;
  read: (b: Float32Array) => number;
};

function latest(b: Float32Array): number {
  return b.length === 0 ? 0 : b[b.length - 1];
}

const TWO_PI = Math.PI * 2;

// beatPulses[i] is a 0..1 saw = phase within the 1x/2x/4x/8x period cycle.
// Saw reads it straight; sin maps that phase through one full sine cycle per
// beat, remapped to 0..1. NaN phase (silence / no rhythm) flows through to the
// tick()'s !isFinite guard, which snaps the param back to its base value.
function beatSaw(i: number): (b: Float32Array) => number {
  return (b) => (b.length > i ? b[i] : 0);
}
function beatSin(i: number): (b: Float32Array) => number {
  return (b) => (b.length > i ? 0.5 + 0.5 * Math.sin(TWO_PI * b[i]) : 0);
}

// Curated audio sources. UI dropdown + persistence use these string keys.
export const MOD_SOURCES: Record<string, SourceDescriptor> = {
  "rms.low":  { buffer: "rmsLow",  read: latest },
  "rms.mid":  { buffer: "rmsMid",  read: latest },
  "rms.high": { buffer: "rmsHigh", read: latest },
  "beat.1x saw": { buffer: "beatPulses", read: beatSaw(0) },
  "beat.2x saw": { buffer: "beatPulses", read: beatSaw(1) },
  "beat.4x saw": { buffer: "beatPulses", read: beatSaw(2) },
  "beat.8x saw": { buffer: "beatPulses", read: beatSaw(3) },
  "beat.1x sin": { buffer: "beatPulses", read: beatSin(0) },
  "beat.2x sin": { buffer: "beatPulses", read: beatSin(1) },
  "beat.4x sin": { buffer: "beatPulses", read: beatSin(2) },
  "beat.8x sin": { buffer: "beatPulses", read: beatSin(3) },
};

export const MOD_SOURCE_KEYS = Object.keys(MOD_SOURCES);

// One-time migration: the beat sources used to be plain "beat.Nx" (saws).
// Map persisted bindings onto the renamed "beat.Nx saw" keys so saved
// modulations survive the rename. Safe to delete once no old configs remain.
const LEGACY_SOURCE_ALIASES: Record<string, string> = {
  "beat.1x": "beat.1x saw",
  "beat.2x": "beat.2x saw",
  "beat.4x": "beat.4x saw",
  "beat.8x": "beat.8x saw",
};

type UiSubscriber = (key: string) => void;
// Per-tick broadcast of the live modulated value, so the UI can show the
// actual driven value on the slider (not just the base).
type ValueSubscriber = (key: string, value: number) => void;

export class Modulator {
  private bindings = new Map<string, ModBinding>();
  private uiSubs = new Set<UiSubscriber>();
  private valueSubs = new Set<ValueSubscriber>();
  // Per-key EMA state for source smoothing, and the resulting smoothed+power
  // signal (0..1) that drives the modulation — exposed for the UI graph.
  private smoothed = new Map<string, number>();
  private processed = new Map<string, number>();
  // Persisted trigger config, keyed by button key.
  private triggers = new Map<string, TriggerBinding>();
  // Button actions, attached by the UI at bind time. NOT persisted (functions
  // can't serialize) — reattached on every panel build / HMR reconstruct.
  private triggerCallbacks = new Map<string, () => void>();
  // Rising-edge state per trigger: true when armed (waiting to cross up).
  // undefined until the first sample, which only sets the initial arm state.
  private triggerArmed = new Map<string, boolean>();

  constructor(
    private store: ParamStore,
    private features: FeatureStore,
  ) {
    this.load();
    this.loadTriggers();
  }

  setBinding(key: string, binding: ModBinding | null): void {
    if (binding === null) {
      if (!this.bindings.delete(key)) return;
      this.smoothed.delete(key);
      this.processed.delete(key);
      this.persist();
      // Snap consumer state AND the slider back to the base value.
      const schema = this.store.schemaFor(key);
      if (schema) {
        const base = this.store.get(key);
        if (typeof base === "number") {
          this.store.notify(key, base);
          this.emitValue(key, base);
        }
      }
      for (const fn of this.uiSubs) fn(key);
      return;
    }
    // Reseed the EMA only when the source itself changes, so tweaking
    // depth/power/smoothing doesn't flick the filter back to the raw value.
    if (this.bindings.get(key)?.source !== binding.source) this.smoothed.delete(key);
    this.bindings.set(key, { ...binding });
    this.persist();
    for (const fn of this.uiSubs) fn(key);
  }

  getBinding(key: string): ModBinding | null {
    const b = this.bindings.get(key);
    return b ? { ...b } : null;
  }

  setTrigger(key: string, binding: TriggerBinding | null): void {
    if (binding === null) {
      if (!this.triggers.delete(key)) return;
      this.triggerArmed.delete(key);
      this.persistTriggers();
      for (const fn of this.uiSubs) fn(key);
      return;
    }
    this.triggers.set(key, { ...binding });
    this.triggerArmed.delete(key);   // re-seed arm state on the next sample
    this.persistTriggers();
    for (const fn of this.uiSubs) fn(key);
  }

  getTrigger(key: string): TriggerBinding | null {
    const t = this.triggers.get(key);
    return t ? { ...t } : null;
  }

  // Attach the action a trigger fires. Called by the UI for every button each
  // panel build; cleared on dispose so stale closures never fire.
  registerTriggerCallback(key: string, fn: () => void): void {
    this.triggerCallbacks.set(key, fn);
  }

  // Live value of a source in 0..1, NaN/unknown → 0. Used by the trigger tick
  // and the trigger source monitor.
  readSource(source: string): number {
    const src = MOD_SOURCES[source];
    if (!src) return 0;
    const v = src.read(this.features.get(src.buffer));
    return Number.isFinite(v) ? v : 0;
  }

  // The smoothed + power-curved signal (0..1) currently driving a key's
  // modulation — what the continuous mod graph displays. 0 when unmodulated.
  processedValue(key: string): number {
    return this.processed.get(key) ?? 0;
  }

  tick(): void {
    for (const [key, b] of this.bindings) {
      const schema = this.store.schemaFor(key);
      if (!schema || schema.kind !== "continuous") continue;
      const src = MOD_SOURCES[b.source];
      if (!src) continue;
      const buf = this.features.get(src.buffer);
      const raw = src.read(buf);
      const base = this.store.get(key) as number;
      const hasData = buf.length > 0 && Number.isFinite(raw);
      // EMA-smooth the (non-negative) source. During no-data the input decays
      // toward 0 so the graph falls, but the param still rests at base below.
      const v = hasData ? Math.max(0, raw) : 0;
      const alpha = Math.max(1e-3, 1 - (b.smoothing ?? 0));
      const prev = this.smoothed.get(key);
      const sm = prev === undefined ? v : prev + alpha * (v - prev);
      this.smoothed.set(key, sm);
      // Power curve on the smoothed value → the 0..1 signal driving modulation.
      const curved = Math.pow(sm, b.power ?? 1);
      this.processed.set(key, curved);

      if (!hasData) {
        this.store.notify(key, base);
        this.emitValue(key, base);
        continue;
      }
      const target = schema.min + (schema.max - schema.min) * curved;
      const out = base + (target - base) * b.depth;
      this.store.notify(key, out);
      this.emitValue(key, out);
    }

    // Button triggers: fire on a rising edge across threshold, re-arm on fall.
    for (const [key, t] of this.triggers) {
      const cb = this.triggerCallbacks.get(key);
      if (!cb) continue;
      const v = this.readSource(t.source);
      const armed = this.triggerArmed.get(key);
      if (armed === undefined) {
        // First sample: just set the initial arm state, never fire on load.
        this.triggerArmed.set(key, v < t.threshold);
        continue;
      }
      if (armed && v >= t.threshold) {
        this.triggerArmed.set(key, false);
        cb();
      } else if (!armed && v < t.threshold) {
        this.triggerArmed.set(key, true);
      }
    }
  }

  subscribe(fn: UiSubscriber): () => void {
    this.uiSubs.add(fn);
    return () => this.uiSubs.delete(fn);
  }

  // UI subscribes here to track the live modulated value of a key each tick.
  subscribeValue(fn: ValueSubscriber): () => void {
    this.valueSubs.add(fn);
    return () => this.valueSubs.delete(fn);
  }

  private emitValue(key: string, value: number): void {
    for (const fn of this.valueSubs) fn(key, value);
  }

  // Bindings + triggers intentionally kept — they persist in localStorage and
  // reload on the next constructor call (used by HMR teardown/reconstruct).
  // Trigger callbacks ARE cleared: they're closures over the old panel and get
  // reattached by the next panel build.
  dispose(): void {
    this.uiSubs.clear();
    this.valueSubs.clear();
    this.triggerCallbacks.clear();
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
      const candidate = val as {
        source?: unknown; depth?: unknown; power?: unknown; smoothing?: unknown;
      };
      if (typeof candidate.source !== "string") continue;
      if (typeof candidate.depth !== "number") continue;
      const source = LEGACY_SOURCE_ALIASES[candidate.source] ?? candidate.source;
      if (!(source in MOD_SOURCES)) continue;
      if (!this.store.schemaFor(key)) continue;
      const binding: ModBinding = { source, depth: candidate.depth };
      if (typeof candidate.power === "number") binding.power = candidate.power;
      if (typeof candidate.smoothing === "number") binding.smoothing = candidate.smoothing;
      this.bindings.set(key, binding);
    }
  }

  private persist(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.bindings)));
    } catch {
      // localStorage unavailable; no-op
    }
  }

  private loadTriggers(): void {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(TRIGGER_STORAGE_KEY);
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
      const candidate = val as { source?: unknown; threshold?: unknown };
      if (typeof candidate.source !== "string") continue;
      if (typeof candidate.threshold !== "number") continue;
      const source = LEGACY_SOURCE_ALIASES[candidate.source] ?? candidate.source;
      if (!(source in MOD_SOURCES)) continue;
      this.triggers.set(key, { source, threshold: candidate.threshold });
    }
  }

  private persistTriggers(): void {
    try {
      localStorage.setItem(TRIGGER_STORAGE_KEY, JSON.stringify(Object.fromEntries(this.triggers)));
    } catch {
      // localStorage unavailable; no-op
    }
  }
}
