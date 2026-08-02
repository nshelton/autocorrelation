import type { ParamStore } from "./ParamStore";
import type { FeatureStore } from "../store/FeatureStore";

const STORAGE_KEY = "autocorrelation.modulation.v1";
// Triggers persist under their own key so the continuous-binding format above
// needs no migration.
const TRIGGER_STORAGE_KEY = "autocorrelation.triggers.v1";

export interface ModBinding {
  source: string;
  // Output range in param units: the smoothed + power-curved 0..1 signal maps
  // into [lo, hi]. Optional — default to the param's full [min, max]. Old
  // persisted bindings used `depth`; it's ignored, lo/hi fall back to the range.
  lo?: number;
  hi?: number;
  // Power curve applied to the 0..1 source value before mapping into [lo,hi]:
  // v^power. >1 emphasizes peaks (crushes lows), <1 lifts lows. Optional; 1.
  power?: number;
  // Cheap per-tick EMA on the source before the power curve: 0 = none,
  // →1 = heavy. alpha = 1 - smoothing (clamped > 0). Optional; treated as 0.
  smoothing?: number;
}

// A button trigger: fire the button's action once when the smoothed + power-
// curved `source` rises across `threshold` (rising edge), re-arming only after
// it drops back below. power/smoothing filter the source the same way as
// continuous modulation (see ModBinding); optional, treated as 1 / 0.
export interface TriggerBinding {
  source: string;
  threshold: number;
  power?: number;
  smoothing?: number;
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
  "rms.total": { buffer: "rms",    read: latest },
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
      this.smoothed.delete(key);
      this.processed.delete(key);
      this.persistTriggers();
      for (const fn of this.uiSubs) fn(key);
      return;
    }
    // Reseed the EMA only when the source changes (not on threshold/power/
    // smoothing tweaks), matching setBinding.
    if (this.triggers.get(key)?.source !== binding.source) this.smoothed.delete(key);
    this.triggers.set(key, { ...binding });
    this.triggerArmed.delete(key);   // re-seed arm state on the next sample
    this.persistTriggers();
    for (const fn of this.uiSubs) fn(key);
  }

  getTrigger(key: string): TriggerBinding | null {
    const t = this.triggers.get(key);
    return t ? { ...t } : null;
  }

  // Keys that currently have a binding / trigger. PresetStore filters these by
  // module prefix to snapshot and restore a module's modulation state.
  bindingKeys(): string[] {
    return [...this.bindings.keys()];
  }

  triggerKeys(): string[] {
    return [...this.triggers.keys()];
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
  // modulation or trigger — what the mod/trigger graph displays. 0 when unset.
  processedValue(key: string): number {
    return this.processed.get(key) ?? 0;
  }

  // EMA-smooth (alpha = 1 - smoothing) then apply the power curve to a source,
  // updating the per-key smoothing + processed state. Returns the 0..1 signal.
  // Shared by continuous modulation and button triggers so both filter the
  // input identically.
  private processSource(key: string, source: string, power: number, smoothing: number): number {
    const v = Math.max(0, this.readSource(source));
    const alpha = Math.max(1e-3, 1 - smoothing);
    const prev = this.smoothed.get(key);
    const sm = prev === undefined ? v : prev + alpha * (v - prev);
    this.smoothed.set(key, sm);
    const curved = Math.pow(sm, power);
    this.processed.set(key, curved);
    return curved;
  }

  tick(): void {
    for (const [key, b] of this.bindings) {
      const schema = this.store.schemaFor(key);
      if (!schema || schema.kind !== "continuous") continue;
      if (!MOD_SOURCES[b.source]) continue;
      const curved = this.processSource(key, b.source, b.power ?? 1, b.smoothing ?? 0);
      // Map the 0..1 signal into [lo, hi] (param units; default to full range).
      // During silence curved decays to 0, so the param settles at lo.
      const lo = b.lo ?? schema.min;
      const hi = b.hi ?? schema.max;
      const out = lo + (hi - lo) * curved;
      this.store.notify(key, out);
      this.emitValue(key, out);
    }

    // Button triggers: fire on a rising edge across threshold (of the same
    // smoothed + power-curved signal), re-arm on fall. processSource runs
    // before the callback check so the trigger graph updates regardless.
    for (const [key, t] of this.triggers) {
      const v = this.processSource(key, t.source, t.power ?? 1, t.smoothing ?? 0);
      const cb = this.triggerCallbacks.get(key);
      if (!cb) continue;
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
        source?: unknown; lo?: unknown; hi?: unknown; power?: unknown; smoothing?: unknown;
      };
      if (typeof candidate.source !== "string") continue;
      const source = LEGACY_SOURCE_ALIASES[candidate.source] ?? candidate.source;
      if (!(source in MOD_SOURCES)) continue;
      // NOTE: do NOT drop bindings whose param schema isn't registered yet —
      // component-param schemas register after the Modulator is constructed, so
      // dropping here loses them on a full page reload. tick() guards on the
      // schema being a registered continuous param, so an unknown key is just
      // inactive until its schema arrives. (Old bindings carried `depth`; it's
      // ignored — lo/hi fall back to the param's full range.)
      const binding: ModBinding = { source };
      if (typeof candidate.lo === "number") binding.lo = candidate.lo;
      if (typeof candidate.hi === "number") binding.hi = candidate.hi;
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
      const candidate = val as {
        source?: unknown; threshold?: unknown; power?: unknown; smoothing?: unknown;
      };
      if (typeof candidate.source !== "string") continue;
      if (typeof candidate.threshold !== "number") continue;
      const source = LEGACY_SOURCE_ALIASES[candidate.source] ?? candidate.source;
      if (!(source in MOD_SOURCES)) continue;
      const binding: TriggerBinding = { source, threshold: candidate.threshold };
      if (typeof candidate.power === "number") binding.power = candidate.power;
      if (typeof candidate.smoothing === "number") binding.smoothing = candidate.smoothing;
      this.triggers.set(key, binding);
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
