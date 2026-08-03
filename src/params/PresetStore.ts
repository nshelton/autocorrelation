import type { ParamStore, ParamValue } from "./ParamStore";
import type { ModBinding, Modulator, TriggerBinding } from "./Modulator";
import { presetTween } from "./PresetTween";

// Seconds to glide from the live values to a loaded preset's. 0 = snap. Read
// per apply(), so changing it mid-tween doesn't disturb the one in flight.
const TWEEN_KEY = "system.presetTweenSecs";

// What a preset section owns. `id` is the storage/fold key; `prefixes` are the
// param key namespaces it captures — usually one (a component's param prefix),
// but Camera spans both "camera." and "light.". A preset never reaches outside
// its prefixes, which is why `components.<id>.enabled` is not captured for
// components: loading a preset shouldn't toggle the module on or off.
export interface PresetScope {
  id: string;
  // EMPTY = the whole system: every registered param and every modulation,
  // component enable flags included. That's the system-preset scope.
  prefixes: string[];
}

export interface Preset {
  name: string;
  params: Record<string, ParamValue>;
  mods: Record<string, ModBinding>;
  triggers: Record<string, TriggerBinding>;
  // Small JPEG data URL of the scene at save time. System presets only —
  // module presets carry no image. Kept tiny (see render/thumbnail.ts)
  // because every preset shares one localStorage key.
  thumb?: string;
}

type ScopeState = { current: string | null; list: Preset[] };

const STORAGE_KEY = "autocorrelation.presets.v1";

export class PresetStore {
  private scopes = new Map<string, ScopeState>();
  private subs = new Set<(scope: string) => void>();

  constructor(
    private store: ParamStore,
    private modulator: Modulator,
  ) {
    this.load();
  }

  list(scope: PresetScope): Preset[] {
    return this.state(scope.id).list;
  }

  current(scope: PresetScope): string | null {
    return this.state(scope.id).current;
  }

  // Overwrites the preset of the same name, otherwise appends. Either way the
  // saved preset becomes current, so the UI immediately reads back clean.
  save(scope: PresetScope, name: string, thumb?: string): void {
    const st = this.state(scope.id);
    const i = st.list.findIndex((p) => p.name === name);
    // Overwriting without a fresh capture keeps the existing image.
    const preset: Preset = { name, ...this.capture(scope), thumb: thumb ?? st.list[i]?.thumb };
    if (i >= 0) st.list[i] = preset;
    else st.list.push(preset);
    st.current = name;
    this.persist();
    this.emit(scope.id);
  }

  apply(scope: PresetScope, name: string): void {
    const st = this.state(scope.id);
    const preset = st.list.find((p) => p.name === name);
    if (!preset) return;
    // Structure snaps, values glide. Booleans (module enables) and discrete
    // dropdowns (primitive type, force field, tonemap mode) have no meaningful
    // in-between, so they land immediately; continuous params and colors are
    // handed to the tween. Everything goes through store.set in the end, so the
    // usual "user" subscribers still mirror into params bags and re-pull sliders.
    const targets = new Map<string, number>();
    for (const [key, value] of Object.entries(preset.params)) {
      const schema = this.store.schemaFor(key);
      if (!schema) continue;
      if (typeof value === "number" && (schema.kind === "continuous" || schema.kind === "color")) {
        targets.set(key, value);
      } else {
        this.store.set(key, value);
      }
    }
    presetTween.start(this.store, targets, this.tweenSecs());
    // Clear in-scope bindings the preset doesn't carry, then write its own —
    // otherwise a modulation added since the save would survive the load.
    for (const key of this.modulator.bindingKeys()) {
      if (inScope(scope, key) && !(key in preset.mods)) this.modulator.setBinding(key, null);
    }
    for (const [key, b] of Object.entries(preset.mods)) this.modulator.setBinding(key, b);
    for (const key of this.modulator.triggerKeys()) {
      if (inScope(scope, key) && !(key in preset.triggers)) this.modulator.setTrigger(key, null);
    }
    for (const [key, t] of Object.entries(preset.triggers)) this.modulator.setTrigger(key, t);
    st.current = name;
    this.persist();
    this.emit(scope.id);
  }

  remove(scope: PresetScope, name: string): void {
    const st = this.state(scope.id);
    const i = st.list.findIndex((p) => p.name === name);
    if (i < 0) return;
    st.list.splice(i, 1);
    if (st.current === name) st.current = null;
    this.persist();
    this.emit(scope.id);
  }

  // True when the live state has drifted from the current preset — the "*" in
  // the panel. No current preset means nothing to be dirty against.
  isDirty(scope: PresetScope): boolean {
    const st = this.state(scope.id);
    if (!st.current) return false;
    const saved = st.list.find((p) => p.name === st.current);
    if (!saved) return false;
    const { params, mods, triggers } = saved;
    return stable(this.capture(scope)) !== stable({ params, mods, triggers });
  }

  subscribe(fn: (scope: string) => void): () => void {
    this.subs.add(fn);
    return () => this.subs.delete(fn);
  }

  private capture(scope: PresetScope): Omit<Preset, "name"> {
    const params: Record<string, ParamValue> = {};
    for (const schema of this.store.schemasInOrder()) {
      if (inScope(scope, schema.key)) params[schema.key] = this.store.get(schema.key);
    }
    const mods: Record<string, ModBinding> = {};
    for (const key of this.modulator.bindingKeys()) {
      if (!inScope(scope, key)) continue;
      const b = this.modulator.getBinding(key);
      if (b) mods[key] = b;
    }
    const triggers: Record<string, TriggerBinding> = {};
    for (const key of this.modulator.triggerKeys()) {
      if (!inScope(scope, key)) continue;
      const t = this.modulator.getTrigger(key);
      if (t) triggers[key] = t;
    }
    return { params, mods, triggers };
  }

  // 0 when the schema isn't registered (tests, early boot) — presets snap.
  private tweenSecs(): number {
    if (!this.store.schemaFor(TWEEN_KEY)) return 0;
    const v = this.store.get(TWEEN_KEY);
    return typeof v === "number" ? v : 0;
  }

  private state(scope: string): ScopeState {
    let st = this.scopes.get(scope);
    if (!st) {
      st = { current: null, list: [] };
      this.scopes.set(scope, st);
    }
    return st;
  }

  private emit(scope: string): void {
    for (const fn of this.subs) fn(scope);
  }

  private load(): void {
    let parsed: unknown;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      parsed = JSON.parse(raw);
    } catch {
      return;
    }
    if (!parsed || typeof parsed !== "object") return;
    for (const [scope, val] of Object.entries(parsed as Record<string, unknown>)) {
      const st = val as Partial<ScopeState>;
      if (!Array.isArray(st?.list)) continue;
      const list = st.list.filter(
        (p): p is Preset => !!p && typeof p.name === "string" && !!p.params,
      );
      // Normalize: presets written before mods/triggers existed lack those.
      for (const p of list) {
        p.mods ??= {};
        p.triggers ??= {};
      }
      const current = typeof st.current === "string" && list.some((p) => p.name === st.current)
        ? st.current
        : null;
      this.scopes.set(scope, { current, list });
    }
  }

  private persist(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.scopes)));
    } catch (e) {
      // Unavailable or over quota (thumbnails are the bulk of it) — the preset
      // lives for this session only. Loud, because silently losing a save the
      // user just made looks like the feature is broken.
      console.warn("PresetStore: could not persist presets", e);
    }
  }
}

function inScope(scope: PresetScope, key: string): boolean {
  if (scope.prefixes.length === 0) return true; // whole-system scope
  return scope.prefixes.some((p) => key.startsWith(`${p}.`));
}

// Key-sorted stringify so dirty-checking compares content, not insertion order
// (a freshly captured mods map iterates in binding order, a loaded one in JSON
// order). Undefined-valued fields are skipped so an omitted optional and an
// explicit `undefined` compare equal.
function stable(v: unknown): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v) ?? "null";
  if (Array.isArray(v)) return `[${v.map(stable).join(",")}]`;
  const o = v as Record<string, unknown>;
  const parts = Object.keys(o)
    .sort()
    .filter((k) => o[k] !== undefined)
    .map((k) => `${JSON.stringify(k)}:${stable(o[k])}`);
  return `{${parts.join(",")}}`;
}
