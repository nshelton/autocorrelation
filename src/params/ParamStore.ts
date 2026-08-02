export type ParamValue = number | boolean;

export type ParamSchema = {
  key: string;
  label: string;
  default: ParamValue;
  reconfig: boolean;
} & (
  | { kind: "discrete"; options: number[]; optionLabels?: string[] }
  | { kind: "continuous"; min: number; max: number; step: number }
  | { kind: "boolean" }
  // Packed 0xRRGGBB sRGB integer. Stored as a plain number so persistence,
  // presets and reset need no special casing; only the widget differs.
  | { kind: "color" }
);

export const COLOR_MAX = 0xffffff;

type Subscriber = (key: string, value: ParamValue, source: "user" | "modulator") => void;

const STORAGE_KEY = "autocorrelation.params.v1";

export class ParamStore {
  private values = new Map<string, ParamValue>();
  private schemas = new Map<string, ParamSchema>();
  private subscribers = new Set<Subscriber>();
  private persisted: Record<string, ParamValue>;

  constructor() {
    this.persisted = this.readPersisted();
  }

  register(schema: ParamSchema): void {
    this.schemas.set(schema.key, schema);
    // Idempotent: keep in-memory value across HMR re-registers. Only seed
    // from persisted/default the first time a key shows up.
    if (this.values.has(schema.key)) return;
    const initial = this.persisted[schema.key];
    // Drop a persisted value whose type no longer matches the schema (e.g.
    // a key was repurposed). Fall back to default rather than throwing.
    const usable =
      initial !== undefined && this.matchesKind(schema, initial)
        ? initial
        : schema.default;
    this.values.set(schema.key, usable);
  }

  get(key: string): ParamValue {
    if (!this.values.has(key)) throw new Error(`ParamStore: unknown key ${key}`);
    return this.values.get(key)!;
  }

  set(key: string, value: ParamValue): void {
    const schema = this.schemas.get(key);
    if (!schema) throw new Error(`ParamStore: unknown key ${key}`);
    if (!this.validate(schema, value)) {
      console.warn(`ParamStore: rejected ${key}=${value} (out of range or wrong type)`);
      return;
    }
    this.values.set(key, value);
    this.writePersisted();
    for (const fn of this.subscribers) fn(key, value, "user");
  }

  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  notify(key: string, value: ParamValue): void {
    for (const fn of this.subscribers) fn(key, value, "modulator");
  }

  schemaFor(key: string): ParamSchema | undefined {
    return this.schemas.get(key);
  }

  reset(): void {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // localStorage unavailable; values still reset in-memory
    }
    this.persisted = {};
    const changed: Array<[string, ParamValue]> = [];
    for (const [key, schema] of this.schemas) {
      const oldValue = this.values.get(key);
      if (oldValue !== schema.default) {
        this.values.set(key, schema.default);
        changed.push([key, schema.default]);
      }
    }
    for (const [key, value] of changed) {
      for (const fn of this.subscribers) fn(key, value, "user");
    }
  }

  getAll(): Record<string, ParamValue> {
    return Object.fromEntries(this.values);
  }

  schemasInOrder(): ParamSchema[] {
    return Array.from(this.schemas.values());
  }

  private matchesKind(schema: ParamSchema, value: ParamValue): boolean {
    if (schema.kind === "boolean") return typeof value === "boolean";
    return typeof value === "number";
  }

  private validate(schema: ParamSchema, value: ParamValue): boolean {
    if (schema.kind === "boolean") return typeof value === "boolean";
    if (typeof value !== "number") return false;
    if (schema.kind === "discrete") return schema.options.includes(value);
    if (schema.kind === "color") return value >= 0 && value <= COLOR_MAX;
    return value >= schema.min && value <= schema.max;
  }

  private readPersisted(): Record<string, ParamValue> {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return typeof parsed === "object" && parsed !== null ? parsed : {};
    } catch {
      return {};
    }
  }

  private writePersisted(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.values)));
    } catch {
      // localStorage unavailable; no-op
    }
  }
}
