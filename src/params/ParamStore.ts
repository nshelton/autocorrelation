export type ParamValue = number;

export type ParamSchema = {
  key: string;
  label: string;
  default: ParamValue;
  reconfig: boolean;
} & (
  | { kind: "discrete"; options: ParamValue[] }
  | { kind: "continuous"; min: number; max: number; step: number }
);

type Subscriber = (key: string, value: ParamValue) => void;

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
    const initial = this.persisted[schema.key];
    this.values.set(schema.key, initial !== undefined ? initial : schema.default);
  }

  get(key: string): ParamValue {
    if (!this.values.has(key)) throw new Error(`ParamStore: unknown key ${key}`);
    return this.values.get(key)!;
  }

  set(key: string, value: ParamValue): void {
    const schema = this.schemas.get(key);
    if (!schema) throw new Error(`ParamStore: unknown key ${key}`);
    if (!this.validate(schema, value)) {
      console.warn(`ParamStore: rejected ${key}=${value} (out of range)`);
      return;
    }
    this.values.set(key, value);
    this.writePersisted();
    for (const fn of this.subscribers) fn(key, value);
  }

  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
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
      for (const fn of this.subscribers) fn(key, value);
    }
  }

  getAll(): Record<string, ParamValue> {
    return Object.fromEntries(this.values);
  }

  schemasInOrder(): ParamSchema[] {
    return Array.from(this.schemas.values());
  }

  private validate(schema: ParamSchema, value: ParamValue): boolean {
    if (schema.kind === "discrete") return schema.options.includes(value);
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
