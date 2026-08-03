import type { ParamStore } from "./ParamStore";

// Glides numeric params toward a preset's values instead of snapping to them.
// One tween at a time, shared by module and system presets: starting a new one
// re-reads the CURRENT (possibly mid-flight) values as its start, so rapid
// preset switching chases smoothly rather than jumping back to a stale origin.
//
// Module singleton ticked from App's RAF loop, same as shTween — PresetStore
// reaches it directly rather than threading it through every call site.
type Track = { key: string; from: number; to: number; color: boolean };

class PresetTween {
  private tracks: Track[] = [];
  private elapsed = 0;
  private duration = 0;

  // `targets` holds only tweenable params (continuous + color). Keys already at
  // their target are dropped, which is most of them on a typical switch — that
  // is what keeps a 150-param system preset from writing the whole store every
  // frame for the length of the tween.
  start(store: ParamStore, targets: Map<string, number>, duration: number): void {
    this.tracks = [];
    for (const [key, to] of targets) {
      const schema = store.schemaFor(key);
      if (!schema) continue;
      const from = store.get(key);
      if (typeof from !== "number" || from === to) continue;
      this.tracks.push({ key, from, to, color: schema.kind === "color" });
    }
    this.duration = duration;
    this.elapsed = 0;
    if (duration <= 0) this.settle(store);
  }

  tick(dt: number, store: ParamStore): void {
    if (this.tracks.length === 0) return;
    this.elapsed += dt;
    const t = this.elapsed / this.duration;
    if (t >= 1) {
      this.settle(store);
      return;
    }
    const e = ease(t);
    for (const tr of this.tracks) {
      store.set(tr.key, tr.color ? mixColor(tr.from, tr.to, e) : tr.from + (tr.to - tr.from) * e);
    }
  }

  get active(): boolean {
    return this.tracks.length > 0;
  }

  cancel(): void {
    this.tracks = [];
  }

  // Land exactly on the targets — the eased walk never quite reaches them, and
  // a preset must end bit-identical or it reads as permanently "modified".
  private settle(store: ParamStore): void {
    for (const tr of this.tracks) store.set(tr.key, tr.to);
    this.tracks = [];
  }
}

// Cubic ease-in-out: no velocity discontinuity at either end.
function ease(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// Packed 0xRRGGBB lerped per channel. Interpolating the packed integer instead
// would run through unrelated hues on the way.
function mixColor(from: number, to: number, t: number): number {
  const r = Math.round((((from >> 16) & 0xff) * (1 - t)) + (((to >> 16) & 0xff) * t));
  const g = Math.round((((from >> 8) & 0xff) * (1 - t)) + (((to >> 8) & 0xff) * t));
  const b = Math.round(((from & 0xff) * (1 - t)) + ((to & 0xff) * t));
  return (r << 16) | (g << 8) | b;
}

export const presetTween = new PresetTween();
