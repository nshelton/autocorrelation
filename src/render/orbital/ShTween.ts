import type { ParamStore } from "../../params/ParamStore";

// Time-based lerp of a group of param-store keys toward target values. Driven
// by the App RAF loop via tick(dt); kicked off by the "Randomize SH" button.
// It's a module singleton (not component state) on purpose: the orbitalCloud.*
// SH keys are read by BOTH the cloud and the volume renderers, so the tween
// must keep running while you toggle between those views — neither component's
// update() can be relied on to tick it.
class ParamTween {
  private keys: string[] = [];
  private from: number[] = [];
  private to: number[] = [];
  private elapsed = 0;
  private duration = 0;
  private active = false;

  // Lerp `keys` from their current store values to `targets` over `secs`.
  // secs <= 0 snaps immediately. A new start() supersedes any in-flight tween
  // (re-reads `from` from the live values, so it eases out of mid-flight).
  start(store: ParamStore, keys: string[], targets: number[], secs: number): void {
    if (secs <= 0) {
      for (let i = 0; i < keys.length; i++) store.set(keys[i], targets[i]);
      this.active = false;
      return;
    }
    this.keys = keys;
    this.from = keys.map((k) => store.get(k) as number);
    this.to = targets.slice();
    this.elapsed = 0;
    this.duration = secs;
    this.active = true;
  }

  tick(dt: number, store: ParamStore): void {
    if (!this.active) return;
    this.elapsed += dt;
    const raw = Math.min(1, this.elapsed / this.duration);
    // smoothstep ease — zero velocity at both endpoints so the morph settles
    // gently instead of stopping dead.
    const t = raw * raw * (3 - 2 * raw);
    for (let i = 0; i < this.keys.length; i++) {
      store.set(this.keys[i], this.from[i] + (this.to[i] - this.from[i]) * t);
    }
    if (raw >= 1) this.active = false;
  }
}

export const shTween = new ParamTween();
