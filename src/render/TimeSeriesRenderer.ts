import { ColorRepresentation, Object3D, Vector3 } from "three";
import { layouts, type LineLayout } from "./lineLayouts";

export type { LineLayout, LineLayoutFn } from "./lineLayouts";

/**
 * Per-channel autogain config. The renderer keeps a running peak that
 * grows instantly with the latest sample and decays as exp(-dt/τ).
 *
 * Both fields are read every frame so the user can drive them from a live
 * param store. The algorithm assumes `source()` returns a ring buffer whose
 * last entry is the newest sample (matches RMS history).
 */
export interface AutoGainOptions {
  tauSecs: () => number;
  dtSecs: () => number;
}

export interface TimeSeriesRendererOptions {
  source: () => Float32Array;
  color?: ColorRepresentation;
  layout?: LineLayout;
  autoGain?: AutoGainOptions;
}

/**
 * Owns the shared per-frame loop: read source buffer, optionally autogain
 * it, dispatch the layout fn to map (i, n, value) → (x, y, z), and let the
 * subclass paint that into a Three primitive (line strip, instanced quads).
 *
 * Subclasses provide `object3d`, `allocate`, `writeOne`, `commit`, `dispose`.
 * The base never touches subclass fields from its constructor — subclasses
 * call `update()` themselves once their own state is initialized.
 */
export abstract class TimeSeriesRenderer {
  abstract readonly object3d: Object3D;
  protected source: () => Float32Array;
  protected layout: LineLayout;
  protected color: ColorRepresentation;
  protected lastLength = -1;

  private autoGain?: AutoGainOptions;
  private autoBuf?: Float32Array;
  private runningMax = 0;

  constructor(opts: TimeSeriesRendererOptions) {
    this.source = opts.source;
    this.layout = opts.layout ?? "linear";
    this.color = opts.color ?? 0xffffff;
    this.autoGain = opts.autoGain;
  }

  update(): void {
    const raw = this.source();
    const buf = this.autoGain ? this.applyAutoGain(raw) : raw;
    const n = buf.length;
    if (n !== this.lastLength) this.allocate(n);
    const layout = layouts[this.layout];
    for (let i = 0; i < n; i++) {
      this.writeOne(i, n, layout(i, n, buf[i]));
    }
    this.commit();
    this.lastLength = n;
  }

  abstract dispose(): void;
  protected abstract allocate(n: number): void;
  protected abstract writeOne(i: number, n: number, v: Vector3): void;
  protected abstract commit(): void;

  /**
   * Per-channel autogain: tracks a running peak, decays it by exp(-dt/τ),
   * normalizes to a parallel ring buffer. Older entries keep their
   * time-of-arrival normalization (so the line doesn't pump when the peak
   * shifts), only the newest sample uses the newest peak.
   *
   * Cold start (runningMax == 0) seeds the peak + autoBuf from the full
   * incoming buffer in one pass, so nothing slowly fills in over τ seconds.
   *
   * Divisor floor 1e-3 prevents divide-by-near-zero blowups during silence.
   */
  private applyAutoGain(raw: Float32Array): Float32Array {
    if (raw.length === 0 || !this.autoGain) return raw;
    if (!this.autoBuf || this.autoBuf.length !== raw.length) {
      this.autoBuf = new Float32Array(raw.length);
      this.runningMax = 0;
    }
    const auto = this.autoBuf;
    const eps = 1e-3;
    if (this.runningMax === 0) {
      let mx = 0;
      for (let i = 0; i < raw.length; i++) if (raw[i] > mx) mx = raw[i];
      this.runningMax = mx;
      const denom = Math.max(mx, eps);
      for (let i = 0; i < raw.length; i++) auto[i] = raw[i] / denom;
      return auto;
    }
    const retention = Math.exp(
      -this.autoGain.dtSecs() / this.autoGain.tauSecs(),
    );
    const latest = raw[raw.length - 1];
    this.runningMax = Math.max(latest, retention * this.runningMax);
    const denom = Math.max(this.runningMax, eps);
    auto.copyWithin(0, 1);
    auto[auto.length - 1] = latest / denom;
    return auto;
  }
}
