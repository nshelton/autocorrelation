import { ColorRepresentation, Object3D } from "three";

export type TimeSeriesScale = "linear" | "logx";

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
  autoGain?: AutoGainOptions;
  scale?: TimeSeriesScale;
}

/**
 * Owns the shared per-frame loop: read source buffer, optionally autogain
 * it, hand each sample to the subclass as (i, n, x, y) where x is mapped
 * through the configured x-scale and y = sample value. Position/scale via
 * object3d to fit on screen.
 *
 * Subclasses provide `object3d`, `allocate`, `writeOne`, `commit`, `dispose`.
 * The base never touches subclass fields from its constructor — subclasses
 * call `update()` themselves once their own state is initialized.
 */
export abstract class TimeSeriesRenderer {
  abstract readonly object3d: Object3D;
  protected source: () => Float32Array;
  protected color: ColorRepresentation;
  protected scale: TimeSeriesScale;
  protected lastLength = -1;

  private autoGain?: AutoGainOptions;
  private autoBuf?: Float32Array;
  private lastRaw?: Float32Array;
  private runningMax = 0;

  constructor(opts: TimeSeriesRendererOptions) {
    this.source = opts.source;
    this.color = opts.color ?? 0xffffff;
    this.scale = opts.scale ?? "linear";
    this.autoGain = opts.autoGain;
  }

  update(): void {
    const raw = this.source();
    const buf = this.autoGain ? this.applyAutoGain(raw) : raw;
    const n = buf.length;
    if (n !== this.lastLength) this.allocate(n);
    for (let i = 0; i < n; i++) {
      this.writeOne(i, n, this.xForIndex(i, n), buf[i]);
    }
    this.commit();
    this.lastLength = n;
  }

  abstract dispose(): void;
  protected abstract allocate(n: number): void;
  protected abstract writeOne(i: number, n: number, x: number, y: number): void;
  protected abstract commit(): void;

  private xForIndex(i: number, n: number): number {
    if (n <= 1) return 0;
    if (this.scale === "logx") return Math.log2(i + 1) / Math.log2(n);
    return i / (n - 1);
  }

  /**
   * Per-channel autogain. Each auto[i] holds the normalization that was
   * applied at the time raw[i] arrived — older entries keep their original
   * denom, so the line doesn't pump when the running peak shifts. Only the
   * newest sample uses the latest denom.
   *
   * Driven by raw advancement (detected via reference change + adjacent-
   * sample match), not by RAF rate: when raw doesn't change, auto doesn't
   * either, keeping scroll in lockstep with the un-gained renderer.
   *
   * Divisor floor 1e-3 prevents divide-by-near-zero blowups during silence.
   */
  private applyAutoGain(raw: Float32Array): Float32Array {
    if (raw.length === 0 || !this.autoGain) return raw;
    const eps = 1e-3;

    // Cold start or rmsHistoryLen reconfigure — full re-normalize. Seeds
    // runningMax from the entire buffer so we don't divide by a tiny
    // just-arrived sample on the first frame.
    if (!this.autoBuf || this.autoBuf.length !== raw.length) {
      this.autoBuf = new Float32Array(raw.length);
      let mx = 0;
      for (let i = 0; i < raw.length; i++) if (raw[i] > mx) mx = raw[i];
      this.runningMax = mx;
      const denom = Math.max(mx, eps);
      for (let i = 0; i < raw.length; i++) this.autoBuf[i] = raw[i] / denom;
      this.lastRaw = raw;
      return this.autoBuf;
    }

    const auto = this.autoBuf;

    // Same array reference as last call: no features arrived since.
    if (raw === this.lastRaw) return auto;

    // The worklet posts freshly shifted buffers. Detect how many samples the
    // ring advanced so already-normalized values stay attached to their time
    // slots even if RAF misses/batches one or more feature messages.
    const n = raw.length;
    const advanceBy = this.findAdvanceBy(raw, this.lastRaw);
    const retention = Math.exp(
      -this.autoGain.dtSecs() / this.autoGain.tauSecs(),
    );

    const appendNormalized = (i: number) => {
      const latest = raw[i];
      this.runningMax = Math.max(latest, retention * this.runningMax);
      auto[i] = latest / Math.max(this.runningMax, eps);
    };

    if (advanceBy > 0) {
      auto.copyWithin(0, advanceBy);
      for (let i = n - advanceBy; i < n; i++) appendNormalized(i);
    } else {
      // Non-adjacent jump/gap: preserve the visible normalized history instead
      // of re-normalizing the whole buffer, which causes the popping this
      // renderer-side autogain is designed to avoid.
      auto.copyWithin(0, 1);
      appendNormalized(n - 1);
    }

    this.lastRaw = raw;
    return auto;
  }

  private findAdvanceBy(
    raw: Float32Array,
    lastRaw: Float32Array | undefined,
  ): number {
    if (!lastRaw || lastRaw.length !== raw.length) return 0;
    const n = raw.length;

    for (let shift = 1; shift < n; shift++) {
      let matches = true;
      for (let i = 0; i < n - shift; i++) {
        if (raw[i] !== lastRaw[i + shift]) {
          matches = false;
          break;
        }
      }
      if (matches) return shift;
    }

    return 0;
  }
}
