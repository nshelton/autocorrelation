import { ColorRepresentation, Object3D } from "three";

export type TimeSeriesScale = "linear" | "logx";

export interface TimeSeriesRendererOptions {
  source: () => Float32Array;
  color?: ColorRepresentation;
  colorByValue?: boolean;
  scale?: TimeSeriesScale;
}

/**
 * Owns the shared per-frame loop: read source buffer, hand each sample to
 * the subclass as (i, n, x, y) where x is mapped through the configured
 * x-scale and y = sample value. Position/scale via object3d to fit on screen.
 *
 * Subclasses provide `object3d`, `allocate`, `writeOne`, `commit`, `dispose`.
 * The base never touches subclass fields from its constructor — subclasses
 * call `update()` themselves once their own state is initialized.
 */
export abstract class TimeSeriesRenderer {
  abstract readonly object3d: Object3D;
  protected source: () => Float32Array;
  protected color: ColorRepresentation;
  protected colorByValue: boolean;
  protected scale: TimeSeriesScale;
  protected lastLength = -1;

  constructor(opts: TimeSeriesRendererOptions) {
    this.source = opts.source;
    this.color = opts.color ?? 0xffffff;
    this.colorByValue = opts.colorByValue ?? false;
    this.scale = opts.scale ?? "linear";
  }

  update(): void {
    const buf = this.source();
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

  protected brightnessForValue(value: number): number {
    if (!this.colorByValue) return 1;
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(1, Math.abs(value)));
  }

  private xForIndex(i: number, n: number): number {
    if (n <= 1) return 0;
    if (this.scale === "logx") return Math.log2(i + 1) / Math.log2(n);
    return i / (n - 1);
  }
}
