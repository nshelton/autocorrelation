import {
  BufferAttribute,
  BufferGeometry,
  ColorRepresentation,
  DynamicDrawUsage,
  LineBasicMaterial,
  LineSegments,
  Object3D,
} from "three";

export interface BeatGridRendererOptions {
  /** Returns `[period, score]` (NaN-padded if no fit). Period is in lag units;
   * the renderer draws a vertical segment at lag = k·period for k = 1, 2, …
   * up to `maxLines`, while period × k stays inside `lagDomain`. */
  source: () => Float32Array;
  /** Cap on the number of grid segments the renderer can show. Sets the
   * preallocated vertex-buffer size (`2 * maxLines` vertices). Pick a value
   * that covers the densest expected grid (`ceil(lagDomain / minPeriod)`). */
  maxLines: number;
  lagDomain: number;
  /** Maps a lag (possibly fractional) to an x-coordinate. Pass the same
   * mapping the underlying chart uses so grid lines stay pixel-aligned with
   * the autocorrelation peaks they describe. */
  xForLag: (lag: number, lagDomain: number) => number;
  color?: ColorRepresentation;
}

export class BeatGridRenderer {
  readonly object3d: Object3D;
  private source: () => Float32Array;
  private maxLines: number;
  private lagDomain: number;
  private yTop: number;
  private yBottom: number;
  private xForLag: (lag: number, lagDomain: number) => number;
  private positions: Float32Array;
  private positionAttribute: BufferAttribute;

  constructor(opts: BeatGridRendererOptions) {
    this.source = opts.source;
    this.maxLines = opts.maxLines;
    this.lagDomain = opts.lagDomain;
    this.yTop = -0.6;
    this.yBottom = -0.8;
    this.xForLag = opts.xForLag;

    const vertexCount = 2 * this.maxLines;
    this.positions = new Float32Array(vertexCount * 3);

    this.positionAttribute = new BufferAttribute(this.positions, 3);
    this.positionAttribute.setUsage(DynamicDrawUsage);

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", this.positionAttribute);

    const material = new LineBasicMaterial({ color: opts.color ?? 0xffffff });
    this.object3d = new LineSegments(geometry, material);
  }

  update(): void {
    const src = this.source();
    const period = src[0];
    if (Number.isNaN(period) || period <= 0) {
      // No fit — collapse every segment to a single point so it draws nothing.
      // (LineSegments has a fixed vertex contract; we can't skip indices.)
      for (let i = 0; i < this.maxLines * 2; i++) {
        this.writeVertex(i, 0, this.yTop, 0);
      }
      this.positionAttribute.needsUpdate = true;
      return;
    }
    for (let k = 1; k <= this.maxLines; k++) {
      const lag = k * period;
      const top = (k - 1) * 2;
      const bot = (k - 1) * 2 + 1;
      // > lagDomain - 1 to match the inclusive upper edge of the chart line's
      // domain (xForLag maps i = lagDomain - 1 to x = 1.0).
      if (lag > this.lagDomain - 1) {
        this.writeVertex(top, 0, this.yTop, 0);
        this.writeVertex(bot, 0, this.yTop, 0);
        continue;
      }
      const x = this.xForLag(lag, this.lagDomain);
      this.writeVertex(top, x, this.yTop, 0);
      this.writeVertex(bot, x, this.yBottom, 0);
    }
    this.positionAttribute.needsUpdate = true;
  }

  dispose(): void {
    const seg = this.object3d as LineSegments;
    (seg.geometry as BufferGeometry).dispose();
    (seg.material as LineBasicMaterial).dispose();
    seg.parent?.remove(seg);
  }

  private writeVertex(vertexIndex: number, x: number, y: number, z: number): void {
    const off = vertexIndex * 3;
    this.positions[off] = x;
    this.positions[off + 1] = y;
    this.positions[off + 2] = z;
  }
}
