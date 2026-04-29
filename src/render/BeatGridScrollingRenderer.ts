import {
  BufferAttribute,
  BufferGeometry,
  ColorRepresentation,
  DynamicDrawUsage,
  LineBasicMaterial,
  LineSegments,
  Object3D,
} from "three";

export interface BeatGridScrollingRendererOptions {
  /** Returns `[period, phase, score, ...]` (NaN-padded if no fit). All in
   * hop units; `phase` is "how many hops ago the most-recent beat fell".
   * The renderer maps each beat to a buffer index `(domain - 1) - phase -
   * k·period` for k = 0, 1, …, scrolling left as phase advances each hop. */
  source: () => Float32Array;
  /** Cap on the number of beat lines the renderer can show. Sets the
   * preallocated vertex-buffer size (`2 * maxLines`). */
  maxLines: number;
  /** Length of the scrolling buffer (= rmsHistoryLen). The newest sample sits
   * at index `domain - 1`; oldest at 0. */
  domain: number;
  yTop: number;
  yBottom: number;
  /** Maps a buffer index (possibly fractional) to an x-coordinate. Pass the
   * same mapping the underlying chart uses so the grid stays pixel-aligned
   * with the rms-history line it overlays. */
  xForIndex: (index: number, domain: number) => number;
  color?: ColorRepresentation;
}

export class BeatGridScrollingRenderer {
  readonly object3d: Object3D;
  private source: () => Float32Array;
  private maxLines: number;
  private domain: number;
  private yTop: number;
  private yBottom: number;
  private xForIndex: (index: number, domain: number) => number;
  private positions: Float32Array;
  private positionAttribute: BufferAttribute;

  constructor(opts: BeatGridScrollingRendererOptions) {
    this.source = opts.source;
    this.maxLines = opts.maxLines;
    this.domain = opts.domain;
    this.yTop = opts.yTop;
    this.yBottom = opts.yBottom;
    this.xForIndex = opts.xForIndex;

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
    const phase = src[1];
    if (Number.isNaN(period) || Number.isNaN(phase) || period <= 0) {
      for (let i = 0; i < this.maxLines * 2; i++) {
        this.writeVertex(i, 0, this.yTop, 0);
      }
      this.positionAttribute.needsUpdate = true;
      return;
    }
    const newest = this.domain - 1;
    for (let k = 0; k < this.maxLines; k++) {
      const idx = newest - phase - k * period;
      const top = k * 2;
      const bot = k * 2 + 1;
      if (idx < 0 || idx > newest) {
        this.writeVertex(top, 0, this.yTop, 0);
        this.writeVertex(bot, 0, this.yTop, 0);
        continue;
      }
      const x = this.xForIndex(idx, this.domain);
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
