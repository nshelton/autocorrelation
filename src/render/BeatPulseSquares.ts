import {
  Group,
  Mesh,
  MeshBasicMaterial,
  Object3D,
  PlaneGeometry,
} from "three";

export interface BeatPulseSquaresOptions {
  /** Returns one saw value in [0, 1] per square (NaN → black). Length must
   * match the renderer's `count` (e.g., 4 for a 2×2 grid). Order maps to the
   * grid in row-major: index 0 = top-left, 1 = top-right, 2 = bottom-left,
   * 3 = bottom-right. */
  source: () => Float32Array;
  /** Number of squares; layout assumes a sqrt(count)×sqrt(count) grid. */
  count: number;
  /** Center of the whole grid in scene space. */
  centerX: number;
  centerY: number;
  /** Edge length of each square (in scene units). */
  cellSize: number;
  /** Gap between squares. */
  gap: number;
}

export class BeatPulseSquares {
  readonly object3d: Object3D;
  private source: () => Float32Array;
  private count: number;
  private materials: MeshBasicMaterial[] = [];
  private geometries: PlaneGeometry[] = [];

  constructor(opts: BeatPulseSquaresOptions) {
    this.source = opts.source;
    this.count = opts.count;
    this.object3d = new Group();

    const side = Math.round(Math.sqrt(opts.count));
    if (side * side !== opts.count) {
      throw new Error(`BeatPulseSquares: count must be a perfect square (got ${opts.count})`);
    }
    const stride = opts.cellSize + opts.gap;
    // Center the grid so the average of all square positions is (centerX, centerY).
    const half = (side - 1) / 2;
    for (let i = 0; i < opts.count; i++) {
      const row = Math.floor(i / side);
      const col = i % side;
      const x = opts.centerX + (col - half) * stride;
      // Row 0 is the TOP row visually, so flip Y.
      const y = opts.centerY + (half - row) * stride;
      const geom = new PlaneGeometry(opts.cellSize, opts.cellSize);
      const mat = new MeshBasicMaterial({ color: 0x000000 });
      const mesh = new Mesh(geom, mat);
      mesh.position.set(x, y, 0);
      this.geometries.push(geom);
      this.materials.push(mat);
      this.object3d.add(mesh);
    }
  }

  update(): void {
    const src = this.source();
    for (let i = 0; i < this.count; i++) {
      const raw = src[i];
      const v = Number.isNaN(raw) ? 0 : Math.max(0, Math.min(1, raw));
      this.materials[i].color.setRGB(v, v, v);
      // this.object3d.scale.set(v,v,v);
    }
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    this.object3d.parent?.remove(this.object3d);
  }
}
