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
}

export class BeatPulseSquares {
  readonly object3d: Object3D;
  private source: () => Float32Array;
  private count: number;
  private materials: MeshBasicMaterial[] = [];
  private geometries: PlaneGeometry[] = [];

  constructor(opts: BeatPulseSquaresOptions) {
    this.source = opts.source;
    this.count = 4;
    this.object3d = new Group();

    for (let i = 0; i < this.count; i++) {
      const x = i * 0.15;
      const y = 0;

      const geom = new PlaneGeometry(0.1, 0.5);
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
