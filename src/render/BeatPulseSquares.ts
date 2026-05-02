import { Group, Mesh, MeshBasicMaterial, Object3D, PlaneGeometry } from "three";

export class BeatPulseSquares {
  readonly object3d: Object3D;
  private count: number;
  private materials: MeshBasicMaterial[] = [];
  private geometries: PlaneGeometry[] = [];
  private meshObjects: Mesh[] = [];

  constructor() {
    this.count = 4;
    this.object3d = new Group();

    for (let i = 0; i < this.count; i++) {
      const x = i - 1.5;
      const y = 0;

      const geom = new PlaneGeometry(1, 0.1);
      const mat = new MeshBasicMaterial({ color: 0x0ff000 });
      const mesh = new Mesh(geom, mat);
      mesh.position.set(x, y - 1.1, 0);
      this.geometries.push(geom);
      this.materials.push(mat);
      this.meshObjects.push(mesh);
      this.object3d.add(mesh);
    }
  }

  update(beatData: Float32Array): void {
    for (let i = 0; i < this.count; i++) {
      const raw = beatData[i];
      const v = Number.isNaN(raw) ? 0 : Math.max(0, Math.min(1, raw));
      const e = Math.pow(v, 0.5);
      this.materials[i].color.setRGB(e, e, e);
      this.meshObjects[i].scale.set(e, 1, e);
    }
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    this.object3d.parent?.remove(this.object3d);
  }
}
