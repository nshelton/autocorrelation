import { Mesh, MeshBasicMaterial, PlaneGeometry } from "three";

export class DebugGrid {
  readonly object3d: Mesh;

  constructor() {
    const geometry = new PlaneGeometry(4, 4, 4, 4);
    const material = new MeshBasicMaterial({
      color: 0x666666,
      wireframe: true,
    });
    this.object3d = new Mesh(geometry, material);
  }

  dispose(): void {
    (this.object3d.geometry as PlaneGeometry).dispose();
    (this.object3d.material as MeshBasicMaterial).dispose();
    this.object3d.parent?.remove(this.object3d);
  }
}
