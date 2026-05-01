import {
  Color,
  ColorRepresentation,
  Group,
  Mesh,
  MeshBasicMaterial,
  Object3D,
  SphereGeometry,
} from "three";

export interface StaticBeatGridCirclesOptions {
  baseColor?: ColorRepresentation;
  radius?: number;
  segments?: number;
}

export class StaticBeatGridCircles {
  readonly object3d: Object3D = new Group();

  private geometry: SphereGeometry;
  private material: MeshBasicMaterial;
  private spheres: Mesh[] = [];

  constructor(opts: StaticBeatGridCirclesOptions = {}) {
    const color = new Color(opts.baseColor ?? 0x66ccff);
    this.geometry = new SphereGeometry(
      opts.radius ?? 0.02,
      opts.segments ?? 16,
      opts.segments ?? 16,
    );
    this.material = new MeshBasicMaterial({ color });

    for (let i = 0; i < 16; i++) {
      const sphere = new Mesh(this.geometry, this.material);
      sphere.position.set(i, 0.5, 0);
      this.object3d.add(sphere);
      this.spheres.push(sphere);
    }
  }

  update(beatGrid: Float32Array, teaSize: number): void {
    const scale = (4 * beatGrid[0]) / teaSize;
    this.object3d.scale.set(scale, scale, scale);
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.object3d.parent?.remove(this.object3d);
  }
}
