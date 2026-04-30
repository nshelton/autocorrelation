import {
  Group,
  InstancedMesh,
  Matrix4,
  MeshBasicMaterial,
  PlaneGeometry,
  Quaternion,
  Vector3,
} from "three";
import {
  TimeSeriesRenderer,
  type TimeSeriesRendererOptions,
} from "./TimeSeriesRenderer";

// Unit quad with its base at y=0, top at y=1, centered horizontally on x=0.
// Per-instance scale.y = layout's y output → bar grows up to that value.
const unitQuad = (): PlaneGeometry => {
  const g = new PlaneGeometry(1, 1);
  g.translate(0, 0.5, 0);
  return g;
};

export class TimeSeriesBarRenderer extends TimeSeriesRenderer {
  // Stable wrapper. InstancedMesh count is fixed at construction, so on
  // length change we swap the inner mesh; callers keep a valid object3d.
  readonly object3d = new Group();
  private mesh!: InstancedMesh;
  private material!: MeshBasicMaterial;
  private mat = new Matrix4();
  private pos = new Vector3();
  private scl = new Vector3();
  private quat = new Quaternion();

  constructor(opts: TimeSeriesRendererOptions) {
    super(opts);
    this.material = new MeshBasicMaterial({
      color: this.color,
      transparent: true,
      depthWrite: false,
    });
    this.update();
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.mesh.dispose();
    this.object3d.parent?.remove(this.object3d);
  }

  protected allocate(n: number): void {
    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.mesh.dispose();
      this.object3d.remove(this.mesh);
    }
    this.mesh = new InstancedMesh(unitQuad(), this.material, Math.max(n, 1));
    this.mesh.frustumCulled = false;
    this.mesh.count = n;
    this.object3d.add(this.mesh);
  }

  protected writeOne(i: number, n: number, v: Vector3): void {
    const w = n <= 1 ? 1 : 1 / n;
    this.pos.set(v.x, 0, v.z);
    this.scl.set(w, v.y, 1);
    this.mat.compose(this.pos, this.quat, this.scl);
    this.mesh.setMatrixAt(i, this.mat);
  }

  protected commit(): void {
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
