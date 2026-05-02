import {
  AdditiveBlending,
  Color,
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
// Per-instance scale.y = sample value → bar grows up to that value.
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
  private instanceColor = new Color();

  constructor(opts: TimeSeriesRendererOptions) {
    super(opts);
    this.material = new MeshBasicMaterial({
      color: this.color,
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
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

  protected writeOne(i: number, n: number, x: number, y: number): void {
    const w = n <= 1 ? 1 : 1 / n;
    this.pos.set(x, 0, 0);
    this.scl.set(w, y, 1);
    this.mat.compose(this.pos, this.quat, this.scl);
    this.mesh.setMatrixAt(i, this.mat);

    const brightness = this.brightnessForValue(y);
    this.instanceColor.setRGB(brightness, brightness, brightness);
    this.mesh.setColorAt(i, this.instanceColor);
  }

  protected commit(): void {
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}
