import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  DynamicDrawUsage,
  Group,
  Line,
  LineBasicMaterial,
  Vector3,
} from "three";
import {
  TimeSeriesRenderer,
  type TimeSeriesRendererOptions,
} from "./TimeSeriesRenderer";

export class TimeSeriesLineRenderer extends TimeSeriesRenderer {
  // Stable wrapper. Inner Line is built lazily on first non-empty allocate
  // and rebuilt on size change — three's WebGPU renderer caches a GPU
  // buffer per attribute reference and doesn't always re-upload after a
  // setAttribute swap, so we replace the whole Line instead.
  readonly object3d = new Group();
  private line?: Line;
  private geometry?: BufferGeometry;
  private material?: LineBasicMaterial;
  private positions?: Float32Array;
  private positionAttribute?: BufferAttribute;

  constructor(opts: TimeSeriesRendererOptions) {
    super(opts);
    this.update();
  }

  // Defer everything (incl. allocate) until source actually has samples;
  // a 0-byte initial GPU buffer breaks subsequent re-uploads.
  update(): void {
    if (this.source().length === 0) return;
    super.update();
  }

  dispose(): void {
    this.geometry?.dispose();
    this.material?.dispose();
    this.object3d.parent?.remove(this.object3d);
  }

  protected allocate(n: number): void {
    if (this.line) {
      this.geometry?.dispose();
      this.material?.dispose();
      this.object3d.remove(this.line);
    }
    this.positions = new Float32Array(n * 3);
    this.positionAttribute = new BufferAttribute(this.positions, 3);
    this.positionAttribute.setUsage(DynamicDrawUsage);
    this.geometry = new BufferGeometry();
    this.geometry.setAttribute("position", this.positionAttribute);
    this.material = new LineBasicMaterial({
      color: this.color,
      blending: AdditiveBlending,
      transparent: true,
      opacity: 1,
      depthWrite: false,
    });
    this.line = new Line(this.geometry, this.material);
    this.line.frustumCulled = false;
    this.object3d.add(this.line);
  }

  protected writeOne(i: number, _n: number, v: Vector3): void {
    const p = this.positions!;
    p[i * 3] = v.x;
    p[i * 3 + 1] = v.y;
    p[i * 3 + 2] = v.z;
  }

  protected commit(): void {
    if (this.positionAttribute) this.positionAttribute.needsUpdate = true;
  }
}
