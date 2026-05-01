import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  ColorRepresentation,
  DynamicDrawUsage,
  Group,
  LineBasicMaterial,
  LineSegments,
  Object3D,
} from "three";

export interface PeakMarkersOptions {
  baseColor?: ColorRepresentation;
}

export class PeakMarkers {
  readonly object3d: Object3D = new Group();

  private baseR: number;
  private baseG: number;
  private baseB: number;

  private segments?: LineSegments;
  private geometry?: BufferGeometry;
  private material?: LineBasicMaterial;
  private positions?: Float32Array;
  private colors?: Float32Array;
  private positionAttribute?: BufferAttribute;
  private colorAttribute?: BufferAttribute;
  private maxPeaks = 0;

  constructor(opts: PeakMarkersOptions = {}) {
    const base = new Color(opts.baseColor ?? 0xffff66);
    this.baseR = base.r;
    this.baseG = base.g;
    this.baseB = base.b;
  }

  update(candidates: Float32Array, lagResolution: number): void {
    if (candidates.length === 0) return;

    const peakCount = Math.floor(candidates.length / 3);
    if (peakCount !== this.maxPeaks) this.createLines(peakCount);
    if (!this.positionAttribute || !this.colorAttribute) return;

    const denom = peakCount > 1 ? peakCount - 1 : 1;
    for (let i = 0; i < peakCount; i++) {
      const lag = candidates[3 * i + 0];
      const mag = candidates[3 * i + 1];
      const curvature = candidates[3 * i + 2];

      const id1 = i * 2;
      const id2 = i * 2 + 1;
      if (Number.isNaN(lag)) {
        this.writeVertex(id1, 0, 0, 0, 0);
        this.writeVertex(id2, 0, 0, 0, 0);
        continue;
      }

      const x = lag / lagResolution;
      const brightness = 1.0 - 0.75 * (i / denom);
      this.writeVertex(id1, x, 1, 0, brightness);
      this.writeVertex(id2, x, 0, 0, brightness);
    }

    this.positionAttribute.needsUpdate = true;
    this.colorAttribute.needsUpdate = true;
  }

  dispose(): void {
    this.disposeLines();
    this.object3d.parent?.remove(this.object3d);
  }

  private createLines(peakCount: number): void {
    this.disposeLines();
    this.maxPeaks = peakCount;

    const vertexCount = peakCount * 2;
    this.positions = new Float32Array(vertexCount * 3);
    this.colors = new Float32Array(vertexCount * 3);

    this.positionAttribute = new BufferAttribute(this.positions, 3);
    this.positionAttribute.setUsage(DynamicDrawUsage);
    this.colorAttribute = new BufferAttribute(this.colors, 3);
    this.colorAttribute.setUsage(DynamicDrawUsage);

    this.geometry = new BufferGeometry();
    this.geometry.setAttribute("position", this.positionAttribute);
    this.geometry.setAttribute("color", this.colorAttribute);

    this.material = new LineBasicMaterial({
      vertexColors: true,
      linewidth: 10,
      transparent: true,
      blending: AdditiveBlending,
      opacity: 1,
      depthTest: false,
      depthWrite: false,
    });
    this.segments = new LineSegments(this.geometry, this.material);
    this.segments.frustumCulled = false;
    this.object3d.add(this.segments);
  }

  private disposeLines(): void {
    if (this.segments) this.object3d.remove(this.segments);
    this.geometry?.dispose();
    this.material?.dispose();
    this.segments = undefined;
    this.geometry = undefined;
    this.material = undefined;
    this.positions = undefined;
    this.colors = undefined;
    this.positionAttribute = undefined;
    this.colorAttribute = undefined;
    this.maxPeaks = 0;
  }

  private writeVertex(
    vertexIndex: number,
    x: number,
    y: number,
    z: number,
    brightness: number,
  ): void {
    if (!this.positions || !this.colors) return;

    const off = vertexIndex * 3;
    this.positions[off] = x;
    this.positions[off + 1] = y;
    this.positions[off + 2] = z;
    this.colors[off] = this.baseR * brightness;
    this.colors[off + 1] = this.baseG * brightness;
    this.colors[off + 2] = this.baseB * brightness;
  }
}
