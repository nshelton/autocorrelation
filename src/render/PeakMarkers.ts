import {
  BufferAttribute,
  BufferGeometry,
  Color,
  ColorRepresentation,
  DynamicDrawUsage,
  LineBasicMaterial,
  LineSegments,
  Object3D,
} from "three";

export interface PeakMarkersOptions {
  source: () => Float32Array;
  maxPeaks: number;
  lagDomain: number;
  yCenter: number;
  ySpan: number;
  xForLag: (lag: number, lagDomain: number) => number;
  baseColor?: ColorRepresentation;
}

export class PeakMarkers {
  readonly object3d: Object3D;
  private source: () => Float32Array;
  private maxPeaks: number;
  private lagDomain: number;
  private yCenter: number;
  private ySpan: number;
  private xForLag: (lag: number, lagDomain: number) => number;
  private baseR: number;
  private baseG: number;
  private baseB: number;
  private positions: Float32Array;
  private colors: Float32Array;
  private positionAttribute: BufferAttribute;
  private colorAttribute: BufferAttribute;

  constructor(opts: PeakMarkersOptions) {
    this.source = opts.source;
    this.maxPeaks = opts.maxPeaks;
    this.lagDomain = opts.lagDomain;
    this.yCenter = opts.yCenter;
    this.ySpan = opts.ySpan;
    this.xForLag = opts.xForLag;

    const base = new Color(opts.baseColor ?? 0xffff66);
    this.baseR = base.r;
    this.baseG = base.g;
    this.baseB = base.b;

    const vertexCount = 2 * this.maxPeaks;
    this.positions = new Float32Array(vertexCount * 3);
    this.colors = new Float32Array(vertexCount * 3);

    this.positionAttribute = new BufferAttribute(this.positions, 3);
    this.positionAttribute.setUsage(DynamicDrawUsage);
    this.colorAttribute = new BufferAttribute(this.colors, 3);
    this.colorAttribute.setUsage(DynamicDrawUsage);

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", this.positionAttribute);
    geometry.setAttribute("color", this.colorAttribute);

    const material = new LineBasicMaterial({ vertexColors: true });
    this.object3d = new LineSegments(geometry, material);
  }

  update(): void {
    const src = this.source();
    const denom = this.maxPeaks > 1 ? this.maxPeaks - 1 : 1;
    for (let i = 0; i < this.maxPeaks; i++) {
      const lag = src[2 * i];
      const top = i * 2;
      const bot = i * 2 + 1;
      if (Number.isNaN(lag)) {
        this.writeVertex(top, 0, this.yCenter, 0, 0);
        this.writeVertex(bot, 0, this.yCenter, 0, 0);
        continue;
      }
      const x = this.xForLag(lag, this.lagDomain);
      const brightness = 1.0 - 0.75 * (i / denom);
      this.writeVertex(top, x, this.yCenter + this.ySpan, 0, brightness);
      this.writeVertex(bot, x, this.yCenter - this.ySpan, 0, brightness);
    }
    this.positionAttribute.needsUpdate = true;
    this.colorAttribute.needsUpdate = true;
  }

  dispose(): void {
    const seg = this.object3d as LineSegments;
    (seg.geometry as BufferGeometry).dispose();
    (seg.material as LineBasicMaterial).dispose();
    seg.parent?.remove(seg);
  }

  private writeVertex(vertexIndex: number, x: number, y: number, z: number, brightness: number): void {
    const off = vertexIndex * 3;
    this.positions[off] = x;
    this.positions[off + 1] = y;
    this.positions[off + 2] = z;
    this.colors[off] = this.baseR * brightness;
    this.colors[off + 1] = this.baseG * brightness;
    this.colors[off + 2] = this.baseB * brightness;
  }
}
