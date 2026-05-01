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

export interface StaticBeatGridMarkersOptions {
  baseColor?: ColorRepresentation;
}

export class StaticBeatGridMarkers {
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
  private maxLines = 0;

  constructor(opts: StaticBeatGridMarkersOptions = {}) {
    const base = new Color(opts.baseColor ?? 0x66ccff);
    this.baseR = base.r;
    this.baseG = base.g;
    this.baseB = base.b;
    this.createLines(16);
  }

  update(beatGrid: Float32Array, teaSize: number): void {
    // console.log(beatGrid);
    const scale = (4 * beatGrid[0]) / teaSize;
    this.object3d.scale.set(scale, 0.2, 1);
  }

  dispose(): void {
    this.disposeLines();
    this.object3d.parent?.remove(this.object3d);
  }

  private createLines(lineCount: number): void {
    this.disposeLines();

    this.maxLines = Math.max(0, Math.floor(lineCount));
    const vertexCount = this.maxLines * 2;

    this.positions = new Float32Array(vertexCount * 3);
    this.colors = new Float32Array(vertexCount * 3);

    this.geometry = new BufferGeometry();
    this.positionAttribute = new BufferAttribute(this.positions, 3);
    this.colorAttribute = new BufferAttribute(this.colors, 3);
    this.positionAttribute.setUsage(DynamicDrawUsage);
    this.colorAttribute.setUsage(DynamicDrawUsage);
    this.geometry.setAttribute("position", this.positionAttribute);
    this.geometry.setAttribute("color", this.colorAttribute);

    this.material = new LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
    });

    this.segments = new LineSegments(this.geometry, this.material);
    this.object3d.add(this.segments);

    for (let i = 0; i < this.maxLines; i++) {
      this.writeLine(i, i);
    }

    this.commit();
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
    this.maxLines = 0;
  }

  private writeLine(lineIndex: number, x: number): void {
    const bottom = lineIndex * 2;
    const top = bottom + 1;
    this.writeVertex(bottom, x, 0);
    this.writeVertex(top, x, 1);
  }

  private writeVertex(vertexIndex: number, x: number, y: number): void {
    if (!this.positions || !this.colors) return;

    const off = vertexIndex * 3;
    this.positions[off] = x;
    this.positions[off + 1] = y;
    this.positions[off + 2] = 0;
    this.colors[off] = this.baseR;
    this.colors[off + 1] = this.baseG;
    this.colors[off + 2] = this.baseB;
  }

  private commit(): void {
    if (this.positionAttribute) this.positionAttribute.needsUpdate = true;
    if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
  }
}
