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

export interface BeatGridMarkersOptions {
  baseColor?: ColorRepresentation;
}

export class BeatGridMarkers {
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

  constructor(opts: BeatGridMarkersOptions = {}) {
    const base = new Color(opts.baseColor ?? 0x66ccff);
    this.baseR = base.r;
    this.baseG = base.g;
    this.baseB = base.b;
  }

  /**
   * Draws a scrolling beat grid over a normalized [0, 1] x-domain. `beatGrid`
   * is `[period, phase, score]` from the DSP in hop/rms-history units, where
   * phase is how many samples ago the most recent beat landed.
   */
  update(beatGrid: Float32Array, rmsHistoryLength: number): void {
    if (beatGrid.length < 2 || rmsHistoryLength <= 1) return;

    const period = beatGrid[0];
    const phase = beatGrid[1];
    const score = beatGrid[2] ?? 1;
    const lineCount = this.requiredLineCount(period, rmsHistoryLength);

    if (lineCount > this.maxLines) this.createLines(lineCount);
    if (!this.positionAttribute || !this.colorAttribute) return;

    if (
      lineCount === 0 ||
      Number.isNaN(period) ||
      Number.isNaN(phase) ||
      period <= 0
    ) {
      this.clearLines();
      this.commit();
      return;
    }

    const newest = rmsHistoryLength - 1;
    // const brightness = Math.max(0.15, Math.min(1, Number.isNaN(score) ? 1 : score));
    const brightness = 1;
    let visible = 0;

    for (let k = 0; k < lineCount; k++) {
      const idx = newest - phase - k * period;
      if (idx < 0 || idx > newest) continue;

      const x = idx / newest;
      this.writeLine(visible, x, brightness);
      visible++;
    }

    for (let i = visible; i < this.maxLines; i++) this.hideLine(i);
    this.commit();
  }

  dispose(): void {
    this.disposeLines();
    this.object3d.parent?.remove(this.object3d);
  }

  private requiredLineCount(period: number, rmsHistoryLength: number): number {
    if (Number.isNaN(period) || period <= 0) return this.maxLines;
    return Math.ceil(rmsHistoryLength / period) + 1;
  }

  private createLines(lineCount: number): void {
    this.disposeLines();
    this.maxLines = lineCount;

    const vertexCount = lineCount * 2;
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
    this.maxLines = 0;
  }

  private clearLines(): void {
    for (let i = 0; i < this.maxLines; i++) this.hideLine(i);
  }

  private writeLine(lineIndex: number, x: number, brightness: number): void {
    const top = lineIndex * 2;
    const bottom = top + 1;
    this.writeVertex(top, x, 1, brightness);
    this.writeVertex(bottom, x, 0, brightness);
  }

  private hideLine(lineIndex: number): void {
    const top = lineIndex * 2;
    const bottom = top + 1;
    this.writeVertex(top, 0, 0, 0);
    this.writeVertex(bottom, 0, 0, 0);
  }

  private writeVertex(
    vertexIndex: number,
    x: number,
    y: number,
    brightness: number,
  ): void {
    if (!this.positions || !this.colors) return;

    const off = vertexIndex * 3;
    this.positions[off] = x;
    this.positions[off + 1] = y;
    this.positions[off + 2] = 0;
    this.colors[off] = this.baseR * brightness;
    this.colors[off + 1] = this.baseG * brightness;
    this.colors[off + 2] = this.baseB * brightness;
  }

  private commit(): void {
    if (this.positionAttribute) this.positionAttribute.needsUpdate = true;
    if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
  }
}
