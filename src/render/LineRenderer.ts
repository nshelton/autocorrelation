import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  ColorRepresentation,
  DynamicDrawUsage,
  Line,
  LineBasicMaterial,
  Object3D,
  Vector3,
} from "three";

export type LineLayoutFn = (i: number, n: number, value: number) => Vector3;
export type LineLayout = "linear" | "log" | "logRight";

export interface LineRendererOptions {
  source: () => Float32Array;
  color?: ColorRepresentation;
  layout?: LineLayout;
}

const linearLayout: LineLayoutFn = (i, n, value) => {
  const x = n <= 1 ? 0 : (i / (n - 1)) * 2 - 1;
  return new Vector3(x, value, 0);
};

const logLeftLayout: LineLayoutFn = (i, n, value) => {
  if (n <= 1) return new Vector3(0, value, 0);
  // log2(i + 1) / log2(n) ∈ [0, log2(n) / log2(n)] for i in [0, n-1]
  // Want [0, 1] → [-1, 1]
  const t = Math.log2(i + 1) / Math.log2(n);
  const y = Math.log2(value + 1);
  return new Vector3(t, y, 0);
};

const logRightLayout: LineLayoutFn = (i, n, value) => {
  if (n <= 1) return new Vector3(0, value, 0);
  // log2(i + 1) / log2(n) ∈ [0, log2(n) / log2(n)] for i in [0, n-1]
  // Want [0, 1] → [-1, 1]
  const t = Math.log2(n - i + 1) / Math.log2(n);
  const y = Math.log2(value + 1);
  return new Vector3(t, y, 0);
};

const layouts: Record<LineLayout, LineLayoutFn> = {
  linear: linearLayout,
  log: logLeftLayout,
  logRight: logRightLayout,
};

export class LineRenderer {
  readonly object3d: Object3D;
  private source: () => Float32Array;
  private layout: LineLayout;
  private positions: Float32Array;
  private positionAttribute: BufferAttribute;
  private lastLength = -1;

  constructor(opts: LineRendererOptions) {
    this.source = opts.source;
    this.layout = opts.layout ?? "linear";

    const initial = this.source();
    this.positions = new Float32Array(initial.length * 3);
    this.positionAttribute = new BufferAttribute(this.positions, 3);
    this.positionAttribute.setUsage(DynamicDrawUsage);

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", this.positionAttribute);

    const material = new LineBasicMaterial({
      color: opts.color ?? 0xffffff,
      blending: AdditiveBlending,
    });
    this.object3d = new Line(geometry, material);

    this.writeFromSource(initial);
  }

  update(): void {
    const buf = this.source();
    if (buf.length !== this.lastLength) {
      this.positions = new Float32Array(buf.length * 3);
      this.positionAttribute = new BufferAttribute(this.positions, 3);
      this.positionAttribute.setUsage(DynamicDrawUsage);
      ((this.object3d as Line).geometry as BufferGeometry).setAttribute(
        "position",
        this.positionAttribute,
      );
    }
    this.writeFromSource(buf);
  }

  dispose(): void {
    const line = this.object3d as Line;
    (line.geometry as BufferGeometry).dispose();
    (line.material as LineBasicMaterial).dispose();
    line.parent?.remove(line);
  }

  private writeFromSource(buf: Float32Array): void {
    const n = buf.length;
    const layout = layouts[this.layout];
    for (let i = 0; i < n; i++) {
      const v = layout(i, n, buf[i]);
      this.positions[i * 3] = v.x;
      this.positions[i * 3 + 1] = v.y;
      this.positions[i * 3 + 2] = v.z;
    }
    this.positionAttribute.needsUpdate = true;
    this.lastLength = n;
  }
}
