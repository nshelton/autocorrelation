import {
  BufferGeometry,
  BufferAttribute,
  LineSegments,
  LineBasicMaterial,
} from "three";
import type { Component, ComponentDeps } from "./Component";

const AXIS_LEN = 10;
const SQ_INNER = 1;
const SQ_OUTER = 2;

// 3 axes (1 seg each) + 3 planes × 4 quadrants × 4 segs per square = 51 segs.
const SEG_COUNT = 3 + 3 * 4 * 4;

function buildGeometry(): BufferGeometry {
  const positions = new Float32Array(SEG_COUNT * 2 * 3);
  let o = 0;
  const seg = (
    x0: number, y0: number, z0: number,
    x1: number, y1: number, z1: number,
  ) => {
    positions[o++] = x0; positions[o++] = y0; positions[o++] = z0;
    positions[o++] = x1; positions[o++] = y1; positions[o++] = z1;
  };

  seg(-AXIS_LEN, 0, 0, AXIS_LEN, 0, 0);
  seg(0, -AXIS_LEN, 0, 0, AXIS_LEN, 0);
  seg(0, 0, -AXIS_LEN, 0, 0, AXIS_LEN);

  const a = SQ_INNER;
  const b = SQ_OUTER;
  // For each plane, emit 4 quadrant squares at signs (sa, sb) ∈ {±1}².
  // `axis` = which axis is held at 0 (0=X→YZ plane, 1=Y→XZ, 2=Z→XY).
  const planeSquare = (axis: number, sa: number, sb: number) => {
    const u0 = sa * a, u1 = sa * b;
    const v0 = sb * a, v1 = sb * b;
    const pt = (u: number, v: number): [number, number, number] => {
      if (axis === 0) return [0, u, v];
      if (axis === 1) return [u, 0, v];
      return [u, v, 0];
    };
    const c00 = pt(u0, v0);
    const c10 = pt(u1, v0);
    const c11 = pt(u1, v1);
    const c01 = pt(u0, v1);
    seg(c00[0], c00[1], c00[2], c10[0], c10[1], c10[2]);
    seg(c10[0], c10[1], c10[2], c11[0], c11[1], c11[2]);
    seg(c11[0], c11[1], c11[2], c01[0], c01[1], c01[2]);
    seg(c01[0], c01[1], c01[2], c00[0], c00[1], c00[2]);
  };

  for (let axis = 0; axis < 3; axis++) {
    for (const sa of [-1, 1]) {
      for (const sb of [-1, 1]) planeSquare(axis, sa, sb);
    }
  }

  const geom = new BufferGeometry();
  geom.setAttribute("position", new BufferAttribute(positions, 3));
  return geom;
}

export class DebugAxes implements Component {
  static id = "debugAxes";
  static label = "Debug Axes";

  private scene: ComponentDeps["scene"];
  private lines: LineSegments;
  private material: LineBasicMaterial;
  private geometry: BufferGeometry;

  constructor(deps: ComponentDeps) {
    this.scene = deps.scene;
    this.geometry = buildGeometry();
    this.material = new LineBasicMaterial({
      color: 0xffffff,
      depthTest: true,
      depthWrite: true,
    });
    this.lines = new LineSegments(this.geometry, this.material);
    this.lines.frustumCulled = false;
    this.scene.add(this.lines);
  }

  update(): void {}

  dispose(): void {
    this.scene.remove(this.lines);
    this.geometry.dispose();
    this.material.dispose();
  }
}
