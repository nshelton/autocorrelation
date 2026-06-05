import {
  BufferGeometry,
  BufferAttribute,
  LineSegments,
  LineBasicMaterial,
  AdditiveBlending,
} from "three";
import type { Component, ComponentDeps } from "./Component";

const MAX_RES = 64;

type Vec3 = [number, number, number];

// Three flat grid planes through the origin — XY, YZ, XZ — each an N×N set of
// crossing lines. resolution (N) is the line count per axis, shared by all
// planes; sizeX/Y/Z set the extents independently. A plane spanning extents
// (au, bv) is laid out by a (u, v) -> world mapping.
function buildGeometry(n: number, sx: number, sy: number, sz: number): BufferGeometry {
  // 2N lines per plane (N along each in-plane axis) × 3 planes.
  const positions = new Float32Array(3 * 2 * n * 2 * 3);
  let o = 0;
  const seg = (a: Vec3, b: Vec3) => {
    positions[o++] = a[0]; positions[o++] = a[1]; positions[o++] = a[2];
    positions[o++] = b[0]; positions[o++] = b[1]; positions[o++] = b[2];
  };

  const plane = (au: number, bv: number, toXYZ: (u: number, v: number) => Vec3) => {
    const u0 = -au / 2, u1 = au / 2;
    const v0 = -bv / 2, v1 = bv / 2;
    for (let j = 0; j < n; j++) {
      const v = v0 + (bv * j) / (n - 1);
      seg(toXYZ(u0, v), toXYZ(u1, v)); // line along u
    }
    for (let i = 0; i < n; i++) {
      const u = u0 + (au * i) / (n - 1);
      seg(toXYZ(u, v0), toXYZ(u, v1)); // line along v
    }
  };

  plane(sx, sy, (u, v) => [u, v, 0]); // XY
  plane(sy, sz, (u, v) => [0, u, v]); // YZ
  plane(sx, sz, (u, v) => [u, 0, v]); // XZ

  const geom = new BufferGeometry();
  geom.setAttribute("position", new BufferAttribute(positions, 3));
  return geom;
}

export class Grid implements Component {
  static id = "grid";
  static label = "Grid";
  static paramPrefix = "grid";
  static paramOpts = {
    sizeX: { min: 0.1, max: 10, step: 0.1 },
    sizeY: { min: 0.1, max: 10, step: 0.1 },
    sizeZ: { min: 0.1, max: 10, step: 0.1 },
    resolution: { min: 2, max: MAX_RES, step: 1 },
    opacity: { min: 0, max: 1, step: 0.01 },
  };
  static paramDefaults = {
    sizeX: 2,
    sizeY: 2,
    sizeZ: 2,
    resolution: 8,
    opacity: 0.5,
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private material: LineBasicMaterial;
  private lines: LineSegments | null = null;
  private lastX = NaN;
  private lastY = NaN;
  private lastZ = NaN;
  private lastRes = NaN;
  private lastOpacity = NaN;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    // Gray + additive so overlapping planes brighten where they cross.
    // depthWrite off is standard for additive transparency.
    this.material = new LineBasicMaterial({
      color: 0x888888,
      transparent: true,
      opacity: params.opacity,
      blending: AdditiveBlending,
      depthWrite: false,
    });
    this.rebuild();
  }

  // Recreate the LineSegments from scratch rather than swapping `.geometry` on
  // the live mesh: the WebGPU renderer doesn't reliably re-bind a reassigned
  // geometry, so a fresh mesh is the only way slider edits show up in realtime.
  private rebuild(): void {
    const n = Math.max(2, Math.round(this.params.resolution));
    if (this.lines) {
      this.scene.remove(this.lines);
      this.lines.geometry.dispose();
    }
    this.lines = new LineSegments(
      buildGeometry(n, this.params.sizeX, this.params.sizeY, this.params.sizeZ),
      this.material,
    );
    this.lines.frustumCulled = false;
    this.scene.add(this.lines);
    this.lastX = this.params.sizeX;
    this.lastY = this.params.sizeY;
    this.lastZ = this.params.sizeZ;
    this.lastRes = this.params.resolution;
  }

  update(): void {
    if (
      this.params.sizeX !== this.lastX ||
      this.params.sizeY !== this.lastY ||
      this.params.sizeZ !== this.lastZ ||
      this.params.resolution !== this.lastRes
    ) {
      this.rebuild();
    }
    if (this.params.opacity !== this.lastOpacity) {
      this.material.opacity = this.params.opacity;
      this.lastOpacity = this.params.opacity;
    }
  }

  dispose(): void {
    if (this.lines) {
      this.scene.remove(this.lines);
      this.lines.geometry.dispose();
      this.lines = null;
    }
    this.material.dispose();
  }
}
