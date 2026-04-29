import { describe, it, expect } from "vitest";
import { Group, Mesh, MeshBasicMaterial } from "three";
import { BeatPulseSquares } from "../../src/render/BeatPulseSquares";

describe("BeatPulseSquares", () => {
  it("constructs `count` square meshes laid out in a grid", () => {
    const data = new Float32Array([0, 0, 0, 0]);
    const r = new BeatPulseSquares({
      source: () => data,
      count: 4,
      centerX: 1.2,
      centerY: -0.5,
      cellSize: 0.08,
      gap: 0.02,
    });
    expect(r.object3d).toBeInstanceOf(Group);
    expect(r.object3d.children.length).toBe(4);
    // Row-major: index 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right.
    // stride = 0.08 + 0.02 = 0.10; half = 0.5; offsets: -0.05, +0.05.
    const tl = r.object3d.children[0] as Mesh;
    const tr = r.object3d.children[1] as Mesh;
    const bl = r.object3d.children[2] as Mesh;
    const br = r.object3d.children[3] as Mesh;
    expect(tl.position.x).toBeCloseTo(1.15);
    expect(tl.position.y).toBeCloseTo(-0.45); // top row: above center
    expect(tr.position.x).toBeCloseTo(1.25);
    expect(tr.position.y).toBeCloseTo(-0.45);
    expect(bl.position.x).toBeCloseTo(1.15);
    expect(bl.position.y).toBeCloseTo(-0.55);
    expect(br.position.x).toBeCloseTo(1.25);
    expect(br.position.y).toBeCloseTo(-0.55);
  });

  it("update() maps source values to grayscale RGB", () => {
    const data = new Float32Array([1.0, 0.5, 0.25, 0.0]);
    const r = new BeatPulseSquares({
      source: () => data,
      count: 4,
      centerX: 0,
      centerY: 0,
      cellSize: 0.1,
      gap: 0.0,
    });
    r.update();
    const cs = r.object3d.children.map(
      (c) => ((c as Mesh).material as MeshBasicMaterial).color,
    );
    expect(cs[0].r).toBeCloseTo(1.0);
    expect(cs[0].g).toBeCloseTo(1.0);
    expect(cs[0].b).toBeCloseTo(1.0);
    expect(cs[1].r).toBeCloseTo(0.5);
    expect(cs[2].r).toBeCloseTo(0.25);
    expect(cs[3].r).toBe(0);
  });

  it("update() with NaN values renders black", () => {
    const data = new Float32Array([NaN, NaN, NaN, NaN]);
    const r = new BeatPulseSquares({
      source: () => data,
      count: 4,
      centerX: 0,
      centerY: 0,
      cellSize: 0.1,
      gap: 0.0,
    });
    r.update();
    for (const child of r.object3d.children) {
      const mat = (child as Mesh).material as MeshBasicMaterial;
      expect(mat.color.r).toBe(0);
      expect(mat.color.g).toBe(0);
      expect(mat.color.b).toBe(0);
    }
  });

  it("update() clamps source values out of [0, 1]", () => {
    const data = new Float32Array([2.5, -0.5, 1.0, 0.0]);
    const r = new BeatPulseSquares({
      source: () => data,
      count: 4,
      centerX: 0,
      centerY: 0,
      cellSize: 0.1,
      gap: 0.0,
    });
    r.update();
    const cs = r.object3d.children.map(
      (c) => ((c as Mesh).material as MeshBasicMaterial).color,
    );
    expect(cs[0].r).toBeCloseTo(1.0); // 2.5 → clamped to 1
    expect(cs[1].r).toBe(0); // -0.5 → clamped to 0
    expect(cs[2].r).toBeCloseTo(1.0);
    expect(cs[3].r).toBe(0);
  });

  it("rejects non-square counts", () => {
    expect(
      () =>
        new BeatPulseSquares({
          source: () => new Float32Array(3),
          count: 3,
          centerX: 0,
          centerY: 0,
          cellSize: 0.1,
          gap: 0,
        }),
    ).toThrow();
  });

  it("dispose() releases geometries and materials", () => {
    const data = new Float32Array([0, 0, 0, 0]);
    const r = new BeatPulseSquares({
      source: () => data,
      count: 4,
      centerX: 0,
      centerY: 0,
      cellSize: 0.1,
      gap: 0,
    });
    let disposed = 0;
    for (const child of r.object3d.children) {
      const mesh = child as Mesh;
      mesh.geometry.addEventListener("dispose", () => {
        disposed += 1;
      });
    }
    r.dispose();
    expect(disposed).toBe(4);
  });
});
