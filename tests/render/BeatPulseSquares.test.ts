import { describe, it, expect } from "vitest";
import { Group, Mesh, MeshBasicMaterial } from "three";
import { BeatPulseSquares } from "../../src/render/BeatPulseSquares";

describe("BeatPulseSquares", () => {
  it("constructs 4 meshes in a Group", () => {
    const data = new Float32Array([0, 0, 0, 0]);
    const r = new BeatPulseSquares({ source: () => data });
    expect(r.object3d).toBeInstanceOf(Group);
    expect(r.object3d.children.length).toBe(4);
  });

  it("update() maps source values to grayscale RGB", () => {
    const data = new Float32Array([1.0, 0.5, 0.25, 0.0]);
    const r = new BeatPulseSquares({ source: () => data });
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
    const r = new BeatPulseSquares({ source: () => data });
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
    const r = new BeatPulseSquares({ source: () => data });
    r.update();
    const cs = r.object3d.children.map(
      (c) => ((c as Mesh).material as MeshBasicMaterial).color,
    );
    expect(cs[0].r).toBeCloseTo(1.0); // 2.5 → clamped to 1
    expect(cs[1].r).toBe(0); // -0.5 → clamped to 0
    expect(cs[2].r).toBeCloseTo(1.0);
    expect(cs[3].r).toBe(0);
  });

  it("dispose() releases geometries and materials", () => {
    const data = new Float32Array([0, 0, 0, 0]);
    const r = new BeatPulseSquares({ source: () => data });
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
