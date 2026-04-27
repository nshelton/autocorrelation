import { describe, it, expect } from "vitest";
import { BufferGeometry, Line, Vector3 } from "three";
import { LineRenderer } from "../../src/render/LineRenderer";

describe("LineRenderer", () => {
  it("produces a Line with N positions for a Float32Array of length N", () => {
    const data = new Float32Array(8);
    const lr = new LineRenderer({ source: () => data });
    expect(lr.object3d).toBeInstanceOf(Line);
    const geom = (lr.object3d as Line).geometry as BufferGeometry;
    const pos = geom.getAttribute("position");
    expect(pos.count).toBe(8);
  });

  it("default layout maps i in [0, n-1] to x in [-1, 1] and value to y", () => {
    const data = new Float32Array([0, 0.5, 1, -1]);
    const lr = new LineRenderer({ source: () => data });
    lr.update();
    const pos = ((lr.object3d as Line).geometry as BufferGeometry).getAttribute("position");
    // i=0 → x=-1, i=3 → x=1, linear in between
    expect(pos.getX(0)).toBeCloseTo(-1);
    expect(pos.getX(3)).toBeCloseTo(1);
    expect(pos.getY(0)).toBeCloseTo(0);
    expect(pos.getY(1)).toBeCloseTo(0.5);
    expect(pos.getY(2)).toBeCloseTo(1);
    expect(pos.getY(3)).toBeCloseTo(-1);
    expect(pos.getZ(0)).toBeCloseTo(0);
  });

  it("custom layout function drives positions", () => {
    const data = new Float32Array([1, 2]);
    const layout = (i: number, _n: number, value: number) =>
      new Vector3(i * 10, value * 100, 7);
    const lr = new LineRenderer({ source: () => data, layout });
    lr.update();
    const pos = ((lr.object3d as Line).geometry as BufferGeometry).getAttribute("position");
    expect(pos.getX(0)).toBe(0);
    expect(pos.getY(0)).toBe(100);
    expect(pos.getZ(0)).toBe(7);
    expect(pos.getX(1)).toBe(10);
    expect(pos.getY(1)).toBe(200);
  });

  it("update() reflects changes to the source", () => {
    const buf = new Float32Array([0, 0]);
    const lr = new LineRenderer({ source: () => buf });
    lr.update();
    buf[0] = 0.9;
    lr.update();
    const pos = ((lr.object3d as Line).geometry as BufferGeometry).getAttribute("position");
    expect(pos.getY(0)).toBeCloseTo(0.9);
  });
});
