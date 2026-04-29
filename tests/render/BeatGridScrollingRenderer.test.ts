import { describe, it, expect } from "vitest";
import { BufferGeometry, LineSegments } from "three";
import { BeatGridScrollingRenderer } from "../../src/render/BeatGridScrollingRenderer";

const linearXForIndex = (index: number, domain: number) =>
  domain <= 1 ? 0 : (index / (domain - 1)) * 2 - 1;

describe("BeatGridScrollingRenderer", () => {
  it("constructs LineSegments with 2 * maxLines vertices", () => {
    const data = new Float32Array([NaN, NaN, NaN]);
    const r = new BeatGridScrollingRenderer({
      source: () => data,
      maxLines: 16,
      domain: 512,
      yTop: -0.1,
      yBottom: -0.18,
      xForIndex: linearXForIndex,
    });
    expect(r.object3d).toBeInstanceOf(LineSegments);
    const geom = (r.object3d as LineSegments).geometry as BufferGeometry;
    expect(geom.getAttribute("position").count).toBe(32);
  });

  it("collapses every segment when period is NaN", () => {
    const data = new Float32Array([NaN, NaN, NaN]);
    const r = new BeatGridScrollingRenderer({
      source: () => data,
      maxLines: 8,
      domain: 512,
      yTop: -0.1,
      yBottom: -0.18,
      xForIndex: linearXForIndex,
    });
    r.update();
    const pos = ((r.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");
    for (let i = 0; i < 16; i++) {
      expect(pos.getX(i)).toBe(0);
      expect(pos.getY(i)).toBeCloseTo(-0.1);
    }
  });

  it("collapses every segment when phase is NaN", () => {
    const data = new Float32Array([24, NaN, NaN]);
    const r = new BeatGridScrollingRenderer({
      source: () => data,
      maxLines: 4,
      domain: 512,
      yTop: -0.1,
      yBottom: -0.18,
      xForIndex: linearXForIndex,
    });
    r.update();
    const pos = ((r.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");
    for (let i = 0; i < 8; i++) {
      expect(pos.getX(i)).toBe(0);
      expect(pos.getY(i)).toBeCloseTo(-0.1);
    }
  });

  it("places verticals at (domain - 1 - phase - k·period) when phase = 0", () => {
    // period 24, phase 0, domain 512 → beats at indices 511, 487, 463, ...
    const data = new Float32Array([24, 0, 1.0]);
    const r = new BeatGridScrollingRenderer({
      source: () => data,
      maxLines: 4,
      domain: 512,
      yTop: -0.1,
      yBottom: -0.18,
      xForIndex: linearXForIndex,
    });
    r.update();
    const pos = ((r.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");

    const expected = [511, 487, 463, 439];
    for (let k = 0; k < 4; k++) {
      const x = (expected[k] / 511) * 2 - 1;
      expect(pos.getX(2 * k)).toBeCloseTo(x);
      expect(pos.getX(2 * k + 1)).toBeCloseTo(x);
      expect(pos.getY(2 * k)).toBeCloseTo(-0.1);
      expect(pos.getY(2 * k + 1)).toBeCloseTo(-0.18);
    }
  });

  it("phase shifts the grid left along the domain", () => {
    // period 24, phase 8 → most-recent beat at index 511 - 8 = 503.
    const data = new Float32Array([24, 8, 1.0]);
    const r = new BeatGridScrollingRenderer({
      source: () => data,
      maxLines: 4,
      domain: 512,
      yTop: -0.1,
      yBottom: -0.18,
      xForIndex: linearXForIndex,
    });
    r.update();
    const pos = ((r.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");
    const expectedX0 = (503 / 511) * 2 - 1;
    expect(pos.getX(0)).toBeCloseTo(expectedX0);
    // Next beat: 503 - 24 = 479.
    const expectedX1 = (479 / 511) * 2 - 1;
    expect(pos.getX(2)).toBeCloseTo(expectedX1);
  });

  it("collapses out-of-range slots when k·period exceeds domain", () => {
    // period 200, phase 0, domain 512, maxLines 4 → beats at 511, 311, 111;
    // k=3 lands at -89 (out of range) → collapsed.
    const data = new Float32Array([200, 0, 1.0]);
    const r = new BeatGridScrollingRenderer({
      source: () => data,
      maxLines: 4,
      domain: 512,
      yTop: -0.1,
      yBottom: -0.18,
      xForIndex: linearXForIndex,
    });
    r.update();
    const pos = ((r.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");

    // First three slots are real.
    for (let k = 0; k < 3; k++) {
      expect(pos.getY(2 * k)).toBeCloseTo(-0.1);
      expect(pos.getY(2 * k + 1)).toBeCloseTo(-0.18);
    }
    // Slot 3: collapsed.
    expect(pos.getX(6)).toBe(0);
    expect(pos.getX(7)).toBe(0);
    expect(pos.getY(6)).toBeCloseTo(-0.1);
    expect(pos.getY(7)).toBeCloseTo(-0.1);
  });

  it("dispose() releases geometry + material", () => {
    const data = new Float32Array([24, 0, 1.0]);
    const r = new BeatGridScrollingRenderer({
      source: () => data,
      maxLines: 4,
      domain: 512,
      yTop: -0.1,
      yBottom: -0.18,
      xForIndex: linearXForIndex,
    });
    const geom = (r.object3d as LineSegments).geometry as BufferGeometry;
    let geomDisposed = false;
    geom.addEventListener("dispose", () => {
      geomDisposed = true;
    });
    r.dispose();
    expect(geomDisposed).toBe(true);
  });
});
