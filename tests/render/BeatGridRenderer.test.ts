import { describe, it, expect } from "vitest";
import { BufferGeometry, LineSegments } from "three";
import { BeatGridRenderer } from "../../src/render/BeatGridRenderer";

const linearXForLag = (lag: number, lagDomain: number) =>
  lagDomain <= 1 ? 0 : (lag / (lagDomain - 1)) * 2 - 1;

describe("BeatGridRenderer", () => {
  it("constructs LineSegments with 2 * maxLines vertices", () => {
    const data = new Float32Array([NaN, NaN]);
    const r = new BeatGridRenderer({
      source: () => data,
      maxLines: 16,
      lagDomain: 256,
      yTop: -0.6,
      yBottom: -0.7,
      xForLag: linearXForLag,
    });
    expect(r.object3d).toBeInstanceOf(LineSegments);
    const geom = (r.object3d as LineSegments).geometry as BufferGeometry;
    expect(geom.getAttribute("position").count).toBe(32);
  });

  it("update() with NaN period collapses every segment to a single point", () => {
    const data = new Float32Array([NaN, NaN]);
    const r = new BeatGridRenderer({
      source: () => data,
      maxLines: 8,
      lagDomain: 256,
      yTop: -0.6,
      yBottom: -0.7,
      xForLag: linearXForLag,
    });
    r.update();
    const pos = ((r.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");
    for (let i = 0; i < 16; i++) {
      expect(pos.getX(i)).toBe(0);
      expect(pos.getY(i)).toBeCloseTo(-0.6);
      expect(pos.getZ(i)).toBe(0);
    }
  });

  it("update() places vertical segments at integer multiples of period", () => {
    // period = 24, lagDomain = 256 → multiples 24, 48, 72, ... while ≤ 255.
    const data = new Float32Array([24, 0.5]);
    const r = new BeatGridRenderer({
      source: () => data,
      maxLines: 16,
      lagDomain: 256,
      yTop: -0.6,
      yBottom: -0.7,
      xForLag: linearXForLag,
    });
    r.update();
    const pos = ((r.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");

    // First multiple: lag 24 → x = (24/255)*2 - 1.
    const expectedX1 = (24 / 255) * 2 - 1;
    expect(pos.getX(0)).toBeCloseTo(expectedX1);
    expect(pos.getX(1)).toBeCloseTo(expectedX1);
    expect(pos.getY(0)).toBeCloseTo(-0.6);
    expect(pos.getY(1)).toBeCloseTo(-0.7);

    // Second multiple: lag 48.
    const expectedX2 = (48 / 255) * 2 - 1;
    expect(pos.getX(2)).toBeCloseTo(expectedX2);
    expect(pos.getX(3)).toBeCloseTo(expectedX2);

    // Third multiple: lag 72.
    const expectedX3 = (72 / 255) * 2 - 1;
    expect(pos.getX(4)).toBeCloseTo(expectedX3);
    expect(pos.getX(5)).toBeCloseTo(expectedX3);
  });

  it("update() collapses out-of-range multiples (lag > lagDomain - 1)", () => {
    // period = 100, lagDomain = 256 → multiples 100, 200; 300 is out of range.
    const data = new Float32Array([100, 0.5]);
    const r = new BeatGridRenderer({
      source: () => data,
      maxLines: 8,
      lagDomain: 256,
      yTop: -0.6,
      yBottom: -0.7,
      xForLag: linearXForLag,
    });
    r.update();
    const pos = ((r.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");

    // First two multiples are real (100, 200).
    const expectedX1 = (100 / 255) * 2 - 1;
    expect(pos.getX(0)).toBeCloseTo(expectedX1);
    expect(pos.getY(0)).not.toBeCloseTo(pos.getY(1));

    const expectedX2 = (200 / 255) * 2 - 1;
    expect(pos.getX(2)).toBeCloseTo(expectedX2);

    // Third multiple (300) and beyond: out of range — collapsed to (0, yTop, 0).
    for (let k = 3; k <= 8; k++) {
      const top = (k - 1) * 2;
      const bot = (k - 1) * 2 + 1;
      expect(pos.getX(top)).toBe(0);
      expect(pos.getX(bot)).toBe(0);
      expect(pos.getY(top)).toBeCloseTo(-0.6);
      expect(pos.getY(bot)).toBeCloseTo(-0.6);
    }
  });

  it("dispose() releases geometry + material", () => {
    const data = new Float32Array([24, 0.5]);
    const r = new BeatGridRenderer({
      source: () => data,
      maxLines: 8,
      lagDomain: 256,
      yTop: -0.6,
      yBottom: -0.7,
      xForLag: linearXForLag,
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
