import { describe, it, expect } from "vitest";
import { BufferGeometry, LineSegments } from "three";
import { PeakMarkers } from "../../src/render/PeakMarkers";

const linearXForLag = (lag: number, lagDomain: number) =>
  lagDomain <= 1 ? 0 : (lag / (lagDomain - 1)) * 2 - 1;

describe("PeakMarkers", () => {
  it("constructs LineSegments with 2 * maxPeaks vertices", () => {
    const data = new Float32Array(30).fill(NaN);
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
    });
    expect(pm.object3d).toBeInstanceOf(LineSegments);
    const geom = (pm.object3d as LineSegments).geometry as BufferGeometry;
    expect(geom.getAttribute("position").count).toBe(20);
    expect(geom.getAttribute("color").count).toBe(20);
  });

  it("update() with all-NaN source collapses every segment to (0, yCenter, 0) with black color", () => {
    const data = new Float32Array(30).fill(NaN);
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
    });
    pm.update();
    const geom = (pm.object3d as LineSegments).geometry as BufferGeometry;
    const pos = geom.getAttribute("position");
    const col = geom.getAttribute("color");
    for (let i = 0; i < 20; i++) {
      expect(pos.getX(i)).toBe(0);
      expect(pos.getY(i)).toBeCloseTo(-1.0);
      expect(pos.getZ(i)).toBe(0);
      expect(col.getX(i)).toBe(0);
      expect(col.getY(i)).toBe(0);
      expect(col.getZ(i)).toBe(0);
    }
  });

  it("update() places real peaks as vertical segments at xForLag(lag)", () => {
    const data = new Float32Array(30).fill(NaN);
    // Two peaks: [lag, mag, sharpness] stride 3.
    data[0] = 64; // peak 0 lag
    data[1] = 0.9; // peak 0 mag
    data[2] = 0.5; // peak 0 sharpness (ignored by renderer)
    data[3] = 128; // peak 1 lag
    data[4] = 0.5; // peak 1 mag
    data[5] = 0.4; // peak 1 sharpness
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
    });
    pm.update();
    const pos = ((pm.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");

    // Peak 0 (lag=64): x = (64/255)*2 - 1 ≈ -0.498
    const expectedX0 = (64 / 255) * 2 - 1;
    expect(pos.getX(0)).toBeCloseTo(expectedX0);
    expect(pos.getX(1)).toBeCloseTo(expectedX0);
    // Top vertex (yCenter + ySpan) and bottom vertex (yCenter - ySpan).
    const ys = [pos.getY(0), pos.getY(1)];
    expect(Math.max(...ys)).toBeCloseTo(-1.0 + 0.4);
    expect(Math.min(...ys)).toBeCloseTo(-1.0 - 0.4);

    // Peak 1 (lag=128).
    const expectedX1 = (128 / 255) * 2 - 1;
    expect(pos.getX(2)).toBeCloseTo(expectedX1);
    expect(pos.getX(3)).toBeCloseTo(expectedX1);

    // Slot 2 (NaN) collapsed to center.
    expect(pos.getX(4)).toBe(0);
    expect(pos.getY(4)).toBeCloseTo(-1.0);
  });

  it("color brightens at slot 0 and dims toward slot maxPeaks-1", () => {
    const data = new Float32Array(30);
    for (let i = 0; i < 10; i++) {
      data[3 * i] = 20 + 5 * i; // lag
      data[3 * i + 1] = 0.9; // mag
      data[3 * i + 2] = 0.5; // sharpness (ignored by renderer)
    }
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
      baseColor: 0xffff00, // pure yellow → R=1, G=1, B=0
    });
    pm.update();
    const col = ((pm.object3d as LineSegments).geometry as BufferGeometry).getAttribute("color");
    // Slot 0: full brightness → R=1, G=1.
    expect(col.getX(0)).toBeCloseTo(1.0);
    expect(col.getY(0)).toBeCloseTo(1.0);
    // Slot 9: dimmest → ~0.25 of full.
    expect(col.getX(2 * 9)).toBeCloseTo(0.25);
    expect(col.getY(2 * 9)).toBeCloseTo(0.25);
    // Both endpoints of a segment share the same color.
    expect(col.getX(0)).toBe(col.getX(1));
    expect(col.getY(0)).toBe(col.getY(1));
  });

  it("dispose() releases geometry + material", () => {
    const data = new Float32Array(30).fill(NaN);
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
    });
    const geom = (pm.object3d as LineSegments).geometry as BufferGeometry;
    let geomDisposed = false;
    geom.addEventListener("dispose", () => {
      geomDisposed = true;
    });
    pm.dispose();
    expect(geomDisposed).toBe(true);
  });
});
