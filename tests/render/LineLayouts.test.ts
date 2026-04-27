import { describe, it, expect } from "vitest";
import { linearLayout, logSpectrumLayout } from "../../src/render/LineLayouts";

describe("linearLayout", () => {
  it("maps i in [0, n-1] to x in [-1, 1] and value to y centered on yOffset", () => {
    const layout = linearLayout(0.5, 0.4);
    const v0 = layout(0, 4, 0);
    const v3 = layout(3, 4, 1);
    expect(v0.x).toBeCloseTo(-1);
    expect(v3.x).toBeCloseTo(1);
    expect(v0.y).toBeCloseTo(0.5);            // value 0 → y = yOffset
    expect(v3.y).toBeCloseTo(0.5 + 0.4);      // value 1 → y = yOffset + height
    expect(v0.z).toBeCloseTo(0);
  });

  it("guards n <= 1 by mapping to x = 0", () => {
    const layout = linearLayout(0, 1);
    expect(layout(0, 1, 0.7).x).toBeCloseTo(0);
  });
});

describe("logSpectrumLayout", () => {
  it("maps bin index 0 to x = -1 and the last bin to x = +1 via log2", () => {
    const layout = logSpectrumLayout(-0.6, 0.5);
    const first = layout(0, 1024, 0);
    const last = layout(1023, 1024, 1);
    expect(first.x).toBeCloseTo(-1);
    expect(last.x).toBeCloseTo(1);
  });

  it("compresses high indices and stretches low indices (log spacing)", () => {
    const layout = logSpectrumLayout(0, 1);
    const a = layout(1, 1024, 0).x;        // bin 1
    const b = layout(2, 1024, 0).x;        // bin 2
    const c = layout(511, 1024, 0).x;      // one before mid
    const d = layout(512, 1024, 0).x;      // mid
    // per-bin span at low end should exceed per-bin span at high end
    const lowSpan = b - a;
    const highSpan = d - c;
    expect(lowSpan).toBeGreaterThan(highSpan);
  });

  it("places value 0 at yOffset and value 1 at yOffset + height", () => {
    const layout = logSpectrumLayout(-0.6, 0.5);
    expect(layout(0, 1024, 0).y).toBeCloseTo(-0.6);
    expect(layout(0, 1024, 1).y).toBeCloseTo(-0.1);
  });
});
