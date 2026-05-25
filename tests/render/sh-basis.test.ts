import { describe, expect, it } from "vitest";
import { evalShJs, SH_COUNT } from "../../src/render/orbital/sh-basis";

const C00 = 0.5 * Math.sqrt(1 / Math.PI);          // ≈ 0.28209479
const C1  = Math.sqrt(3 / (4 * Math.PI));           // ≈ 0.48860251

function coefs(updates: Record<number, number>): Float32Array {
  const a = new Float32Array(SH_COUNT);
  for (const k in updates) a[+k] = updates[+k];
  return a;
}

describe("evalShJs", () => {
  it("SH_COUNT is 16 (l up through 3)", () => {
    expect(SH_COUNT).toBe(16);
  });

  it("Y_0^0 contributes the constant 1/(2√π) regardless of direction", () => {
    const c = coefs({ 0: 1 });
    expect(evalShJs(c, 1, 0, 0)).toBeCloseTo(C00, 6);
    expect(evalShJs(c, 0, 0.7, 0.5)).toBeCloseTo(C00, 6);
    expect(evalShJs(c, -0.3, -0.4, 0.8)).toBeCloseTo(C00, 6);
  });

  it("Y_1^0 (p_z) at +z direction = √(3/4π); at xy-plane = 0", () => {
    const c = coefs({ 2: 1 });
    expect(evalShJs(c, 0, 0, 1)).toBeCloseTo(C1, 6);
    expect(evalShJs(c, 1, 0, 0)).toBeCloseTo(0, 6);
    expect(evalShJs(c, 0, 1, 0)).toBeCloseTo(0, 6);
    // Sign flips across z=0.
    expect(evalShJs(c, 0, 0, -1)).toBeCloseTo(-C1, 6);
  });

  it("Y_1^1 (p_x) is nonzero on +x, zero on +y and +z, flips sign on -x", () => {
    const c = coefs({ 3: 1 });
    expect(evalShJs(c, 1, 0, 0)).toBeCloseTo(C1, 6);
    expect(evalShJs(c, 0, 1, 0)).toBeCloseTo(0, 6);
    expect(evalShJs(c, 0, 0, 1)).toBeCloseTo(0, 6);
    expect(evalShJs(c, -1, 0, 0)).toBeCloseTo(-C1, 6);
  });

  it("Y_1^-1 (p_y) is nonzero on +y, zero on +x and +z", () => {
    const c = coefs({ 1: 1 });
    expect(evalShJs(c, 0, 1, 0)).toBeCloseTo(C1, 6);
    expect(evalShJs(c, 1, 0, 0)).toBeCloseTo(0, 6);
    expect(evalShJs(c, 0, 0, 1)).toBeCloseTo(0, 6);
  });

  it("Y_2^0 (d_z²) at +z is (1/4)√(5/π) · 2 = (1/2)√(5/π)", () => {
    const c = coefs({ 6: 1 });
    const expected = 0.25 * Math.sqrt(5 / Math.PI) * (3 * 1 - 1);
    expect(evalShJs(c, 0, 0, 1)).toBeCloseTo(expected, 6);
  });

  it("Y_2^2 (d_x²-y²) at +x is positive; at +y is negative", () => {
    const c = coefs({ 8: 1 });
    const ampPlus = evalShJs(c, 1, 0, 0);
    const ampMinus = evalShJs(c, 0, 1, 0);
    expect(ampPlus).toBeGreaterThan(0);
    expect(ampMinus).toBeLessThan(0);
    expect(ampPlus).toBeCloseTo(-ampMinus, 6);
  });

  it("Y_3^3 (f at l=3, m=3) is bipolar in φ — six lobes around z-axis", () => {
    const c = coefs({ 15: 1 });
    // At φ = 0 (x-axis), x³ - 3xy² = 1 - 0 = 1 → positive.
    expect(evalShJs(c, 1, 0, 0)).toBeGreaterThan(0);
    // At φ = π/3 (cos π/3, sin π/3, 0), x³ - 3xy² = (1/8) - 3·(1/2)·(3/4) = -1 → negative.
    const c60x = Math.cos(Math.PI / 3);
    const c60y = Math.sin(Math.PI / 3);
    expect(evalShJs(c, c60x, c60y, 0)).toBeLessThan(0);
  });

  it("is linear in the coefficient vector", () => {
    const a = coefs({ 0: 1, 2: 0.5 });
    const b = coefs({ 6: -0.3, 12: 0.7 });
    const sum = new Float32Array(16);
    for (let i = 0; i < 16; i++) sum[i] = a[i] + b[i];
    const x = 0.3, y = -0.4, z = 0.87;
    const va = evalShJs(a, x, y, z);
    const vb = evalShJs(b, x, y, z);
    const vs = evalShJs(sum, x, y, z);
    expect(vs).toBeCloseTo(va + vb, 6);
  });

  it("input vector does not have to be unit-length; result is computed on (x,y,z) directly", () => {
    // Note: callers are expected to pass normalized direction. This test just
    // documents that we DO NOT normalize internally — the polynomials are
    // evaluated on whatever (x,y,z) you pass.
    const c = coefs({ 0: 1 });
    expect(evalShJs(c, 5, 5, 5)).toBeCloseTo(C00, 6);
  });
});
