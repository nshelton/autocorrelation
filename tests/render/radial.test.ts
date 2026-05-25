import { describe, expect, it } from "vitest";
import { evalRadialJs } from "../../src/render/orbital/radial";

describe("evalRadialJs", () => {
  it("R_1(0) = 1 (L_0(0) · exp(0) = 1 · 1)", () => {
    expect(evalRadialJs(0, 1)).toBeCloseTo(1, 6);
  });

  it("R_1(1) = exp(-1) ≈ 0.3679", () => {
    expect(evalRadialJs(1, 1)).toBeCloseTo(Math.exp(-1), 6);
  });

  it("R_2(0) = L_1(0) · exp(0) = 1", () => {
    expect(evalRadialJs(0, 2)).toBeCloseTo(1, 6);
  });

  it("R_2(2) = L_1(2) · exp(-1) = -1 · 0.3679 = -0.3679", () => {
    expect(evalRadialJs(2, 2)).toBeCloseTo(-Math.exp(-1), 6);
  });

  it("R_3(0) = L_2(0) · exp(0) = 1", () => {
    expect(evalRadialJs(0, 3)).toBeCloseTo(1, 6);
  });

  it("R_4(0) = L_3(0) · exp(0) = 1", () => {
    expect(evalRadialJs(0, 4)).toBeCloseTo(1, 6);
  });

  it("R_n(r → ∞) decays to ≈ 0 for any n in {1..4}", () => {
    // r=1000 picked so the exp(-r/n) factor swamps the Laguerre polynomial
    // even at n=4 (where the polynomial grows as ~r³/6 before being damped).
    // At r=100 with n=4: |R_4| ≈ 2.4e-7 — too large for a 1e-10 threshold.
    for (const n of [1, 2, 3, 4]) {
      expect(Math.abs(evalRadialJs(1000, n))).toBeLessThan(1e-10);
    }
  });

  it("R_2 has a node (zero crossing) at r = 2 (L_1(x) = 1-x → zero at x=1 → r=1; with n=2 arg is 2r/n=r so zero at r=1)", () => {
    // Wait — with R_n(r) = L_{n-1}(2r/n)·exp(-r/n):
    // For n=2: arg = 2r/2 = r. L_1(r) = 1-r → zero at r=1. So R_2(1) should be 0.
    expect(evalRadialJs(1, 2)).toBeCloseTo(0, 6);
  });

  it("clamps unknown n to a safe shape (e.g. R_1)", () => {
    // Implementation choice: switch falls through to L_0 (constant 1). We
    // document this so the shader's identical fallback isn't a surprise.
    expect(evalRadialJs(0, 99)).toBeCloseTo(1, 6);
    expect(evalRadialJs(1, 99)).toBeCloseTo(Math.exp(-1), 6);
  });
});
