import { describe, expect, it } from "vitest";
import { createCurlNoise } from "../../src/render/curl-noise";

// Simple seeded PRNG for deterministic tests — Mulberry32.
function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("curl-noise", () => {
  it("returns a 3-vector (Float32Array length 3)", () => {
    const curl = createCurlNoise({ prng: makePrng(1) });
    const out = new Float32Array(3);
    curl(0.5, 0.5, 0.5, out);
    expect(out.length).toBe(3);
    expect(Number.isFinite(out[0])).toBe(true);
    expect(Number.isFinite(out[1])).toBe(true);
    expect(Number.isFinite(out[2])).toBe(true);
  });

  it("is deterministic given the same seed", () => {
    const a = createCurlNoise({ prng: makePrng(42) });
    const b = createCurlNoise({ prng: makePrng(42) });
    const oa = new Float32Array(3);
    const ob = new Float32Array(3);
    a(1.1, 2.2, 3.3, oa);
    b(1.1, 2.2, 3.3, ob);
    expect(oa[0]).toBe(ob[0]);
    expect(oa[1]).toBe(ob[1]);
    expect(oa[2]).toBe(ob[2]);
  });

  it("produces different values from different seeds", () => {
    const a = createCurlNoise({ prng: makePrng(1) });
    const b = createCurlNoise({ prng: makePrng(2) });
    const oa = new Float32Array(3);
    const ob = new Float32Array(3);
    a(0.5, 0.5, 0.5, oa);
    b(0.5, 0.5, 0.5, ob);
    expect(oa[0] !== ob[0] || oa[1] !== ob[1] || oa[2] !== ob[2]).toBe(true);
  });

  it("is approximately divergence-free (∇·curl(F) ≈ 0)", () => {
    // Divergence at point p ≈ (Fx(p+εx) - Fx(p-εx)) / 2ε
    //                        + (Fy(p+εy) - Fy(p-εy)) / 2ε
    //                        + (Fz(p+εz) - Fz(p-εz)) / 2ε
    // For a curl-of-something vector field, this should be near zero by
    // construction (∇·∇×F = 0 in continuous math; finite-difference noise
    // gives a small residual).
    const curl = createCurlNoise({ prng: makePrng(7) });
    const eps = 0.001;
    const sample = (x: number, y: number, z: number) => {
      const out = new Float32Array(3);
      curl(x, y, z, out);
      return out;
    };

    let maxDiv = 0;
    const probes = [
      [0.1, 0.2, 0.3],
      [-0.5, 0.7, 1.2],
      [2.4, -1.1, 0.0],
      [3.7, 3.7, 3.7],
    ];
    for (const [x, y, z] of probes) {
      const fxp = sample(x + eps, y, z)[0];
      const fxm = sample(x - eps, y, z)[0];
      const fyp = sample(x, y + eps, z)[1];
      const fym = sample(x, y - eps, z)[1];
      const fzp = sample(x, y, z + eps)[2];
      const fzm = sample(x, y, z - eps)[2];
      const div = (fxp - fxm + fyp - fym + fzp - fzm) / (2 * eps);
      maxDiv = Math.max(maxDiv, Math.abs(div));
    }
    // Loose bound — finite differences over noise give a small but nonzero
    // residual. Tighten if you find a tighter eps gives lower residuals.
    expect(maxDiv).toBeLessThan(1.0);
  });

  it("scale parameter changes the field's spatial frequency", () => {
    // At a smaller scale (zoomed in), neighboring samples should differ less.
    const tightScale = createCurlNoise({ prng: makePrng(1), scale: 0.1 });
    const looseScale = createCurlNoise({ prng: makePrng(1), scale: 5.0 });
    const out1 = new Float32Array(3);
    const out2 = new Float32Array(3);

    tightScale(0, 0, 0, out1);
    tightScale(0.05, 0, 0, out2);
    const tightDelta = Math.abs(out2[0] - out1[0]);

    looseScale(0, 0, 0, out1);
    looseScale(0.05, 0, 0, out2);
    const looseDelta = Math.abs(out2[0] - out1[0]);

    // Bigger scale value = higher input multiplier = faster variation
    // across the same spatial step. So looseDelta > tightDelta.
    expect(looseDelta).toBeGreaterThan(tightDelta);
  });
});
