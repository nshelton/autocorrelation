import { createNoise3D } from "simplex-noise";

export interface CurlNoiseOpts {
  // PRNG returning floats in [0, 1). Defaults to Math.random.
  prng?: () => number;
  // Multiplier on input coords before noise lookup. Larger = higher spatial
  // frequency (more detail per unit distance).
  scale?: number;
}

// createCurlNoise builds a 3D vector field F such that F = ∇ × P, where
// P is itself a 3D vector noise field (three independent scalar noises
// offset in input space). ∇ · F = 0 by construction (continuous-math
// identity), so the field has no sources/sinks — particles advected by
// F swirl rather than accumulate.
//
// Writes result into `out` (a length-3 Float32Array) to avoid per-call
// allocations in the hot loop.
export function createCurlNoise(opts: CurlNoiseOpts = {}) {
  const prng = opts.prng ?? Math.random;
  const scale = opts.scale ?? 1.0;
  // Three independent noise functions, one per vector component.
  const nx = createNoise3D(prng);
  const ny = createNoise3D(prng);
  const nz = createNoise3D(prng);

  // Finite-difference epsilon. Tight enough that curl ≈ true curl; loose
  // enough that the noise gradient is well-conditioned.
  const eps = 1e-3;
  const invTwoEps = 1 / (2 * eps);

  return (x: number, y: number, z: number, out: Float32Array): void => {
    const sx = x * scale;
    const sy = y * scale;
    const sz = z * scale;

    // curl(P) = (dPz/dy - dPy/dz, dPx/dz - dPz/dx, dPy/dx - dPx/dy)
    const dPz_dy = (nz(sx, sy + eps, sz) - nz(sx, sy - eps, sz)) * invTwoEps;
    const dPy_dz = (ny(sx, sy, sz + eps) - ny(sx, sy, sz - eps)) * invTwoEps;
    const dPx_dz = (nx(sx, sy, sz + eps) - nx(sx, sy, sz - eps)) * invTwoEps;
    const dPz_dx = (nz(sx + eps, sy, sz) - nz(sx - eps, sy, sz)) * invTwoEps;
    const dPy_dx = (ny(sx + eps, sy, sz) - ny(sx - eps, sy, sz)) * invTwoEps;
    const dPx_dy = (nx(sx, sy + eps, sz) - nx(sx, sy - eps, sz)) * invTwoEps;

    out[0] = dPz_dy - dPy_dz;
    out[1] = dPx_dz - dPz_dx;
    out[2] = dPy_dx - dPx_dy;
  };
}
