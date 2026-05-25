# OrbitalCloud component — design

Date: 2026-05-20
Status: draft, awaiting plan

## Concept

A new `Component` that renders 1M GPU-resident walker particles sampling the probability density `|ψ|²` of a hydrogen-orbital-like wavefunction. The angular shape is controlled by 16 spherical-harmonic coefficient sliders (s, p, d, f), the radial profile by a discrete principal quantum number `n ∈ {1..4}`, and an external magnetic field `B` drives Larmor precession of the cloud. Particles are bipolar-colored by `sign(ψ)`, giving the classic textbook lobed look.

Audio-reactivity is **out of scope**. The component exposes parameters as sliders only; the future parameter router will wire audio features → parameters.

## Math

### Wavefunction

ψ(p) = R_n(r) · Σ_{l, m} c_{l,m} · Y_l^m(θ, φ)

where (r, θ, φ) are spherical coordinates of particle position p, the sum runs over `l ∈ {0,1,2,3}` and `m ∈ {-l..+l}` (16 real SH coefficients `c_{l,m}`), and R_n(r) is the hydrogen radial function:

R_n(r) = L_{n-1}(2r / n) · exp(-r / n)

with L_k the degree-k associated Laguerre polynomial. Four cases (n=1..4) are hardcoded as a `switch` inside the shader. Higher `n` adds radial nodes (concentric zero-density shells) that are visually distinct.

A `radialScale` slider stretches `r` before passing into R_n, so the same orbital can be made tight or diffuse without changing `n`.

### Real spherical harmonics

Real Y_l^m for l up through 3 (16 entries total). The 16 sliders are indexed and labelled by (l, m):

- l=0: (0,0)  →  `s`
- l=1: (1,-1)(1,0)(1,1)  →  `p_y, p_z, p_x`
- l=2: (2,-2)(2,-1)(2,0)(2,1)(2,2)  →  `d_xy, d_yz, d_z², d_zx, d_x²-y²`
- l=3: (3,-3)(3,-2)(3,-1)(3,0)(3,1)(3,2)(3,3)  →  the seven f-orbitals

The standard real-SH closed forms are hardcoded; no recurrence.

### Walker dynamics

Per particle, per compute-shader step:

```
ψ_p   = ψ(p)
U(p)  = -log(max(|ψ_p|², ε))
∇U    = finite-difference of U at p ± ε·x̂, p ± ε·ŷ, p ± ε·ẑ   (6 ψ evals)
v     = drift_gain · (-∇U)
      + precession_gain · cross(B, p)
p    += v · dt + diffusion · randn(3) · sqrt(dt)
```

- **Drift** keeps the cloud shaped like the orbital (walkers climb `log|ψ|²`).
- **Precession** rotates the cloud as a whole around the B axis at angular velocity `precession_gain · |B|`.
- **Diffusion** prevents collapse to ψ maxima and keeps the lobes filled in.
- ε ≈ 1e-4 in the floor on `|ψ|²` is the standard QMC drift-stability trick.

Boundary: if `|p| > boundaryRadius`, respawn p uniformly inside the sphere of radius `boundaryRadius / 2`. Keeps the cloud bounded when sliders combine to produce escapes.

### Sign carrier

After the position update, each particle writes `sign(ψ_p)` into a parallel storage buffer that the points material reads for color.

## Architecture

### Files

```
src/render/components/
  OrbitalCloud.ts          — Component class (registered in App.ts component list)
src/render/orbital/
  wavefunction.ts          — TSL Fn for ψ(p) and ∇log|ψ|²(p)
  sh-basis.ts              — real SH eval up to l=3 (TSL Fn + JS mirror for tests)
  radial.ts                — R_n(r) Laguerre × exp (TSL Fn + JS mirror)
tests/render/
  sh-basis.test.ts         — Y_l^m at known angles match tabulated values
  radial.test.ts           — R_n(r) at known r match tabulated values
```

The TSL helpers in `src/render/orbital/` are pure functions (no state). They are imported by `OrbitalCloud.ts`. The JS mirrors are imported by the tests; the shader path uses the TSL Fn variants. Both share the same coefficient layout, so they can't drift in shape — but their numerical values are intentionally identical too (closed forms, not recurrence).

### Runtime layout

`OrbitalCloud`:
- Position storage: `instancedArray(N, 'vec3')`, seeded uniformly inside `boundaryRadius`.
- Sign storage: `instancedArray(N, 'float')`, all zero at init.
- TSL compute function `updateWalkers(dt)` reading the 16 SH coefficient uniforms, `n`, B vector, dynamics gains. Invoked via `renderer.compute(updateNode, [N])` each frame inside `update()`.
- Render: `Points` mesh with `PointsNodeMaterial`. `positionNode` reads from the position buffer; `colorNode` reads from the sign buffer and lerps between two colors (positive/negative — hardcoded warm/cool for v1, can become sliders later).
- Discrete `numParticles` change tears down both storage buffers and the `Points` mesh, rebuilds at new size (mirrors `ParticleView.rebuildBodies`).

### Parameters

| Group | Slider | Range | Default | Notes |
|---|---|---|---|---|
| Orbital | `c_0_0` | [-1, 1] | 1.0 | s |
| Orbital | `c_1_m` (×3) | [-1, 1] | 0 | p |
| Orbital | `c_2_m` (×5) | [-1, 1] | 0 | d |
| Orbital | `c_3_m` (×7) | [-1, 1] | 0 | f |
| Radial | `n` | discrete {1, 2, 3, 4} | 2 | Laguerre degree |
| Radial | `radialScale` | [0.2, 5.0] | 1.0 | r → r/scale before R_n |
| Magnetic | `Bx` | [-1, 1] | 0 | B vector x |
| Magnetic | `By` | [-1, 1] | 1 | B vector y |
| Magnetic | `Bz` | [-1, 1] | 0 | B vector z |
| Dynamics | `diffusion` | [0, 0.2] | 0.02 | Brownian σ |
| Dynamics | `driftGain` | [0, 5] | 1.0 | gradient pull |
| Dynamics | `precessionGain` | [0, 10] | 1.5 | B × p multiplier |
| Dynamics | `timescale` | [0, 3] | 1.0 | dt multiplier |
| Render | `numParticles` | discrete {10K, 100K, 500K, 1000K} | 100000 | rebuilds buffers |
| Render | `pointSize` | [0.5, 8] | 2.0 | pixel size |
| Render | `boundaryRadius` | [1, 20] | 8.0 | respawn radius |

Total: 28 sliders (16 orbital + 2 radial + 3 magnetic + 4 dynamics + 3 render) plus 1 discrete `numParticles` and 1 discrete `n`. Param prefix: `orbitalCloud`.

All sliders are hot (no buffer rebuild) except `numParticles`. The 16 SH coefficients, B vector, dynamics gains, `radialScale`, `pointSize`, `boundaryRadius`, and the discrete `n` (which only selects a Laguerre case in the shader `switch`) are uniforms updated each frame from the params bag; the shader reads the uniforms directly so the slider → visual latency is one frame.

### Compute shader sketch (TSL pseudocode)

```ts
const updateWalkers = Fn(([positions, signs, dt]) => {
  const i = instanceIndex;
  const p = positions.element(i);

  // 1. Evaluate ψ at p and at ±ε on each axis (7 evals total → 6 for gradient + 1 for sign).
  const psi_p = evalPsi(p, c_lm_uniforms, n_uniform, radialScale_uniform);
  // ...finite-difference ∇log|ψ|² ...
  const gradU = ...; // vec3

  // 2. Compose velocity.
  const drift     = gradU.mul(-driftGain);
  const precess   = cross(B_uniform, p).mul(precessionGain);
  const noise     = randn3(i, frameSeed).mul(diffusion).mul(sqrt(dt));

  // 3. Update position; respawn if |p_new| > boundaryRadius.
  const p_new = p.add(drift.add(precess).mul(dt)).add(noise);
  // ...respawn check + uniform-in-ball reseed...

  positions.element(i).assign(p_new);
  signs.element(i).assign(sign(psi_p));
});
```

The `Fn` definition is built once at component construction and re-invoked each frame with the current dt.

### Per-frame flow in `update()`

```
1. If numParticles changed → rebuild storage buffers + Points mesh.
2. Push 16 SH coefs + n + radialScale + B + 4 dynamics gains into uniforms.
3. dt = clock.getDelta() · timescale.
4. renderer.compute(updateWalkers(positions, signs, dt), [N]).
5. (Render runs in App's RAF loop as usual.)
```

## Risks / open questions

- **TSL compute maturity.** Three.js `instancedArray` + `Fn` + `renderer.compute()` is the documented pattern in r170+, but this codebase has prior Three.js quirks (dedupe note in CLAUDE.md). Mitigation: implementation should land 100K first, verify WebGPU compute runs cleanly end-to-end, then bump to 1M.
- **Finite-difference gradient cost.** 6 extra ψ evaluations per particle per frame. At 1M × 60Hz × 16 SH terms ≈ 6 GFLOPS — trivial on modern GPUs. If a target machine is slow, drop the `numParticles` slider; if we ever hit a real ceiling, switch to analytical ∇Y_l^m and ∂R_n/∂r (worth it only if profiling shows the gradient dominates).
- **Singular drift when ψ → 0.** `|ψ|² → max(|ψ|², ε)` before the log. Standard.
- **Test coverage.** Compute-shader correctness is not unit-testable here. Tests cover only the JS mirrors of SH and radial functions, at known-value points (e.g. Y_1^0 at θ=0 = √(3/4π); R_2(0) = 1). Visual inspection covers the rest.
- **Renderer ordering.** App owns the RAF loop and calls `componentManager.update()` once per frame. `OrbitalCloud.update()` does the compute dispatch synchronously; nothing else in the pipeline depends on ordering with other components.

## Out of scope

- Audio reactivity (deferred to the future parameter router; this component exposes plain sliders).
- Time evolution of ψ itself (e.g. e^{-iHt/ℏ} phase rotation) — the wavefunction is static between slider changes; only walkers move.
- Multi-orbital basis (mixing different n in one cloud) — single global n. If we want 2s + 3p mixing later it becomes a `c_{n,l,m}` cube instead of a (l,m) plane and slider count climbs to ~30.
- Anisotropic B-field gradients (Stern-Gerlach-style splitting). B is uniform.
