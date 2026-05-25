# OrbitalCloud

GPU-resident particle visualizer that renders 10K–1M walker particles sampling |ψ|² of a hydrogen-orbital-like wavefunction. Sliders shape the orbital, set the magnetic-field axis, and tune the dynamics; audio wiring is deferred to the future parameter router.

## What it draws

Each particle is a walker that:
1. **Drifts up ∇log|ψ|²** — keeps the cloud's shape matched to the orbital's probability density.
2. **Diffuses** — Brownian jitter so walkers don't collapse onto ψ-maxima.
3. **Precesses around the B axis** — Larmor rotation; the B vector is meant to be audio-modulated.
4. **Respawns on lifetime expiry or boundary escape** — uniform in a ball of `boundaryRadius / 2`.

Particles are colored bipolar by `sign(ψ)`: warm red where ψ > 0, cool blue where ψ < 0. Iconic textbook orbital look. Rendered via additive blending so overlapping points brighten into a glow where density is high.

## Math

### Wavefunction

```
ψ(r, θ, φ) = R_n(r) · Σ_{l, m} c_{l,m} · Y_l^m(θ, φ)
```

- **R_n(r) = L_{n-1}(2r/n) · exp(-r/n)** — simple-Laguerre × exp form for n ∈ {1, 2, 3, 4}. Higher n adds radial nodes (concentric zero-density shells). Physics normalization is dropped — visualization only cares about relative density.
- **Y_l^m for l ∈ {0..3}** — 16 real spherical harmonics (s, p, d, f orbitals). One slider per (l, m).
- See `src/render/orbital/sh-basis.ts` and `src/render/orbital/radial.ts` for the closed-form polynomials; both files ship JS + TSL implementations that share the same numerics. Tests cover the JS mirror at tabulated values.

### Walker update (per particle per compute dispatch)

```
ψ_p          = ψ(p)                                    # current value
∇log|ψ|²(p)  = central finite-difference over ε        # 6 ψ evals
drift        = drift_gain · ∇log|ψ|²(p)
precession   = precession_gain · cross(B, p)
diffusion    = diffusion · randn(3) · √dt
p_new        = p + (drift + precession) · dt + diffusion

if |p_new| > boundaryRadius OR lifetime < 0:
  p_new    = uniform-in-ball(boundaryRadius / 2)
  lifetime = lifetime_slider · jitter(0.5..1.0)
else:
  lifetime -= dt

sign[i] = sign(ψ(p_new))                               # for coloring
```

The `|ψ|² → max(|ψ|², ε)` floor before `log` is the standard QMC drift-stability trick.

## Architecture

### File layout

```
src/render/components/OrbitalCloud.ts   — Component class, compute kernel, render setup
src/render/orbital/sh-basis.ts          — real SH (l=0..3), JS + TSL
src/render/orbital/radial.ts            — Laguerre × exp (n=1..4), JS + TSL
tests/render/sh-basis.test.ts           — SH at known directions
tests/render/radial.test.ts             — R_n at known r values
```

### GPU resources

Three storage buffers, one TSL compute kernel, one `THREE.Points` mesh.

| Buffer | Type | Compute access | Vertex access |
|---|---|---|---|
| `positionsStorage` | vec3 per particle | read/write via `.element(instanceIndex)` | read-only via `.toAttribute()` |
| `signsStorage` | float per particle | write via `.element(instanceIndex).assign(sign(ψ))` | read via `.toAttribute()` |
| `lifetimesStorage` | float per particle | read/write | not used in render |

**Why two access paths for the same buffer:** WebGPU forbids read/write storage bindings in the vertex pipeline stage. `.element(instanceIndex)` generates a read/write binding (fine for compute, illegal for vertex). `.toAttribute()` exposes the same buffer as a read-only instanced attribute (legal for vertex). Both alias the same underlying GPU memory — no copy.

### Renderer

`THREE.Points` + `PointsNodeMaterial` + `AdditiveBlending`. Each storage entry corresponds to one vertex (vertex-index = particle-index), so `positionsStorage.toAttribute()` is the natural per-particle binding — no instancing.

Material setup:
- `positionNode = positionsStorage.toAttribute()` — per-particle position from compute
- `colorNode = mix(NEG_COLOR, POS_COLOR, sign·0.5 + 0.5)` — bipolar lerp on sign
- `blending = AdditiveBlending`, `transparent = true`, `depthWrite = false`

### Per-frame flow (in `OrbitalCloud.update()`)

1. Push uniforms from the params bag (16 SH coefs, n, radialScale, B vector, dynamics gains, lifetime, boundary, dt, frame counter).
2. `renderer.computeAsync(updateKernel)` — runs the per-particle kernel on the GPU.
3. App's RAF loop renders next frame; vertex shader pulls fresh positions from the storage buffer.

CPU does no per-particle work after init.

## Parameters

All exposed as sliders/dropdowns in the tweakpane "Orbital Cloud" folder. Per-component "Reset to defaults" button restores just this component's params without touching the enabled flag or other components.

| Group | Param | Range | Default | Notes |
|---|---|---|---|---|
| Orbital | `c_l_m` (×16) | [-1, 1] | 0 (except `c_0_0 = 1`) | One SH coefficient per (l, m), l ∈ {0..3} |
| Radial | `n` | discrete {1, 2, 3, 4} | 2 | Laguerre degree → radial nodes |
| Radial | `radialScale` | [0.2, 5.0] | 1.0 | Stretches r before R_n |
| Magnetic | `Bx`, `By`, `Bz` | [-1, 1] | (0, 1, 0) | B vector for Larmor precession |
| Dynamics | `diffusion` | [0, 0.2] | 0.02 | Brownian σ |
| Dynamics | `driftGain` | [0, 5] | 1.0 | ∇log|ψ|² multiplier |
| Dynamics | `precessionGain` | [0, 10] | 1.5 | B × p multiplier |
| Dynamics | `timescale` | [0, 3] | 1.0 | dt multiplier |
| Dynamics | `lifetime` | [0.5, 30] | 5.0 | Mean particle lifetime, seconds |
| Render | `numParticles` | discrete {10K, 100K, 500K, 1M} | 100K | Rebuilds storage buffers + mesh on change |
| Render | `pointSize` | [0.5, 8] | 2.0 | **No visible effect** — see pitfalls |
| Render | `boundaryRadius` | [1, 20] | 8.0 | Respawn ball radius |

`numParticles` is the only reconfig param (tears down and rebuilds storage + mesh). All others are hot — uniform value updates each frame.

## Pitfalls and design notes

### WebGPU point primitives are clamped to 1px

On Apple Silicon and many other GPUs, the WebGPU point primitive size is clamped to 1px regardless of `pointSize`/`sizeNode`. We attempted several workarounds:

- **`PointsNodeMaterial.sizeNode`** — silently clamped to 1px by the driver.
- **`SpriteNodeMaterial`** with `Sprite.count = N` — `Sprite.count` doesn't exist in three r170, so only one quad rendered.
- **`InstancedMesh` + `PlaneGeometry` + manual billboarding** — `storage.toAttribute()` indexes per-vertex, not per-instance. PlaneGeometry has 4 vertices, so each quad's corners read positions[0..3] instead of all reading positions[i] → stretched-triangle soup.
- **`SpriteNodeMaterial` + billboarding** — material overrides positionNode internally for single-sprite billboarding, ignoring our storage attribute.

The current Points implementation accepts 1px particles. Visual density comes from particle count + additive blending: 1M overlapping 1px hits brighten into a glow where density is high. The `pointSize` slider is kept for API symmetry (the live-update subscriber still mutates its uniform) but has no visible effect.

A working quad-particle path would likely require explicit `InstancedBufferAttribute` aliasing the same underlying buffer, or rendering a separate non-MRT post-pass. Worth revisiting if the 1px aesthetic becomes a limitation.

### Why additive blending works on Points but froze the GPU on quads

The scene's MRT pass writes color + view-space normal targets simultaneously. Additive blending into the normal target corrupts it; GTAO then ray-marches garbage normals. With 1px point fragments (~1M total) the work stays bounded. With billboarded quads (~100M+ fragments) it saturates the GPU hard enough to freeze the OS desktop.

### Hash seed precision

The compute kernel's randomness uses `hash(seed)` keyed off `(instanceIndex, frame)`. Naively combining via a large prime (`frame · 0x9E3779B1`) loses the instanceIndex contribution in float32 precision after a few frames — every particle gets the same noise. The fix is to wrap frame modulo a small power of two and combine with a small prime (`fWrap = frame.mod(65536); seed = float(instanceIndex) + fWrap · 13.37`) so the combined seed stays under ~1e6, well inside float32's exact-integer range (~16M).

### Booleans in WGSL

WGSL forbids `bool + bool`. The "boolean OR" pattern (`expired OR outside_boundary`) is implemented as `outsideMask.select(float(1), float(0)).add(expired.select(float(1), float(0))).greaterThan(0)` — cast each bool to f32, add, then compare.

### Sprite Class Note

`THREE.Sprite` in r170 has no `.count` property despite what some three.js examples imply. Using `(sprite as any).count = N` is a silent no-op — only one billboard renders. Don't try to use Sprite for instanced compute particles in this version of three.

## Audio integration

None today. All params are exposed as sliders for manual exploration. The intended path forward (see `ROADMAP.md`) is the parameter router: a layer that maps audio features (rmsLow/Mid/High, beat phase, tempo) to specific param keys. When that lands, route, e.g., `rmsLow → orbitalCloud.c_0_0`, `beat phase → orbitalCloud.Bx`, etc., without any changes to this component.

## Performance

At 100K particles on a modern Apple Silicon GPU: ~1 ms compute, ~0.5 ms render. 1M particles: ~3 ms compute, ~2 ms render. Numbers vary widely by GPU; if framerate suffers, drop `numParticles` first.

The compute kernel's hot path is 7 ψ evaluations per particle per frame (1 for sign, 6 for gradient). Each ψ evaluation is ~16 SH terms + 1 radial. That's a few hundred ALU ops per particle. At 1M × 60 Hz = ~20 GFLOPS — trivial on any discrete GPU; bounded but real on integrated/mobile GPUs.
