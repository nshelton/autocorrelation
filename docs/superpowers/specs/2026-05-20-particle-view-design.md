# Particle View Component

**Date:** 2026-05-20
**Status:** Approved (pending review)

## Motivation

The first new visualizer component built on top of the toggleable-components system shipped in `2026-05-20-toggleable-components-design.md`. Establishes a second non-debug visualizer (BoxView is the first) and exercises the Component contract from a different angle — many small rigid bodies driven by a force field, with lifetime-based recycling and a configurable count.

## Non-goals

- Multiple attractors. Start with one; the design hooks (single `attractorStrength` slider, fixed position) are scoped so adding 2-4 is a focused follow-up.
- Attractor visualization (debug sphere). Particles' orbital motion is sufficient visual feedback at this stage.
- GPU-based simulation. Rapier is CPU-only and has no API to source positions from a `GPUBuffer`. The honest path is CPU rapier + per-frame upload to the InstancedMesh.
- Stable Keplerian orbits. Curl noise running concurrently would destroy them anyway; the look is chaotic-swirl-around-attractor.
- Gravity. No `world.gravity` value beyond zero.
- Audio reactivity. ParticleView is a self-contained physics+noise system. Hooking it to FeatureStore buffers (RMS-driven spawn rate, etc.) is a future enhancement.
- Color modulation by lifetime fraction. Uniform white initially; trivial follow-up.

## Design

### Component contract

New file: `src/render/components/ParticleView.ts`, registered in `src/render/components/index.ts`. Implements `Component` and exposes static `id`/`label`/`paramPrefix`/`paramOpts`/`paramDefaults`. ComponentManager handles the enable checkbox and the stable params bag the same way it does for BoxView.

```ts
export class ParticleView implements Component {
  static id = "particleView";
  static label = "Particle View";
  static paramPrefix = "particleView";
  static paramOpts = { /* see Params below */ };
  static paramDefaults = { /* see Params below */ };

  constructor(deps: ComponentDeps, params: Record<string, number>) { /* … */ }
  update(): void { /* per-frame */ }
  dispose(): void { /* tear down */ }
}
```

Registered in `src/render/components/index.ts`:
```ts
export const COMPONENTS: readonly ComponentClass[] = [
  DebugView,
  BoxView as unknown as ComponentClass,
  ParticleView as unknown as ComponentClass,
];
```

### Params

| Key | Kind | Range | Default | Reconfig? |
|---|---|---|---|---|
| `particleView.numParticles` | discrete | [500, 1000, 2000, 5000, 10000] | 2000 | yes |
| `particleView.lifetime` | continuous | [1, 10] s | 3 | hot |
| `particleView.noiseScale` | continuous | [0.1, 5.0] | 1.5 | hot |
| `particleView.noiseStrength` | continuous | [0, 20] | 5 | hot |
| `particleView.containerSize` | continuous | [0.5, 4] | 1.5 | yes (wall rebuild) |
| `particleView.restitution` | continuous | [0, 1] | 0.6 | hot |
| `particleView.damping` | continuous | [0, 2] | 0.2 | hot |
| `particleView.attractorStrength` | continuous | [0, 50] | 5 | hot |
| `particleView.attractorMinRadius` | continuous | [0.05, 0.5] | 0.2 | hot |

**Lifetime semantics:** the slider sets the base lifetime `L_base ∈ [1, 10]s`. Each particle's actual lifetime is `L_base + random()` (random in `[0, 1]s`) — so each particle gets a small per-instance jitter on top of the slider value. This staggers respawns so the population doesn't pulse.

**Reconfig** means the param triggers a teardown+rebuild of part of the rigid-body world (numParticles → full pool, containerSize → wall colliders). Hot means it can be read each frame from the bag without rebuilding.

### ComponentManager extension: discrete params

ComponentManager today registers every component param as `kind: "continuous"`. ParticleView's `numParticles` is `discrete`. Extend `ComponentManager.allocateBag()` to inspect a per-component `paramKinds` static, falling back to `"continuous"` when absent:

```ts
// On ComponentClass, optional:
paramKinds?: Record<string, "continuous" | "discrete">;
paramDiscreteOptions?: Record<string, number[]>;
```

`allocateBag` reads from these when present to register a discrete schema instead of continuous. ParticleView declares:
```ts
static paramKinds = { numParticles: "discrete" as const };
static paramDiscreteOptions = {
  numParticles: [500, 1000, 2000, 5000, 10000],
};
```

The slider widget in tweakpane is a dropdown for discrete kinds (already handled by `ParamPanel.addWidget`'s existing branch on `kind === "discrete"`, which `ComponentManager.bindUI()` will defer to via a unified `addParamBinding(folder, schema, paramsBag, key)` helper). This adds a small symmetric branch in `ComponentManager.bindUI`.

### Memory layout

At 10k particles, rapier's `world.step()` dominates the per-frame budget by 1-2 orders of magnitude over any plausible iteration cost. Memory layout matters less than rapier's internal data structures, but the choice that won't get in our way:

```ts
private bodies: RAPIER.RigidBody[];        // rapier-owned handles
private colliders: RAPIER.Collider[];      // rapier-owned handles
private lifetimes: Float32Array;           // remaining seconds, length N
private maxLifetimes: Float32Array;        // original assigned lifetime, length N
private scales: Float32Array;              // per-particle scale multiplier ∈ [0.5, 1.5], length N
```

`scales[i]` is a multiplier on `BASE_RADIUS` (defined in Visual section). The mesh's `IcosahedronGeometry` has radius = `BASE_RADIUS`; per-instance matrix scale = `scales[i]`. Rapier collider radius = `BASE_RADIUS * scales[i] * collisionRatio`.

SoA Float32Arrays for per-particle state. One linear pass per frame. Rapier owns position/rotation/velocity in its own (unspecified) internal layout — we never reach into it; we use the API.

No `SharedArrayBuffer`, no Wasm-memory views. Adding those before measuring would be premature.

### Curl noise force field

Curl noise = `curl(F(p))` for a 3D vector noise field `F`, computed via finite differences:

```
ε = 0.01
F(p) = ( noise(p.x, p.y, p.z), noise(p.x+97, p.y+31, p.z+13), noise(p.x+19, p.y+71, p.z+59) )
curl(F)(p) ≈ (
  (F.z(p + εy) - F.z(p - εy)) - (F.y(p + εz) - F.y(p - εz)),
  (F.x(p + εz) - F.x(p - εz)) - (F.z(p + εx) - F.z(p - εx)),
  (F.y(p + εx) - F.y(p - εx)) - (F.x(p + εy) - F.x(p - εy)),
) / (2ε)
```

Divergence-free by construction (`∇·curl(F) = 0`), so the resulting velocity field has no sources/sinks — particles swirl rather than accumulate.

Implementation uses the `simplex-noise` npm package (lightweight, deterministic, fast). 12 noise samples per particle per frame. At 10k particles × 60Hz = 7.2M samples/sec — well within CPU budget.

Per-particle per frame: scale the curl vector by `noiseStrength * dt`, add to current linvel via `body.setLinvel(currentVel + impulse, true)`. Same pattern BoxView uses for its spring force.

The `noiseScale` param scales the input position before noise lookup (smaller scale = larger noise features = smoother fields). The `noiseStrength` param scales the output.

### Lifecycle (per frame)

```
1. world.step()
2. For i in 0..numParticles:
   a. lifetimes[i] -= dt
   b. If lifetimes[i] <= 0: respawn particle i
      (teleport to spawn point, zero velocity, re-randomize lifetime & scale; update collider half-extents to match new scale)
   c. Apply curl-noise impulse to velocity (setLinvel)
   d. Apply attractor force (addForce, with min-radius clamp)
   e. Read translation+rotation from rapier, write to InstancedMesh matrix
      (with per-instance scale = scales[i])
3. mesh.instanceMatrix.needsUpdate = true
```

Init (first construction):
- Create `RAPIER.World({x:0, y:0, z:0})` (no gravity)
- Allocate the InstancedMesh at `MAX_PARTICLES = 10000` capacity (stays alive across numParticles reconfigs)
- Allocate `lifetimes`, `maxLifetimes`, `scales` as `Float32Array(MAX_PARTICLES)`
- Add 6 static wall colliders forming a `containerSize`-radius cube
- For i in 0..numParticles: create a dynamic rigid body at a random position inside the container, with random initial velocity, random scale `∈ [0.5, 1.5]`, random lifetime, attach a sphere collider with radius `BASE_RADIUS * scales[i] * collisionRatio` (see below)
- Set `mesh.count = numParticles`

`numParticles` reconfig: rebuild the rigid-body pool only. Free the rapier `World` (which drops all bodies+colliders), create a new `World`, add walls, spawn new bodies. Reuse the pre-allocated `InstancedMesh` and Float32Arrays; just update `mesh.count` to the new value.

`containerSize` reconfig: remove the 6 wall colliders from the existing world and create new ones at the new extents. Bodies stay alive.

Spawn point: `(0, 0, 0)` constant. Attractor position: `(0.5, 0, 0)` constant.

### Particle-particle collisions at 10k

The user explicitly chose particle-particle collisions on (not just walls). At 10k bodies this is the load-bearing perf risk. Two mitigations bake into the initial design:

1. **`collisionRatio = 0.5`** — the rapier collider radius is half the visual radius. Visually full-size, but collision pair count stays much lower because particles don't overlap until their visual surfaces are well inside each other. This is a hardcoded constant in v1; can be promoted to a slider later.

2. **`damping = 0.2`** — applied per body via `setLinearDamping`/`setAngularDamping`. Bleeds energy so the system settles into low-energy distributions rather than maintaining max contact density indefinitely.

If at 10k particles the frame budget is still blown, the followups are: (a) tighter collisionRatio (e.g. 0.25), (b) reduce solver iteration count, (c) downgrade to walls-only with a slider. None of these affect the v1 spec.

### Visual

- `IcosahedronGeometry(BASE_RADIUS, 1)` — `detail=1` gives 80 triangles, looks spherical from typical viewing distance.
- `InstancedMesh(geom, mat, MAX_PARTICLES)` allocated to maximum slider value (10000); we render only `numParticles` instances by setting `mesh.count = numParticles`.
- Per-instance color via `InstancedBufferAttribute` — uniform white in v1, infrastructure in place for per-particle modulation later.
- Material: `MeshBasicNodeMaterial` with hand-rolled lambert, same recipe BoxView uses (lights are unreliable on InstancedMesh + MeshStandardNodeMaterial in r170).

`BASE_RADIUS = 0.04` (smaller than BoxView's 0.06 to keep 10k from visually overwhelming the scene).

### Test strategy

**Unit tests (vitest, no browser, no rapier):**
- Curl noise function: divergence test (finite-difference `∇·curl(F)` ≈ 0 within ε), deterministic given seed, returns 3D vector with bounded magnitude.
- Lifetime semantics: a tiny fake-particle simulator (a single Float32Array + a respawn function), step `dt` past expiration, verify respawn was called with the right index.
- ComponentManager discrete-param extension: extend `tests/render/ComponentManager.test.ts` with a fake component that declares `paramKinds.foo = "discrete"` and `paramDiscreteOptions.foo = [1, 2, 4]`, verify the registered schema kind matches.

**Skipped at unit level (manual browser verification):**
- Rapier integration. `RAPIER.init()` is async and heavy in vitest/happy-dom; pulling 10k bodies into a test would be slow and brittle.
- Tweakpane folder/sliders. Already covered by ParamPanel tests for the patterns.
- Visual correctness of curl-noise swirls. Eyeball it in the browser.

**Manual verification checklist:**
- Toggle particleView on; 2000 white icospheres appear, swirl in curl-noise field, bounce off walls, respawn at origin.
- Slide `noiseStrength` up; motion gets more violent.
- Slide `attractorStrength` up; particles get pulled toward `(0.5, 0, 0)`.
- Slide `numParticles` to 10000; reallocation happens cleanly (brief pause is acceptable), 10k particles render.
- HMR: edit ParticleView, save — component reconstructs without console errors.

### Files touched

| File | Change |
|---|---|
| `src/render/components/ParticleView.ts` | **new** — the component |
| `src/render/components/index.ts` | add ParticleView to COMPONENTS array |
| `src/render/components/Component.ts` | extend `ComponentClass` interface with optional `paramKinds` and `paramDiscreteOptions` |
| `src/render/components/ComponentManager.ts` | `allocateBag()` honors discrete kinds; `bindUI()` defers to a shared helper that branches on kind |
| `src/render/curl-noise.ts` | **new** — pure noise function, no three.js / rapier deps |
| `tests/render/curl-noise.test.ts` | **new** — unit tests for the noise function |
| `tests/render/ComponentManager.test.ts` | extend with a discrete-param fake-component test |
| `package.json` | add `simplex-noise` dependency |

## Rationale

**Why CPU rapier + per-frame upload, not GPU sim:** rapier has no API to source from GPU buffers. The two real architectures are "CPU physics, GPU rendering" and "GPU sim, no physics." Since the user wants physics with collisions, CPU rapier wins. Bandwidth (~40 MB/s at 10k particles) is trivial over PCIe.

**Why discrete `numParticles` over continuous-with-rounding:** prevents the user from accidentally sliding to 100k and freezing the tab. The 5 preset steps cover the useful range (light/medium/dense/heavy/stress-test).

**Why bundle the ComponentManager discrete-param extension here:** it's a small extension (one branch in `allocateBag`, one in `bindUI`) that any future component with enum-like params will want. Doing it as a separate refactor would be busywork — natural to fold into the first user of the feature.

**Why fountain model (constant population) over emitter+pool:** the user explicitly said "allocate numParticles at init, or whenever the num particles changes" — that constraint maps directly to "bodies live forever in rapier, get teleported on lifetime expiry." Avoids allocation churn during normal operation.

**Why SoA Float32Arrays + rapier-owned position/rotation:** rapier already manages body state internally; trying to override that buys nothing and fights the engine. Per-particle metadata (lifetime, radius) is small and benefits from linear-access SoA layout. JavaScript engines won't reliably hide-class an AoS struct array to the same shape.

**Why single attractor at fixed position in v1:** YAGNI. The hooks for adding 3 more attractors are obvious extensions (loop over an array). Getting one working informs whether more would feel right.

**Why no attractor visualization in v1:** the orbital motion itself shows where the attractor is. Adding a debug sphere is a 5-line change in a follow-up if the user wants it.

**Why uniform white particles in v1:** matches BoxView. Lifetime-fade modulation is a 3-line change in the color attribute update path once we want it.

**Why `collisionRatio = 0.5`:** brings the effective particle-particle contact count from ~20k (full overlap) down to ~2-3k at 10k particle density. The single biggest perf lever for the "particle-particle + walls" choice.

## Open questions

None at design time.
