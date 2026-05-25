# ParticleView Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ParticleView` Component that renders up to 10k instanced icospheres driven by curl-noise + a single Newtonian attractor, with rapier-based particle-particle and wall collisions, lifetime-based respawn at a fixed spawn point.

**Architecture:** New file `src/render/components/ParticleView.ts` implementing the existing `Component` contract. CPU-only rapier physics; per-frame upload to an `InstancedMesh` allocated at `MAX_PARTICLES = 10000`. Per-particle metadata (lifetime, scale) in SoA `Float32Array`s. Curl-noise function lives in `src/render/curl-noise.ts` as a pure module — testable without three.js or rapier. `ComponentManager` is extended to support discrete params (currently only continuous), so `numParticles` can be a dropdown of preset sizes.

**Tech Stack:** TypeScript 5 strict, Three.js (webgpu), `@dimforge/rapier3d-compat`, `simplex-noise` (new dependency), tweakpane (existing), vitest + happy-dom (existing).

**Source spec:** `docs/superpowers/specs/2026-05-20-particle-view-design.md`

**One deliberate spec deviation:** The spec's test-strategy section calls for a "lifetime semantics" unit test using a fake-particle simulator. This plan does NOT extract that test, because the in-place mutation pattern (`lifetimes[i] -= dt; if (lifetimes[i] <= 0) respawn(i)`) is too trivial to test meaningfully without also mocking a rapier body (which is mostly testing the mock). Lifetime behavior is covered by manual browser verification (Task 8 Step 8: slide `lifetime` to 1s, observe rapid respawn cycles).

---

## File Map

**New files:**
- `src/render/curl-noise.ts` — pure curl-noise vector field function. No three.js / rapier deps.
- `src/render/components/ParticleView.ts` — the component class.
- `tests/render/curl-noise.test.ts` — unit tests for the noise function.

**Modified files:**
- `package.json` + `package-lock.json` — add `simplex-noise` dependency.
- `src/render/components/Component.ts` — extend `ComponentClass` with optional `paramKinds` and `paramDiscreteOptions`.
- `src/render/components/ComponentManager.ts` — honor discrete kinds in `allocateBag()` and `bindUI()`.
- `src/render/components/index.ts` — append `ParticleView` to `COMPONENTS`.
- `tests/render/ComponentManager.test.ts` — extend with a discrete-param fake-component test.

---

## Task 1: Add simplex-noise dependency

**Files:**
- Modify: `package.json`, `package-lock.json`

- [ ] **Step 1: Install the package**

```bash
cd /Users/nshelton/autocorrelation
npm install simplex-noise@4
```

Expected: `simplex-noise` appears in `dependencies` (NOT `devDependencies`), package-lock.json updates.

- [ ] **Step 2: Verify the import works**

Create a throwaway file at `/tmp/test-simplex.ts`:

```ts
import { createNoise3D } from "simplex-noise";
const noise = createNoise3D();
console.log(noise(1, 2, 3));
```

Don't actually run it — TypeScript verification only:

```bash
npx tsc --noEmit /tmp/test-simplex.ts
rm /tmp/test-simplex.ts
```

Expected: no errors. If TS complains about no type declarations, add `@types/simplex-noise` or check that the package ships its own `.d.ts` files (v4 does).

- [ ] **Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "feat(deps): add simplex-noise for ParticleView curl-noise field"
```

---

## Task 2: Curl-noise pure function (TDD)

**Files:**
- Create: `src/render/curl-noise.ts`
- Create: `tests/render/curl-noise.test.ts`

The function produces a divergence-free 3D vector field via the curl of a 3D noise scalar field, computed by finite differences. Three independent noise generators (offset in input space) form the vector noise field.

- [ ] **Step 1: Write the failing tests**

Create `tests/render/curl-noise.test.ts`:

```ts
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
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run tests/render/curl-noise.test.ts
```

Expected: all 5 tests fail with `Cannot find module '../../src/render/curl-noise'`.

- [ ] **Step 3: Implement curl-noise**

Create `src/render/curl-noise.ts`:

```ts
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
npx vitest run tests/render/curl-noise.test.ts
```

Expected: all 5 pass.

- [ ] **Step 5: Run full suite**

```bash
npm test
```

Expected: 41 tests pass (36 existing + 5 new).

- [ ] **Step 6: Commit**

```bash
git add src/render/curl-noise.ts tests/render/curl-noise.test.ts
git commit -m "feat(render): add pure curl-noise vector field function

Divergence-free 3D vector field via curl of a 3-component vector noise
field, finite-difference computed. Pure module, no three.js / rapier
deps. Writes into a caller-provided Float32Array to avoid per-call
allocation in the per-particle hot loop."
```

---

## Task 3: Extend `ComponentClass` interface for discrete params

**Files:**
- Modify: `src/render/components/Component.ts`

Pure type-level change. No tests; covered by Task 4's ComponentManager test extension.

- [ ] **Step 1: Add the optional fields**

Edit `src/render/components/Component.ts`. Find:

```ts
export interface ComponentClass {
  new (deps: ComponentDeps, params?: Record<string, number>): Component;
  id: string;
  label: string;
  paramPrefix?: string;
  paramOpts?: Record<string, { min: number; max: number; step?: number }>;
  paramDefaults?: Record<string, number>;
}
```

Replace with:

```ts
export interface ComponentClass {
  new (deps: ComponentDeps, params?: Record<string, number>): Component;
  id: string;
  label: string;
  paramPrefix?: string;
  paramOpts?: Record<string, { min: number; max: number; step?: number }>;
  paramDefaults?: Record<string, number>;
  // Per-key kind override. Absent or "continuous" → continuous (uses paramOpts
  // min/max/step). "discrete" → uses paramDiscreteOptions for the value set.
  paramKinds?: Record<string, "continuous" | "discrete">;
  // For each key whose paramKinds entry is "discrete", the allowed values.
  // Must be present when paramKinds[key] === "discrete".
  paramDiscreteOptions?: Record<string, number[]>;
}
```

- [ ] **Step 2: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only (main.ts:186, BeatGridMarkers.ts:51). No new errors.

- [ ] **Step 3: Commit**

```bash
git add src/render/components/Component.ts
git commit -m "feat(components): extend ComponentClass with optional discrete-param metadata

Adds paramKinds (continuous/discrete) and paramDiscreteOptions for
components that want enum-like sliders. ComponentManager honors them
in the next commit."
```

---

## Task 4: ComponentManager discrete-param support (TDD)

**Files:**
- Modify: `src/render/components/ComponentManager.ts`
- Test: `tests/render/ComponentManager.test.ts`

Extends `allocateBag()` to register either continuous or discrete schemas based on `paramKinds[key]`. Extends `bindUI()` to render either a slider or a dropdown.

- [ ] **Step 1: Add failing test**

Edit `tests/render/ComponentManager.test.ts`. Append to the existing `describe("ComponentManager", () => { ... })` block (just before the closing `});`):

```ts
  it("registers discrete schemas for keys declared in paramKinds", () => {
    class FakeDiscrete {
      static id = "fakeDisc";
      static label = "Fake Discrete";
      static paramPrefix = "fakeDisc";
      static paramOpts = { count: { min: 0, max: 0, step: 0 } }; // ignored for discrete
      static paramDefaults = { count: 1000 };
      static paramKinds = { count: "discrete" as const };
      static paramDiscreteOptions = { count: [500, 1000, 2000, 5000] };
      public params: Record<string, number>;
      constructor(_deps: ComponentDeps, params: Record<string, number>) {
        this.params = params;
      }
      update(): void {}
      dispose(): void {}
    }

    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeDiscrete as unknown as ComponentClass]);
    mgr.start();
    // The registered schema should be discrete with the right options.
    const schema = deps.paramStore.schemasInOrder().find((s) => s.key === "fakeDisc.count");
    expect(schema).toBeDefined();
    expect(schema!.kind).toBe("discrete");
    if (schema!.kind === "discrete") {
      expect(schema!.options).toEqual([500, 1000, 2000, 5000]);
    }
    // Default value should be the declared default.
    expect(deps.paramStore.get("fakeDisc.count")).toBe(1000);
  });

  it("rejects a discrete value not in the allowed options", () => {
    class FakeDiscrete {
      static id = "fakeDisc2";
      static label = "Fake Discrete 2";
      static paramPrefix = "fakeDisc2";
      static paramOpts = { count: { min: 0, max: 0, step: 0 } };
      static paramDefaults = { count: 1000 };
      static paramKinds = { count: "discrete" as const };
      static paramDiscreteOptions = { count: [500, 1000, 2000] };
      constructor(_deps: ComponentDeps, _params: Record<string, number>) {}
      update(): void {}
      dispose(): void {}
    }

    const deps = makeDeps();
    const mgr = new ComponentManager(deps, [FakeDiscrete as unknown as ComponentClass]);
    mgr.start();
    deps.paramStore.set("fakeDisc2.count", 1234);  // not in [500, 1000, 2000]
    // The store rejects the set; value stays at default.
    expect(deps.paramStore.get("fakeDisc2.count")).toBe(1000);
  });
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run tests/render/ComponentManager.test.ts -t "discrete"
```

Expected: both new tests fail (the manager currently always registers `kind: "continuous"`).

- [ ] **Step 3: Update `allocateBag` for discrete kinds**

In `src/render/components/ComponentManager.ts`, find the `allocateBag` method:

```ts
  private allocateBag(cls: ComponentClass): Record<string, number> | null {
    if (!cls.paramDefaults) return null;
    const { paramStore } = this.deps;
    const bag: Record<string, number> = {};
    const prefix = cls.paramPrefix ?? cls.id;
    for (const [k, def] of Object.entries(cls.paramDefaults)) {
      const fullKey = `${prefix}.${k}`;
      const opts = cls.paramOpts?.[k];
      paramStore.register({
        key: fullKey,
        label: k,
        kind: "continuous",
        reconfig: false,
        default: def,
        min: opts?.min ?? 0,
        max: opts?.max ?? 1,
        step: opts?.step ?? 0.01,
      });
      const v = paramStore.get(fullKey);
      bag[k] = typeof v === "number" ? v : def;
    }
    return bag;
  }
```

Replace with:

```ts
  private allocateBag(cls: ComponentClass): Record<string, number> | null {
    if (!cls.paramDefaults) return null;
    const { paramStore } = this.deps;
    const bag: Record<string, number> = {};
    const prefix = cls.paramPrefix ?? cls.id;
    for (const [k, def] of Object.entries(cls.paramDefaults)) {
      const fullKey = `${prefix}.${k}`;
      const kind = cls.paramKinds?.[k] ?? "continuous";
      if (kind === "discrete") {
        const options = cls.paramDiscreteOptions?.[k];
        if (!options) {
          throw new Error(
            `ComponentManager: ${cls.id}.${k} declared discrete but paramDiscreteOptions[${k}] is missing`,
          );
        }
        paramStore.register({
          key: fullKey,
          label: k,
          kind: "discrete",
          reconfig: false,
          default: def,
          options,
        });
      } else {
        const opts = cls.paramOpts?.[k];
        paramStore.register({
          key: fullKey,
          label: k,
          kind: "continuous",
          reconfig: false,
          default: def,
          min: opts?.min ?? 0,
          max: opts?.max ?? 1,
          step: opts?.step ?? 0.01,
        });
      }
      const v = paramStore.get(fullKey);
      bag[k] = typeof v === "number" ? v : def;
    }
    return bag;
  }
```

- [ ] **Step 4: Update `bindUI` for discrete sliders**

In the same file, find the slider-binding section inside `bindUI`:

```ts
      if (!slot.paramsBag || !slot.cls.paramOpts) continue;
      // Slider bindings are not pushed into paneTeardowns explicitly;
      // tweakpane's folder.dispose() cascades to child bindings and
      // their change listeners, and the folder IS in paneTeardowns.
      for (const [k, opts] of Object.entries(slot.cls.paramOpts)) {
        const fullKey = `${slot.cls.paramPrefix ?? slot.cls.id}.${k}`;
        const slider = folder.addBinding(slot.paramsBag, k, {
          ...opts,
          step: opts.step ?? (opts.max - opts.min) / 100,
        });
        slider.on("change", (e: { value: number }) => {
          paramStore.set(fullKey, e.value);
        });
      }
```

Replace with:

```ts
      if (!slot.paramsBag) continue;
      // Slider/dropdown bindings are not pushed into paneTeardowns
      // explicitly; tweakpane's folder.dispose() cascades to child
      // bindings and their change listeners, and the folder IS in
      // paneTeardowns.
      const allKeys = new Set<string>([
        ...Object.keys(slot.cls.paramOpts ?? {}),
        ...Object.keys(slot.cls.paramDefaults ?? {}),
      ]);
      for (const k of allKeys) {
        const fullKey = `${slot.cls.paramPrefix ?? slot.cls.id}.${k}`;
        const kind = slot.cls.paramKinds?.[k] ?? "continuous";
        let binding;
        if (kind === "discrete") {
          const options = slot.cls.paramDiscreteOptions?.[k] ?? [];
          binding = folder.addBinding(slot.paramsBag, k, {
            options: Object.fromEntries(options.map((v) => [String(v), v])),
          });
        } else {
          const opts = slot.cls.paramOpts?.[k];
          if (!opts) continue;
          binding = folder.addBinding(slot.paramsBag, k, {
            ...opts,
            step: opts.step ?? (opts.max - opts.min) / 100,
          });
        }
        binding.on("change", (e: { value: number }) => {
          paramStore.set(fullKey, e.value);
        });
      }
```

- [ ] **Step 5: Run all ComponentManager tests**

```bash
npx vitest run tests/render/ComponentManager.test.ts
```

Expected: all 13 pass (11 existing + 2 new).

- [ ] **Step 6: Run full suite**

```bash
npm test
```

Expected: 43 pass (41 existing + 2 new).

- [ ] **Step 7: Commit**

```bash
git add src/render/components/ComponentManager.ts tests/render/ComponentManager.test.ts
git commit -m "feat(components): ComponentManager honors discrete-kind params

Per-key paramKinds=\"discrete\" + paramDiscreteOptions on a component
class produces a discrete ParamStore schema (preset values, dropdown
widget in tweakpane). Falls back to continuous when the metadata is
absent — fully backwards-compatible with existing components."
```

---

## Task 5: ParticleView core (init, walls, bodies, curl-noise, lifetime, dispose)

**Files:**
- Create: `src/render/components/ParticleView.ts`
- Modify: `src/render/components/index.ts`

Big task — implements the bulk of the component. Skips attractor (Task 6) and live reconfig (Task 7), so the component is testable in browser at end of this task. `numParticles` is read from the bag at construction time; changes don't take effect until you toggle the component off/on.

- [ ] **Step 1: Write ParticleView**

Create `src/render/components/ParticleView.ts`:

```ts
import {
  InstancedMesh,
  IcosahedronGeometry,
  Object3D,
  Color,
  InstancedBufferAttribute,
} from "three";
import { MeshBasicNodeMaterial } from "three/webgpu";
import {
  vec3,
  vec4,
  float,
  dot,
  max,
  normalWorld,
  instancedBufferAttribute,
} from "three/tsl";
import RAPIER from "@dimforge/rapier3d-compat";
import { createCurlNoise } from "../curl-noise";
import type { Component, ComponentDeps } from "./Component";

const MAX_PARTICLES = 10000;
const BASE_RADIUS = 0.04;
const COLLISION_RATIO = 0.5;
const SPAWN_POINT = { x: 0, y: 0, z: 0 };
// Scale factor random range. Visual radius = BASE_RADIUS * scale.
const SCALE_MIN = 0.5;
const SCALE_MAX = 1.5;
// Per-particle lifetime jitter on top of the slider value.
const LIFETIME_JITTER_SECS = 1.0;

export class ParticleView implements Component {
  static id = "particleView";
  static label = "Particle View";
  static paramPrefix = "particleView";
  static paramOpts = {
    numParticles: { min: 0, max: 0, step: 0 }, // ignored — discrete kind below
    lifetime: { min: 1, max: 10, step: 0.1 },
    noiseScale: { min: 0.1, max: 5.0, step: 0.05 },
    noiseStrength: { min: 0, max: 20, step: 0.1 },
    containerSize: { min: 0.5, max: 4, step: 0.05 },
    restitution: { min: 0, max: 1, step: 0.01 },
    damping: { min: 0, max: 2, step: 0.01 },
  };
  static paramDefaults = {
    numParticles: 2000,
    lifetime: 3,
    noiseScale: 1.5,
    noiseStrength: 5,
    containerSize: 1.5,
    restitution: 0.6,
    damping: 0.2,
  };
  static paramKinds = {
    numParticles: "discrete" as const,
  };
  static paramDiscreteOptions = {
    numParticles: [500, 1000, 2000, 5000, 10000],
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private numParticles: number;
  private mesh: InstancedMesh | null = null;
  private world: RAPIER.World | null = null;
  private bodies: RAPIER.RigidBody[] = [];
  private colliders: RAPIER.Collider[] = [];
  private wallColliders: RAPIER.Collider[] = [];
  private lifetimes!: Float32Array;
  private maxLifetimes!: Float32Array;
  private scales!: Float32Array;
  private dummy = new Object3D();
  private curlOut = new Float32Array(3);
  private curlNoise!: (x: number, y: number, z: number, out: Float32Array) => void;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    this.numParticles = Math.round(params.numParticles);
    void this.init();
  }

  private async init(): Promise<void> {
    await RAPIER.init();
    if (this.disposed) return;

    // SoA storage. Allocated to MAX_PARTICLES once so we never re-allocate.
    this.lifetimes = new Float32Array(MAX_PARTICLES);
    this.maxLifetimes = new Float32Array(MAX_PARTICLES);
    this.scales = new Float32Array(MAX_PARTICLES);

    this.curlNoise = createCurlNoise({ scale: this.params.noiseScale });

    this.world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    this.addWalls(this.params.containerSize);
    this.spawnBodies(this.numParticles);

    // InstancedMesh allocated to MAX_PARTICLES; mesh.count controls how many
    // we render. Per-instance color via our own InstancedBufferAttribute.
    const colorArr = new Float32Array(MAX_PARTICLES * 3);
    const tmpColor = new Color(1, 1, 1);
    for (let i = 0; i < MAX_PARTICLES; i++) tmpColor.toArray(colorArr, i * 3);
    const colorAttr = new InstancedBufferAttribute(colorArr, 3);

    const mat = new MeshBasicNodeMaterial();
    const instColor = vec3(instancedBufferAttribute(colorAttr, "vec3", 3, 0));
    const lightDir = vec3(0.408, 0.866, 0.306);
    const ndotl = max(dot(normalWorld, lightDir), float(0.0));
    const lit = ndotl.mul(0.7).add(0.3);
    mat.colorNode = vec4(instColor.mul(lit), 1.0);

    const geom = new IcosahedronGeometry(BASE_RADIUS, 1);
    const mesh = new InstancedMesh(geom, mat, MAX_PARTICLES);
    mesh.count = this.numParticles;
    this.mesh = mesh;
    this.scene.add(mesh);
  }

  private addWalls(half: number): void {
    if (!this.world) return;
    // Six thin static box colliders forming a closed cube of half-extent
    // `half`. Thin so they don't visibly occupy the scene; restitution from
    // the body side dominates the bounce.
    const t = 0.05; // wall thickness
    const make = (
      hx: number,
      hy: number,
      hz: number,
      x: number,
      y: number,
      z: number,
    ) => {
      const desc = RAPIER.ColliderDesc.cuboid(hx, hy, hz)
        .setTranslation(x, y, z)
        .setRestitution(this.params.restitution);
      this.wallColliders.push(this.world!.createCollider(desc));
    };
    make(t, half + t, half + t, half + t, 0, 0); // +x
    make(t, half + t, half + t, -(half + t), 0, 0); // -x
    make(half + t, t, half + t, 0, half + t, 0); // +y
    make(half + t, t, half + t, 0, -(half + t), 0); // -y
    make(half + t, half + t, t, 0, 0, half + t); // +z
    make(half + t, half + t, t, 0, 0, -(half + t)); // -z
  }

  private spawnBodies(n: number): void {
    if (!this.world) return;
    const c = this.params.containerSize;
    for (let i = 0; i < n; i++) {
      const x = (Math.random() - 0.5) * 2 * c * 0.7;
      const y = (Math.random() - 0.5) * 2 * c * 0.7;
      const z = (Math.random() - 0.5) * 2 * c * 0.7;
      const scale = SCALE_MIN + Math.random() * (SCALE_MAX - SCALE_MIN);
      this.scales[i] = scale;
      this.maxLifetimes[i] = this.params.lifetime + Math.random() * LIFETIME_JITTER_SECS;
      this.lifetimes[i] = Math.random() * this.maxLifetimes[i]; // stagger initial expirations
      const body = this.world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic()
          .setTranslation(x, y, z)
          .setLinvel(
            (Math.random() - 0.5),
            (Math.random() - 0.5),
            (Math.random() - 0.5),
          )
          .setLinearDamping(this.params.damping)
          .setAngularDamping(this.params.damping),
      );
      const collider = this.world.createCollider(
        RAPIER.ColliderDesc.ball(BASE_RADIUS * scale * COLLISION_RATIO)
          .setRestitution(this.params.restitution),
        body,
      );
      this.bodies.push(body);
      this.colliders.push(collider);
    }
  }

  private respawn(i: number): void {
    const body = this.bodies[i];
    const newScale = SCALE_MIN + Math.random() * (SCALE_MAX - SCALE_MIN);
    this.scales[i] = newScale;
    this.maxLifetimes[i] = this.params.lifetime + Math.random() * LIFETIME_JITTER_SECS;
    this.lifetimes[i] = this.maxLifetimes[i];
    body.setTranslation(SPAWN_POINT, true);
    body.setLinvel(
      { x: (Math.random() - 0.5), y: (Math.random() - 0.5), z: (Math.random() - 0.5) },
      true,
    );
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    this.colliders[i].setRadius(BASE_RADIUS * newScale * COLLISION_RATIO);
  }

  update(): void {
    if (!this.world || !this.mesh) return;
    // Hot params read once.
    const noiseStrength = this.params.noiseStrength;
    const damping = this.params.damping;
    // Apply damping changes if they shifted (cheap).
    // Per-step time; rapier defaults to its internal timestep.
    this.world.step();
    const dt = this.world.timestep;

    for (let i = 0; i < this.numParticles; i++) {
      const body = this.bodies[i];
      body.setLinearDamping(damping);
      body.setAngularDamping(damping);

      this.lifetimes[i] -= dt;
      if (this.lifetimes[i] <= 0) {
        this.respawn(i);
      }

      const t = body.translation();
      // Curl noise as a velocity impulse — same pattern BoxView uses for
      // its spring force. Additive on linvel; cheap and stable.
      this.curlNoise(t.x, t.y, t.z, this.curlOut);
      const v = body.linvel();
      body.setLinvel(
        {
          x: v.x + this.curlOut[0] * noiseStrength * dt,
          y: v.y + this.curlOut[1] * noiseStrength * dt,
          z: v.z + this.curlOut[2] * noiseStrength * dt,
        },
        true,
      );

      const r = body.rotation();
      const s = this.scales[i];
      this.dummy.position.set(t.x, t.y, t.z);
      this.dummy.quaternion.set(r.x, r.y, r.z, r.w);
      this.dummy.scale.set(s, s, s);
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this.disposed = true;
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as MeshBasicNodeMaterial).dispose();
      this.mesh.dispose();
      this.mesh = null;
    }
    if (this.world) {
      // Frees all bodies + colliders too.
      this.world.free();
      this.world = null;
    }
    this.bodies = [];
    this.colliders = [];
    this.wallColliders = [];
  }
}
```

- [ ] **Step 2: Register ParticleView in the registry**

Edit `src/render/components/index.ts`. Find:

```ts
import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import type { ComponentClass } from "./Component";

// Order = render order in the scene (insertion order). Also drives the
// order of folders in the tweakpane panel. Add a new component: import it
// here and append to this array.
export const COMPONENTS: readonly ComponentClass[] = [DebugView, BoxView as unknown as ComponentClass];
```

Replace with:

```ts
import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import { ParticleView } from "./ParticleView";
import type { ComponentClass } from "./Component";

// Order = render order in the scene (insertion order). Also drives the
// order of folders in the tweakpane panel. Add a new component: import it
// here and append to this array.
export const COMPONENTS: readonly ComponentClass[] = [
  DebugView,
  BoxView as unknown as ComponentClass,
  ParticleView as unknown as ComponentClass,
];
```

- [ ] **Step 3: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 4: Run full suite**

```bash
npm test
```

Expected: 43 tests pass (no test changes in this task; ParticleView is exercised in browser only).

- [ ] **Step 5: Manual browser sanity check**

```bash
npm run dev
```

In the browser:
- Disable Debug View and Box View (uncheck their folder checkboxes).
- Enable Particle View. ~2000 white icospheres should appear, swirl in a curl-noise field, bounce off invisible walls, and respawn at the origin every 3 seconds (with jitter).
- Slide `noiseStrength` up — particles fly faster.
- Slide `lifetime` to 1s — particles respawn much more often.
- Slide `numParticles` to 5000 — change does NOT take effect until you toggle the component off and on again (we haven't wired reconfig yet — Task 7).

If any of the above is broken: stop, debug, do not commit. Common pitfalls:
- "Cannot read property of undefined" → RAPIER.init() didn't complete; check the async init guard.
- Particles spawn at infinity → check container size param.
- All particles freeze → curl noise returned NaN; check scale param > 0.

- [ ] **Step 6: Commit**

```bash
git add src/render/components/ParticleView.ts src/render/components/index.ts
git commit -m "feat(components): add ParticleView with curl-noise + walls + lifetime

Up to 10000 rapier-driven dynamic spheres in a contained box, swirled
by a divergence-free curl-noise field. Particle-particle and wall
collisions on. Lifetime-based respawn at origin. SoA Float32Arrays for
per-particle metadata; InstancedMesh allocated at MAX_PARTICLES with
mesh.count gating how many are rendered. numParticles slider read at
construction time only; reconfig wired in Task 7."
```

---

## Task 6: Single attractor force

**Files:**
- Modify: `src/render/components/ParticleView.ts`

Adds `attractorStrength` and `attractorMinRadius` params, applies Newtonian inverse-square attraction toward a fixed point each frame.

- [ ] **Step 1: Add the two new params to ParticleView's static metadata**

In `src/render/components/ParticleView.ts`, find:

```ts
  static paramOpts = {
    numParticles: { min: 0, max: 0, step: 0 }, // ignored — discrete kind below
    lifetime: { min: 1, max: 10, step: 0.1 },
    noiseScale: { min: 0.1, max: 5.0, step: 0.05 },
    noiseStrength: { min: 0, max: 20, step: 0.1 },
    containerSize: { min: 0.5, max: 4, step: 0.05 },
    restitution: { min: 0, max: 1, step: 0.01 },
    damping: { min: 0, max: 2, step: 0.01 },
  };
  static paramDefaults = {
    numParticles: 2000,
    lifetime: 3,
    noiseScale: 1.5,
    noiseStrength: 5,
    containerSize: 1.5,
    restitution: 0.6,
    damping: 0.2,
  };
```

Replace with:

```ts
  static paramOpts = {
    numParticles: { min: 0, max: 0, step: 0 }, // ignored — discrete kind below
    lifetime: { min: 1, max: 10, step: 0.1 },
    noiseScale: { min: 0.1, max: 5.0, step: 0.05 },
    noiseStrength: { min: 0, max: 20, step: 0.1 },
    containerSize: { min: 0.5, max: 4, step: 0.05 },
    restitution: { min: 0, max: 1, step: 0.01 },
    damping: { min: 0, max: 2, step: 0.01 },
    attractorStrength: { min: 0, max: 50, step: 0.1 },
    attractorMinRadius: { min: 0.05, max: 0.5, step: 0.01 },
  };
  static paramDefaults = {
    numParticles: 2000,
    lifetime: 3,
    noiseScale: 1.5,
    noiseStrength: 5,
    containerSize: 1.5,
    restitution: 0.6,
    damping: 0.2,
    attractorStrength: 5,
    attractorMinRadius: 0.2,
  };
```

- [ ] **Step 2: Add the attractor position constant**

Near the top of the file (with the other constants like `SPAWN_POINT`), add:

```ts
const ATTRACTOR_POSITION = { x: 0.5, y: 0, z: 0 };
```

- [ ] **Step 3: Apply attractor force in update()**

In `update()`, find the curl-noise impulse block:

```ts
      const t = body.translation();
      // Curl noise as a velocity impulse — same pattern BoxView uses for
      // its spring force. Additive on linvel; cheap and stable.
      this.curlNoise(t.x, t.y, t.z, this.curlOut);
      const v = body.linvel();
      body.setLinvel(
        {
          x: v.x + this.curlOut[0] * noiseStrength * dt,
          y: v.y + this.curlOut[1] * noiseStrength * dt,
          z: v.z + this.curlOut[2] * noiseStrength * dt,
        },
        true,
      );
```

Add an attractor force application immediately after, before the rotation read:

```ts
      const t = body.translation();
      // Curl noise as a velocity impulse — same pattern BoxView uses for
      // its spring force. Additive on linvel; cheap and stable.
      this.curlNoise(t.x, t.y, t.z, this.curlOut);
      const v = body.linvel();
      body.setLinvel(
        {
          x: v.x + this.curlOut[0] * noiseStrength * dt,
          y: v.y + this.curlOut[1] * noiseStrength * dt,
          z: v.z + this.curlOut[2] * noiseStrength * dt,
        },
        true,
      );

      // Newtonian attractor: F = strength * (A - p) / |A - p|^3.
      // Clamp |A - p| at attractorMinRadius to avoid singularity at r=0.
      if (attractorStrength > 0) {
        const dx = ATTRACTOR_POSITION.x - t.x;
        const dy = ATTRACTOR_POSITION.y - t.y;
        const dz = ATTRACTOR_POSITION.z - t.z;
        const r2 = dx * dx + dy * dy + dz * dz;
        const r = Math.sqrt(r2);
        if (r >= attractorMinRadius) {
          const invR3 = 1 / (r * r2);
          const k = attractorStrength * invR3;
          body.addForce({ x: dx * k, y: dy * k, z: dz * k }, true);
        }
      }
```

And in the params-read prelude near the top of `update`, find:

```ts
    const noiseStrength = this.params.noiseStrength;
    const damping = this.params.damping;
```

Replace with:

```ts
    const noiseStrength = this.params.noiseStrength;
    const damping = this.params.damping;
    const attractorStrength = this.params.attractorStrength;
    const attractorMinRadius = this.params.attractorMinRadius;
```

- [ ] **Step 4: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 5: Run full suite**

```bash
npm test
```

Expected: 43 tests pass.

- [ ] **Step 6: Manual browser sanity check**

```bash
npm run dev
```

- Particle View should still work as before with `attractorStrength = 5` (default). Particles should now visibly clump/swirl around `(0.5, 0, 0)` in addition to the curl-noise motion.
- Slide `attractorStrength` to 0 — clumping behavior goes away; particles do the pure curl-noise dance.
- Slide `attractorStrength` to 50 — particles get yanked hard toward the attractor point and orbit chaotically.
- Slide `attractorMinRadius` down to 0.05 — particles can get very close to the attractor; check that velocities stay finite.

If particles freeze, explode, or NaN-out: investigate before committing.

- [ ] **Step 7: Commit**

```bash
git add src/render/components/ParticleView.ts
git commit -m "feat(particles): add single Newtonian attractor at (0.5, 0, 0)

Per particle per frame, F = strength * (A - p) / |A - p|^3 via
body.addForce. Singularity clamped by skipping the force when
|A - p| < attractorMinRadius. Two new sliders: attractorStrength
(0..50, default 5) and attractorMinRadius (0.05..0.5, default 0.2).
Set strength to 0 to disable."
```

---

## Task 7: numParticles + containerSize reconfig handling

**Files:**
- Modify: `src/render/components/ParticleView.ts`

Adds a ParamStore subscription so the slider takes effect live. On `numParticles` change: free the rapier World, rebuild walls + bodies at the new count, set `mesh.count`. On `containerSize` change: just rebuild the wall colliders. Other params (lifetime, noise, attractor, etc.) are already hot — they're read from the bag each frame.

- [ ] **Step 1: Add a constructor parameter for the ParamStore**

This component subscribes directly to ParamStore for its own reconfig keys. ComponentManager already provides `deps.paramStore`.

In `ParticleView.ts`, find the constructor:

```ts
  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    this.numParticles = Math.round(params.numParticles);
    void this.init();
  }
```

Replace with:

```ts
  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.paramStore = deps.paramStore;
    this.params = params;
    this.numParticles = Math.round(params.numParticles);
    void this.init();
  }
```

And add the field near the top of the class (with the other private fields):

```ts
  private paramStore: ComponentDeps["paramStore"];
```

And a field for the unsub function:

```ts
  private storeUnsub: (() => void) | null = null;
```

- [ ] **Step 2: Subscribe to reconfig keys at end of init()**

In `init()`, after `this.scene.add(mesh);`, add:

```ts
    // Listen for reconfig param changes. Hot params are read from `this.params`
    // each frame (the bag is mutated in place by ComponentManager). Reconfig
    // params require structural rebuilds.
    this.storeUnsub = this.paramStore.subscribe((key, value) => {
      if (this.disposed) return;
      if (key === "particleView.numParticles" && typeof value === "number") {
        const n = Math.round(value);
        if (n !== this.numParticles) {
          this.rebuildBodies(n);
        }
      } else if (key === "particleView.containerSize" && typeof value === "number") {
        this.rebuildWalls(value);
      }
    });
```

- [ ] **Step 3: Add the rebuildBodies and rebuildWalls methods**

Add as private methods on the class (alongside `spawnBodies`):

```ts
  private rebuildBodies(n: number): void {
    if (!this.world || !this.mesh) return;
    // Free the entire world (drops all bodies + colliders), recreate it,
    // re-add walls, spawn the new body pool. The InstancedMesh and SoA
    // arrays persist — we just change mesh.count and reuse the storage.
    this.world.free();
    this.bodies = [];
    this.colliders = [];
    this.wallColliders = [];
    this.world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    this.addWalls(this.params.containerSize);
    this.spawnBodies(n);
    this.numParticles = n;
    this.mesh.count = n;
  }

  private rebuildWalls(half: number): void {
    if (!this.world) return;
    for (const c of this.wallColliders) this.world.removeCollider(c, false);
    this.wallColliders = [];
    this.addWalls(half);
  }
```

- [ ] **Step 4: Unsubscribe in dispose()**

In `dispose()`, find:

```ts
  dispose(): void {
    this.disposed = true;
    if (this.mesh) {
```

Replace with:

```ts
  dispose(): void {
    this.disposed = true;
    this.storeUnsub?.();
    this.storeUnsub = null;
    if (this.mesh) {
```

- [ ] **Step 5: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 6: Run full suite**

```bash
npm test
```

Expected: 43 tests pass.

- [ ] **Step 7: Manual browser sanity check**

```bash
npm run dev
```

- Particle View enabled, default 2000 particles visible.
- Open the `numParticles` dropdown, pick 500 — particle count visibly drops to 500. Brief pause (rapier rebuild) is acceptable.
- Pick 10000 — count jumps to 10000. Watch the framerate — if it falls below ~20fps, that's the spec's expected stress-test result; the user can choose to tune collisionRatio or drop back to 5000.
- Slide `containerSize` from 1.5 down to 0.5 — walls visibly close in (particles pack into a smaller volume).
- Slide it back to 3.0 — particles spread out again.
- Toggle the component off and back on — clean teardown and rebuild, no console errors.

If you see "RangeError: ..." or "rapier already disposed" → likely an order-of-ops issue in rebuildBodies (free vs new). Stop and fix before committing.

- [ ] **Step 8: Commit**

```bash
git add src/render/components/ParticleView.ts
git commit -m "feat(particles): live reconfig for numParticles and containerSize

numParticles change: free the rapier World, recreate, re-add walls,
spawn the new pool. mesh.count is reset; the InstancedMesh and SoA
arrays persist. containerSize change: just remove + re-add the wall
colliders. Subscriber unwired in dispose(); guarded against disposed
state."
```

---

## Task 8: Full manual verification

**Files:** none (verification only)

End-to-end browser checkout against the spec's verification checklist.

- [ ] **Step 1: Clean baseline**

In an incognito browser tab to clear persisted state, OR in the existing tab's devtools:
```js
localStorage.removeItem("autocorrelation.params.v1")
```

- [ ] **Step 2: Start dev server**

```bash
npm run dev
```

Open the printed URL, click Mic (or press T for test source).

- [ ] **Step 3: Default state**

- Three component folders in the panel: Debug View, Box View, Particle View — all enabled by default.
- Particle View shows: a `numParticles` dropdown (default 2000), and continuous sliders for `lifetime`, `noiseScale`, `noiseStrength`, `containerSize`, `restitution`, `damping`, `attractorStrength`, `attractorMinRadius`.
- 2000 white icospheres swirl around, bouncing off invisible walls, gently clumping toward `(0.5, 0, 0)`.

- [ ] **Step 4: Noise responses**

- Slide `noiseStrength` to 20 — motion intensifies.
- Slide `noiseStrength` to 0 — particles still get pulled by the attractor but no swirl.
- Slide `noiseScale` to 0.1 — large smooth swirls.
- Slide `noiseScale` to 5.0 — fine chaotic motion.

- [ ] **Step 5: Attractor responses**

- Slide `attractorStrength` to 50 — particles get yanked toward `(0.5, 0, 0)`.
- Slide to 0 — clumping behavior disappears.
- Set strength back to 5; slide `attractorMinRadius` from 0.5 down to 0.05. Particles can get closer to the attractor. No NaN explosions.

- [ ] **Step 6: Container size**

- Slide `containerSize` to 0.5 — walls close in, particles compress.
- Slide to 4 — particles spread out.

- [ ] **Step 7: numParticles reconfig**

- Select 500 from dropdown — particle count visibly drops to 500.
- Select 10000 — count jumps to 10000. Note framerate. Spec says this is the upper stress-test bound.
- Set back to 2000 for the rest of the checks.

- [ ] **Step 8: Lifetime**

- Slide `lifetime` to 1 — particles visibly cycle through respawn-at-origin much faster.
- Slide `lifetime` to 10 — particles stay alive for ~10s before respawning.

- [ ] **Step 9: Toggle and persistence**

- Uncheck Particle View — all particles disappear cleanly. Sliders remain visible.
- Re-check — particles reappear with the same param values you set.
- Reload the page, restart audio. Particle View should still be enabled (or disabled, whatever state you left it in) with all your slider values restored.

- [ ] **Step 10: HMR**

- Edit `src/render/components/ParticleView.ts` (e.g. change `BASE_RADIUS` from 0.04 to 0.06). Save.
- The Particle View should reconstruct without a full page reload. Particles are visibly larger.
- Revert the edit.

- [ ] **Step 11: Coexistence with BoxView**

- Enable Box View alongside Particle View. Both render simultaneously. Framerate stays reasonable (BoxView contributes 1024 boxes + physics; combined load is significant).
- Disable Box View. Particle View remains running.

- [ ] **Step 12: Final commit (if anything was fixed)**

If you fixed something during this task:

```bash
git add -p  # review hunks
git commit -m "fix(particles): <describe>"
```

Otherwise skip.

---

## Done

When all 8 tasks check out:
- `npm test` passes (43 tests)
- `npx tsc --noEmit` shows only the 2 pre-existing errors
- Manual verification all 12 steps pass
- Git log shows ~7 new commits (1 per implementation task, +0/1 from verification)

End state: ParticleView is a fully-toggleable visualizer alongside BoxView and DebugView. Up to 10k physics-driven instanced icospheres swirl in a divergence-free curl-noise field with a Newtonian attractor at `(0.5, 0, 0)`, bouncing off walls and respawning at the origin every few seconds. The ComponentManager has discrete-param support that any future component can opt into.
