# OrbitalCloud Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `OrbitalCloud` Component that renders up to 1M GPU-resident walker particles sampling |ψ|² of a hydrogen-orbital-like wavefunction, colored bipolar by sign(ψ), with a B-field driving Larmor precession. ~28 sliders, no audio wiring (deferred to the param router).

**Architecture:** New file `src/render/components/OrbitalCloud.ts` implementing the existing `Component` contract. WebGPU compute (three's TSL `Fn` + `instancedArray`) updates positions and ψ-sign per frame; rendering is a `Points` mesh with a `PointsNodeMaterial` whose position and color nodes read from the storage buffers. Pure math (SH basis + radial functions) lives in `src/render/orbital/` with JS mirrors that are unit-tested. Compute-shader correctness is verified by visual inspection at each task.

**Tech Stack:** TypeScript 5 strict, Three.js (webgpu, TSL), vitest + happy-dom (existing).

**Source spec:** `docs/superpowers/specs/2026-05-20-orbital-cloud-design.md`

**Risks to watch for during implementation:**
- Three.js TSL compute API has evolved across versions. r170 examples (`webgpu_compute_particles.html`, `webgpu_compute_birds.html`) are the reference. If `PointsNodeMaterial` doesn't accept a positionNode in r170, fall back to `SpriteNodeMaterial` or a tiny `InstancedMesh` of billboarded quads.
- `renderer.computeAsync()` vs `renderer.compute()` — both exist in r170; we use `computeAsync()` to match the async render pattern App already uses (`this.post.renderAsync()`).
- TSL `hash(seed)` returns a float-ish; we'll seed it with `(instanceIndex + frameCounter * largeprime)` for per-particle per-frame randomness.

---

## File Map

**New files:**
- `src/render/orbital/sh-basis.ts` — JS `evalShJs` + TSL `evalShTsl` for real spherical harmonics l=0..3.
- `src/render/orbital/radial.ts` — JS `evalRadialJs` + TSL `evalRadialTsl` for R_n(r), n ∈ {1..4}.
- `src/render/components/OrbitalCloud.ts` — the Component class.
- `tests/render/sh-basis.test.ts` — unit tests for SH JS mirror.
- `tests/render/radial.test.ts` — unit tests for radial JS mirror.

**Modified files:**
- `src/render/components/Component.ts` — add `renderer: WebGPURenderer` to `ComponentDeps`.
- `src/App.ts` — pass `renderer` into ComponentManager deps.
- `src/render/components/ComponentManager.ts` — no change needed (just passes deps through; if there's a typed dep object literal we'll widen it).
- `src/render/components/index.ts` — append `OrbitalCloud`.
- `tests/render/ComponentManager.test.ts` — extend `makeDeps()` to include a fake renderer.

---

## Task 1: Add `renderer` to `ComponentDeps`

**Files:**
- Modify: `src/render/components/Component.ts`
- Modify: `src/App.ts`
- Modify: `tests/render/ComponentManager.test.ts`

The OrbitalCloud component needs `renderer.computeAsync()` for compute dispatch, which means it needs a handle on the `WebGPURenderer`. This is foundational; the next ten tasks all depend on it.

- [ ] **Step 1: Add `renderer` to the deps interface**

Edit `src/render/components/Component.ts`. Find:

```ts
import type { Scene } from "three";
import type { FeatureStore } from "../../store/FeatureStore";
import type { ParamStore } from "../../params/ParamStore";

// Shared dependencies passed to every component constructor. App builds this
// once at start and reuses it for every component instance.
export interface ComponentDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
}
```

Replace with:

```ts
import type { Scene } from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { FeatureStore } from "../../store/FeatureStore";
import type { ParamStore } from "../../params/ParamStore";

// Shared dependencies passed to every component constructor. App builds this
// once at start and reuses it for every component instance.
export interface ComponentDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
  // Required for components that dispatch WebGPU compute (e.g. OrbitalCloud).
  // Existing components ignore it.
  renderer: WebGPURenderer;
}
```

- [ ] **Step 2: Pass the renderer through in App.ts**

In `src/App.ts`, find the ComponentManager construction:

```ts
    this.components = new ComponentManager(
      {
        scene,
        store: this.store,
        paramStore,
        audioContext,
      },
      COMPONENTS,
    );
```

Replace with:

```ts
    this.components = new ComponentManager(
      {
        scene,
        store: this.store,
        paramStore,
        audioContext,
        renderer,
      },
      COMPONENTS,
    );
```

- [ ] **Step 3: Update `makeDeps` in the existing ComponentManager test**

Open `tests/render/ComponentManager.test.ts` and find the `makeDeps()` helper. Add a `renderer` field — a minimal fake is fine since existing component tests never call into it:

Search for `makeDeps`. Inside the returned object (in the same shape as `ComponentDeps`), append a `renderer` property. Example minimal fake:

```ts
    renderer: {
      computeAsync: () => Promise.resolve(),
    } as unknown as import("three/webgpu").WebGPURenderer,
```

(The exact placement matters less than the field being present; TypeScript will complain in `tsc --noEmit` if it's missing.)

- [ ] **Step 4: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only (`main.ts:186`, `BeatGridMarkers.ts:51`). No new errors. If new errors mention `renderer` missing, you have a test or use site that wasn't updated — fix that file.

- [ ] **Step 5: Run full suite**

```bash
npm test
```

Expected: existing tests pass (no test changes in this task that affect counts).

- [ ] **Step 6: Commit**

```bash
git add src/render/components/Component.ts src/App.ts tests/render/ComponentManager.test.ts
git commit -m "$(cat <<'EOF'
feat(components): expose WebGPURenderer in ComponentDeps

OrbitalCloud (next) needs renderer.computeAsync() for compute dispatch.
Existing components ignore the field.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Real spherical harmonics math (TDD on JS mirror)

**Files:**
- Create: `src/render/orbital/sh-basis.ts`
- Create: `tests/render/sh-basis.test.ts`

Two implementations of `evalSh(coefs, x, y, z) → scalar` that sums `c_lm · Y_l^m(x̂, ŷ, ẑ)` over l=0..3 (16 terms). Both share the same closed-form polynomials in unit-vector components (x/r, y/r, z/r). The JS version is for tests; the TSL version is for the compute shader. Identical layout, identical numerics.

Real-SH coefficient layout (constant across the codebase):

| Index | (l, m) | Y_l^m formula (in unit-vector components) |
|---|---|---|
| 0 | (0, 0) | (1/2)√(1/π) |
| 1 | (1, -1) | √(3/4π) · y |
| 2 | (1, 0) | √(3/4π) · z |
| 3 | (1, 1) | √(3/4π) · x |
| 4 | (2, -2) | (1/2)√(15/π) · x·y |
| 5 | (2, -1) | (1/2)√(15/π) · y·z |
| 6 | (2, 0) | (1/4)√(5/π) · (3z² − 1) |
| 7 | (2, 1) | (1/2)√(15/π) · x·z |
| 8 | (2, 2) | (1/4)√(15/π) · (x² − y²) |
| 9 | (3, -3) | (1/4)√(35/2π) · y·(3x² − y²) |
| 10 | (3, -2) | (1/2)√(105/π) · x·y·z |
| 11 | (3, -1) | (1/4)√(21/2π) · y·(5z² − 1) |
| 12 | (3, 0) | (1/4)√(7/π) · z·(5z² − 3) |
| 13 | (3, 1) | (1/4)√(21/2π) · x·(5z² − 1) |
| 14 | (3, 2) | (1/4)√(105/π) · z·(x² − y²) |
| 15 | (3, 3) | (1/4)√(35/2π) · x·(x² − 3y²) |

- [ ] **Step 1: Write the failing tests**

Create `tests/render/sh-basis.test.ts`:

```ts
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
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run tests/render/sh-basis.test.ts
```

Expected: all 9 tests fail with `Cannot find module '../../src/render/orbital/sh-basis'`.

- [ ] **Step 3: Create the orbital directory**

```bash
mkdir -p src/render/orbital
```

- [ ] **Step 4: Implement `evalShJs`**

Create `src/render/orbital/sh-basis.ts`:

```ts
// Real spherical harmonics for l = 0..3 (16 terms total).
//
// Coefficient layout (16 entries, both for JS and TSL paths):
//   0:  (l=0, m= 0)  — s
//   1:  (l=1, m=-1)  — p_y
//   2:  (l=1, m= 0)  — p_z
//   3:  (l=1, m= 1)  — p_x
//   4:  (l=2, m=-2)  — d_xy
//   5:  (l=2, m=-1)  — d_yz
//   6:  (l=2, m= 0)  — d_z²
//   7:  (l=2, m= 1)  — d_zx
//   8:  (l=2, m= 2)  — d_x²−y²
//   9:  (l=3, m=-3)
//  10:  (l=3, m=-2)
//  11:  (l=3, m=-1)
//  12:  (l=3, m= 0)
//  13:  (l=3, m= 1)
//  14:  (l=3, m= 2)
//  15:  (l=3, m= 3)
//
// Closed-form coefficients precomputed as constants. Polynomials are evaluated
// on the input (x, y, z) directly — caller normalizes if they want unit-vector
// inputs (we don't, to save a sqrt when caller already has it).

export const SH_COUNT = 16;

const N00 = 0.5 * Math.sqrt(1 / Math.PI);
const N1  = Math.sqrt(3 / (4 * Math.PI));
const N2A = 0.5 * Math.sqrt(15 / Math.PI);   // for m = ±2, ±1
const N20 = 0.25 * Math.sqrt(5 / Math.PI);    // for m = 0
const N22 = 0.25 * Math.sqrt(15 / Math.PI);
const N33 = 0.25 * Math.sqrt(35 / (2 * Math.PI));
const N32 = 0.25 * Math.sqrt(105 / Math.PI);  // for d at l=3 m=±2
const N31 = 0.25 * Math.sqrt(21 / (2 * Math.PI));
const N30 = 0.25 * Math.sqrt(7 / Math.PI);
const N3_M2 = 0.5 * Math.sqrt(105 / Math.PI); // for l=3 m=-2 (different by factor of 2)

export function evalShJs(coefs: Float32Array, x: number, y: number, z: number): number {
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;

  // l = 0
  let s =
    coefs[0]  * N00 +

  // l = 1
    coefs[1]  * N1  * y +
    coefs[2]  * N1  * z +
    coefs[3]  * N1  * x +

  // l = 2
    coefs[4]  * N2A * x * y +
    coefs[5]  * N2A * y * z +
    coefs[6]  * N20 * (3 * zz - 1) +
    coefs[7]  * N2A * x * z +
    coefs[8]  * N22 * (xx - yy) +

  // l = 3
    coefs[9]  * N33 * y * (3 * xx - yy) +
    coefs[10] * N3_M2 * x * y * z +
    coefs[11] * N31 * y * (5 * zz - 1) +
    coefs[12] * N30 * z * (5 * zz - 3) +
    coefs[13] * N31 * x * (5 * zz - 1) +
    coefs[14] * N32 * z * (xx - yy) +
    coefs[15] * N33 * x * (xx - 3 * yy);

  return s;
}
```

- [ ] **Step 5: Run tests, verify pass**

```bash
npx vitest run tests/render/sh-basis.test.ts
```

Expected: all 9 pass.

- [ ] **Step 6: Add the TSL mirror in the same file**

Append to `src/render/orbital/sh-basis.ts`:

```ts
import { Fn, float } from "three/tsl";

// TSL mirror of evalShJs. Takes a storage-backed coefficient array (indexable
// by `coefs.element(i)`) and three vec3 components, returns a TSL float node.
//
// Called from the compute kernel; see OrbitalCloud.ts. The closed-form
// constants are inlined as floats — TSL doesn't have a constant-folder we
// can rely on but the WGSL compiler does.
//
// IMPORTANT: coefs must be a uniform array (NOT instancedArray), since every
// particle reads the same 16 coefficients. Pass via `uniformArray(16, "float")`
// and update on each frame from the params bag.
export const evalShTsl = /*@__PURE__*/ Fn(
  ([coefs, x, y, z]: [any, any, any, any]) => {
    const xx = x.mul(x);
    const yy = y.mul(y);
    const zz = z.mul(z);

    // Constants — re-declared inside the Fn so they fold into the shader.
    const n00 = float(N00);
    const n1  = float(N1);
    const n2a = float(N2A);
    const n20 = float(N20);
    const n22 = float(N22);
    const n33 = float(N33);
    const n32 = float(N32);
    const n31 = float(N31);
    const n30 = float(N30);
    const n3m2 = float(N3_M2);

    return coefs.element(0).mul(n00)
      .add(coefs.element(1).mul(n1).mul(y))
      .add(coefs.element(2).mul(n1).mul(z))
      .add(coefs.element(3).mul(n1).mul(x))
      .add(coefs.element(4).mul(n2a).mul(x).mul(y))
      .add(coefs.element(5).mul(n2a).mul(y).mul(z))
      .add(coefs.element(6).mul(n20).mul(zz.mul(3).sub(1)))
      .add(coefs.element(7).mul(n2a).mul(x).mul(z))
      .add(coefs.element(8).mul(n22).mul(xx.sub(yy)))
      .add(coefs.element(9).mul(n33).mul(y).mul(xx.mul(3).sub(yy)))
      .add(coefs.element(10).mul(n3m2).mul(x).mul(y).mul(z))
      .add(coefs.element(11).mul(n31).mul(y).mul(zz.mul(5).sub(1)))
      .add(coefs.element(12).mul(n30).mul(z).mul(zz.mul(5).sub(3)))
      .add(coefs.element(13).mul(n31).mul(x).mul(zz.mul(5).sub(1)))
      .add(coefs.element(14).mul(n32).mul(z).mul(xx.sub(yy)))
      .add(coefs.element(15).mul(n33).mul(x).mul(xx.sub(yy.mul(3))));
  },
);
```

(The `any` types are intentional — TSL's generated types are awkward to express in TypeScript; the runtime is duck-typed. Use `// @ts-expect-error` only as a last resort.)

- [ ] **Step 7: Re-run tests (the TSL helper shouldn't break the JS tests)**

```bash
npx vitest run tests/render/sh-basis.test.ts
```

Expected: still 9 pass. (TSL import is dead code at test time but shouldn't error — `three/tsl` is available because it's an existing dep.)

- [ ] **Step 8: Commit**

```bash
git add src/render/orbital/sh-basis.ts tests/render/sh-basis.test.ts
git commit -m "$(cat <<'EOF'
feat(orbital): real spherical harmonics l=0..3 (JS + TSL)

16 real-SH terms in the canonical (l,m) layout: s, p_(yzx), d_(xy,yz,z²,zx,x²−y²),
and seven f. JS evalShJs is unit-tested at known-direction tabulated values; TSL
evalShTsl is the same math expressed via three/tsl node ops for use in the
upcoming OrbitalCloud compute shader.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Radial function math (TDD on JS mirror)

**Files:**
- Create: `src/render/orbital/radial.ts`
- Create: `tests/render/radial.test.ts`

R_n(r) = L_{n-1}(2r/n) · exp(-r/n), for principal quantum number n ∈ {1, 2, 3, 4}. L_k is the simple (non-associated) Laguerre polynomial of degree k. These are NOT the normalized hydrogen radial functions — we drop the normalization constant since visualization only cares about relative density.

L_0(x) = 1
L_1(x) = 1 − x
L_2(x) = 1 − 2x + x²/2
L_3(x) = 1 − 3x + 3x²/2 − x³/6

- [ ] **Step 1: Write the failing tests**

Create `tests/render/radial.test.ts`:

```ts
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
    for (const n of [1, 2, 3, 4]) {
      expect(Math.abs(evalRadialJs(100, n))).toBeLessThan(1e-10);
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
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run tests/render/radial.test.ts
```

Expected: 9 tests fail with `Cannot find module '../../src/render/orbital/radial'`.

- [ ] **Step 3: Implement `evalRadialJs`**

Create `src/render/orbital/radial.ts`:

```ts
// Hydrogen radial functions for principal quantum number n ∈ {1..4}, without
// the physics normalization constant (which doesn't affect visualization).
//
//   R_n(r) = L_{n-1}(2r/n) · exp(-r/n)
//
// where L_k is the simple Laguerre polynomial of degree k:
//   L_0(x) = 1
//   L_1(x) = 1 - x
//   L_2(x) = 1 - 2x + x²/2
//   L_3(x) = 1 - 3x + 3x²/2 - x³/6
//
// Higher n → more radial nodes (zero-density shells). Sign flips at each node.
// Visualization picks up the radial sign via the SH-side sign(ψ) read-out.
//
// Unknown n falls through to R_1; the shader does the same to keep behaviour
// in lockstep.

export function evalRadialJs(r: number, n: number): number {
  switch (n) {
    case 1: {
      // L_0(x) = 1; R_1 = exp(-r)
      return Math.exp(-r);
    }
    case 2: {
      const x = r;            // 2r/n = 2r/2 = r
      const lag = 1 - x;
      return lag * Math.exp(-r / 2);
    }
    case 3: {
      const x = (2 * r) / 3;
      const lag = 1 - 2 * x + (x * x) / 2;
      return lag * Math.exp(-r / 3);
    }
    case 4: {
      const x = r / 2;        // 2r/n = r/2
      const lag = 1 - 3 * x + (3 * x * x) / 2 - (x * x * x) / 6;
      return lag * Math.exp(-r / 4);
    }
    default: {
      // Unknown n → R_1 shape.
      return Math.exp(-r);
    }
  }
}
```

- [ ] **Step 4: Run tests, verify pass**

```bash
npx vitest run tests/render/radial.test.ts
```

Expected: 9 pass.

- [ ] **Step 5: Add the TSL mirror**

Append to `src/render/orbital/radial.ts`:

```ts
import { Fn, If, float } from "three/tsl";

// TSL mirror. n is a uniform float (we cast via toInt() inside the Fn).
// r is a TSL float node.
//
// We branch on the integer value of n using If chains — TSL doesn't have a
// native switch statement, but the chain compiles to a flat select tree in
// WGSL. The fall-through case is R_1 to match evalRadialJs.
export const evalRadialTsl = /*@__PURE__*/ Fn(([r, n]: [any, any]) => {
  const out = float(0).toVar();
  const ni = n.toInt();

  If(ni.equal(1), () => {
    out.assign(r.negate().exp());
  })
    .ElseIf(ni.equal(2), () => {
      const x = r;
      const lag = float(1).sub(x);
      out.assign(lag.mul(r.div(2).negate().exp()));
    })
    .ElseIf(ni.equal(3), () => {
      const x = r.mul(2 / 3);
      const lag = float(1).sub(x.mul(2)).add(x.mul(x).div(2));
      out.assign(lag.mul(r.div(3).negate().exp()));
    })
    .ElseIf(ni.equal(4), () => {
      const x = r.div(2);
      const lag = float(1)
        .sub(x.mul(3))
        .add(x.mul(x).mul(3).div(2))
        .sub(x.mul(x).mul(x).div(6));
      out.assign(lag.mul(r.div(4).negate().exp()));
    })
    .Else(() => {
      out.assign(r.negate().exp());
    });

  return out;
});
```

- [ ] **Step 6: Run tests (no regressions from TSL import)**

```bash
npx vitest run tests/render/radial.test.ts
```

Expected: 9 pass.

- [ ] **Step 7: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 8: Commit**

```bash
git add src/render/orbital/radial.ts tests/render/radial.test.ts
git commit -m "$(cat <<'EOF'
feat(orbital): hydrogen radial functions R_n(r), n=1..4 (JS + TSL)

Simple-Laguerre × exp(-r/n) form, no physics normalization (visualization
only cares about relative density). JS tested at tabulated values incl.
the R_2(r=1) node. TSL Fn mirrors via If/ElseIf chain on the n uniform.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: OrbitalCloud skeleton — static cloud render

**Files:**
- Create: `src/render/components/OrbitalCloud.ts`
- Modify: `src/render/components/index.ts`

Sets up the component with all 28 parameter schemas wired into ComponentManager, creates the position storage buffer, seeds positions uniformly in a sphere of radius `boundaryRadius` on the CPU, and renders them as a `THREE.Points` mesh. **No compute kernel yet** — particles are stationary. This task should produce a visible, static cloud of dots; subsequent tasks add dynamics.

- [ ] **Step 1: Implement OrbitalCloud (skeleton)**

Create `src/render/components/OrbitalCloud.ts`:

```ts
import {
  BufferGeometry,
  BufferAttribute,
  Points,
  Color,
} from "three";
import { PointsNodeMaterial } from "three/webgpu";
import { instancedArray, uniform } from "three/tsl";
import type { Component, ComponentDeps } from "./Component";

// ---- coefficient layout (must match sh-basis.ts) ----
const SH_COUNT = 16;
const SH_LABELS = [
  "c_0_0",
  "c_1_-1", "c_1_0", "c_1_1",
  "c_2_-2", "c_2_-1", "c_2_0", "c_2_1", "c_2_2",
  "c_3_-3", "c_3_-2", "c_3_-1", "c_3_0", "c_3_1", "c_3_2", "c_3_3",
];

// ---- discrete numParticles options ----
const PARTICLE_COUNTS = [10000, 100000, 500000, 1000000] as const;

function buildParamOpts(): Record<string, { min: number; max: number; step?: number }> {
  const opts: Record<string, { min: number; max: number; step?: number }> = {};
  for (const k of SH_LABELS) opts[k] = { min: -1, max: 1, step: 0.01 };
  opts.n              = { min: 0, max: 0, step: 0 };  // discrete; ignored
  opts.radialScale    = { min: 0.2, max: 5.0, step: 0.01 };
  opts.Bx             = { min: -1, max: 1, step: 0.01 };
  opts.By             = { min: -1, max: 1, step: 0.01 };
  opts.Bz             = { min: -1, max: 1, step: 0.01 };
  opts.diffusion      = { min: 0, max: 0.2, step: 0.001 };
  opts.driftGain      = { min: 0, max: 5, step: 0.01 };
  opts.precessionGain = { min: 0, max: 10, step: 0.01 };
  opts.timescale      = { min: 0, max: 3, step: 0.01 };
  opts.numParticles   = { min: 0, max: 0, step: 0 };  // discrete; ignored
  opts.pointSize      = { min: 0.5, max: 8, step: 0.1 };
  opts.boundaryRadius = { min: 1, max: 20, step: 0.1 };
  return opts;
}

function buildParamDefaults(): Record<string, number> {
  const d: Record<string, number> = {};
  for (const k of SH_LABELS) d[k] = 0;
  d.c_0_0 = 1.0;
  d.n              = 2;
  d.radialScale    = 1.0;
  d.Bx             = 0;
  d.By             = 1;
  d.Bz             = 0;
  d.diffusion      = 0.02;
  d.driftGain      = 1.0;
  d.precessionGain = 1.5;
  d.timescale      = 1.0;
  d.numParticles   = 100000;
  d.pointSize      = 2.0;
  d.boundaryRadius = 8.0;
  return d;
}

export class OrbitalCloud implements Component {
  static id = "orbitalCloud";
  static label = "Orbital Cloud";
  static paramPrefix = "orbitalCloud";
  static paramOpts = buildParamOpts();
  static paramDefaults = buildParamDefaults();
  static paramKinds = {
    numParticles: "discrete" as const,
    n: "discrete" as const,
  };
  static paramDiscreteOptions = {
    numParticles: PARTICLE_COUNTS as unknown as number[],
    n: [1, 2, 3, 4],
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  // private renderer: ComponentDeps["renderer"];   // used in Task 5
  // private paramStore: ComponentDeps["paramStore"]; // used in Task 10

  private numParticles: number;
  private points: Points | null = null;
  private material: PointsNodeMaterial | null = null;
  private disposed = false;

  // Storage handles (initialized in init()). Filled in across Tasks 4-6.
  private positionsStorage: any = null;
  private uniforms: any = null;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    // this.renderer = deps.renderer;
    // this.paramStore = deps.paramStore;
    this.params = params;
    this.numParticles = Math.round(params.numParticles);
    this.init();
  }

  private init(): void {
    const N = this.numParticles;
    const R = this.params.boundaryRadius;

    // CPU-seed positions uniformly in a ball of radius R.
    const positionsCpu = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      // Uniform-in-ball: sample direction on sphere, radius ∝ ∛(uniform).
      const u = Math.random();
      const r = R * Math.cbrt(u);
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const sp = Math.sin(phi);
      positionsCpu[i * 3 + 0] = r * sp * Math.cos(theta);
      positionsCpu[i * 3 + 1] = r * sp * Math.sin(theta);
      positionsCpu[i * 3 + 2] = r * Math.cos(phi);
    }

    // Wrap as a TSL storage buffer. (In Task 5 we use this from the compute
    // shader; for now it just feeds the points geometry.)
    this.positionsStorage = instancedArray(positionsCpu, "vec3");

    // Build the points geometry. The position attribute is bound from the
    // storage buffer via `toAttribute()` so the Points mesh and the compute
    // kernel share the same memory.
    const geom = new BufferGeometry();
    // A dummy attribute is required to satisfy three's draw count detection;
    // the actual positions come from positionNode below.
    geom.setAttribute("position", new BufferAttribute(new Float32Array(N * 3), 3));
    geom.setDrawRange(0, N);

    const mat = new PointsNodeMaterial();
    mat.positionNode = this.positionsStorage.toAttribute();
    mat.colorNode = uniform(new Color(1, 1, 1)) as unknown as any;
    mat.sizeNode = uniform(this.params.pointSize);
    mat.transparent = false;

    const pts = new Points(geom, mat);
    pts.frustumCulled = false; // particles can roam past initial bounds
    this.points = pts;
    this.material = mat;
    this.scene.add(pts);
  }

  update(): void {
    // No compute yet — particles are static. Task 5 wires the kernel.
    // pointSize is hot-readable via the uniform we set; we'd need to keep a
    // handle to update it in-flight, deferred to Task 5.
  }

  dispose(): void {
    this.disposed = true;
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.material?.dispose();
      this.points = null;
      this.material = null;
    }
  }
}
```

- [ ] **Step 2: Register OrbitalCloud in the registry**

Edit `src/render/components/index.ts`. Find:

```ts
import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import { ParticleView } from "./ParticleView";
import type { ComponentClass } from "./Component";

export const COMPONENTS: readonly ComponentClass[] = [
  DebugView,
  BoxView as unknown as ComponentClass,
  ParticleView as unknown as ComponentClass,
];
```

Replace with:

```ts
import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import { ParticleView } from "./ParticleView";
import { OrbitalCloud } from "./OrbitalCloud";
import type { ComponentClass } from "./Component";

export const COMPONENTS: readonly ComponentClass[] = [
  DebugView,
  BoxView as unknown as ComponentClass,
  ParticleView as unknown as ComponentClass,
  OrbitalCloud as unknown as ComponentClass,
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

Expected: existing tests pass.

- [ ] **Step 5: Manual browser sanity check**

```bash
npm run dev
```

In the browser:
- Open the panel. There should be a new "Orbital Cloud" folder with ~28 sliders and a discrete `numParticles` (10K/100K/500K/1M) and a discrete `n` (1..4).
- Disable DebugView, BoxView, ParticleView for clarity.
- Enable Orbital Cloud — 100K dots should render as a roughly-spherical cloud filling a sphere of radius 8 around the origin.
- Slide `numParticles` to 10K — NOTHING happens (reconfig wiring lands in Task 10).
- Slide `boundaryRadius` to 2 — NOTHING happens visually (init-only param; the cloud was seeded at construction).
- Toggle the component off and back on — clean teardown and rebuild at the current slider values.

If the panel section is empty, the component class wasn't picked up — check the registry order.
If you see "PointsNodeMaterial is not a constructor", three r170 doesn't export it under that name — try `SpriteNodeMaterial` instead, or import from `three/addons/...`. Note: a quick sanity check is `console.log(Object.keys(await import('three/webgpu')))` from the devtools.

- [ ] **Step 6: Commit**

```bash
git add src/render/components/OrbitalCloud.ts src/render/components/index.ts
git commit -m "$(cat <<'EOF'
feat(orbital): OrbitalCloud component skeleton (static cloud render)

28 sliders + 2 discrete params wired into ComponentManager. CPU-seeded
positions uniform in a ball of radius boundaryRadius, rendered as a
THREE.Points whose positionNode reads a TSL storage buffer. No compute
kernel yet — particles are stationary. Subsequent tasks add dynamics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Diffusion-only compute kernel

**Files:**
- Modify: `src/render/components/OrbitalCloud.ts`

Adds a TSL compute kernel that runs every frame, applying Brownian-motion diffusion to particle positions. No ψ, no precession, no drift yet — just `pos += diffusion · randn · sqrt(dt)`. After this task, the cloud should slowly spread out from its initial ball.

- [ ] **Step 1: Add the diffusion compute kernel**

Edit `src/render/components/OrbitalCloud.ts`. Add these imports at the top:

```ts
import { Fn, instanceIndex, hash, vec3, float, uniform } from "three/tsl";
```

(Remove the duplicate `uniform` import if `uniform` is already imported from a previous edit.)

Replace the constructor's commented-out `this.renderer` line with the active assignment:

```ts
  private renderer: ComponentDeps["renderer"];
  // (other private fields unchanged)

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.renderer = deps.renderer;
    this.params = params;
    this.numParticles = Math.round(params.numParticles);
    this.init();
  }
```

In `init()`, after creating `this.positionsStorage`, add the uniforms and compute kernel:

```ts
    // Uniforms updated each frame from the params bag in update().
    this.uniforms = {
      dt:        uniform(0.0),
      diffusion: uniform(this.params.diffusion),
      frame:     uniform(0),
    };

    // Compute kernel: pos += diffusion * randn(3) * sqrt(dt).
    // randn produced via 2x hash() of (instanceIndex, frame) per axis,
    // Box-Muller-transformed to gaussian. Cheap; the visual smoothing
    // hides any imperfections in randomness.
    const positions = this.positionsStorage;
    const dtU = this.uniforms.dt;
    const diffU = this.uniforms.diffusion;
    const frameU = this.uniforms.frame;

    this.updateKernel = Fn(() => {
      const p = positions.element(instanceIndex);

      // Three independent hashes per particle per frame, mapped to N(0,1).
      // We approximate with hash mapped from [0,1) → [-0.5, 0.5) ×√12,
      // giving uniform with stddev 1. Visually indistinguishable from a
      // proper gaussian at the diffusion magnitudes we use.
      const seed = float(instanceIndex).add(frameU.mul(0x9E3779B1));
      const rx = hash(seed.add(0)).sub(0.5).mul(Math.sqrt(12));
      const ry = hash(seed.add(1)).sub(0.5).mul(Math.sqrt(12));
      const rz = hash(seed.add(2)).sub(0.5).mul(Math.sqrt(12));

      const sigma = diffU.mul(dtU.sqrt());
      const dp = vec3(rx, ry, rz).mul(sigma);
      p.assign(p.add(dp));
    })().compute(this.numParticles);
```

Add the `updateKernel` field declaration near the top of the class (with the other private fields):

```ts
  private updateKernel: any = null;
  private frameCounter = 0;
```

- [ ] **Step 2: Drive the kernel from `update()`**

Replace the empty `update()` body with:

```ts
  update(): void {
    if (!this.updateKernel || this.disposed) return;
    // Frame-locked dt; clock-locked dt would be more precise but irrelevant
    // since the kernel just adds randn(3) per particle (statistical, not
    // deterministic).
    const dt = (1 / 60) * this.params.timescale;
    this.uniforms.dt.value = dt;
    this.uniforms.diffusion.value = this.params.diffusion;
    this.uniforms.frame.value = ++this.frameCounter;
    void this.renderer.computeAsync(this.updateKernel);
  }
```

- [ ] **Step 3: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only. If errors mention TSL `hash` or `instanceIndex` types, those are loose `any`-typed in three's TSL — cast accordingly.

- [ ] **Step 4: Run full suite**

```bash
npm test
```

Expected: existing tests pass.

- [ ] **Step 5: Manual browser sanity check**

```bash
npm run dev
```

- Enable Orbital Cloud (others off). Initial cloud at radius ~8.
- Within ~10 seconds the cloud should visibly expand outward as particles diffuse.
- Slide `diffusion` to 0.2 — expansion much faster.
- Slide `diffusion` to 0 — expansion stops, cloud holds shape.
- Slide `timescale` to 0 — kernel runs but with dt=0; no expansion.
- Slide `timescale` to 3 — expansion proceeds 3x faster.
- Slide `numParticles` to 10K — STILL doesn't take effect (Task 10).

Console errors? Common pitfalls:
- "uniform is not a function" → import path; should be `from "three/tsl"`.
- "computeAsync is not a function" → renderer is somehow not a WebGPURenderer; check Task 1 wiring.
- "Cannot read positionNode" → toAttribute() didn't bind to the buffer correctly; fall back to setting positionNode = positionsStorage.element(instanceIndex) inside an explicit Fn on the material.

- [ ] **Step 6: Commit**

```bash
git add src/render/components/OrbitalCloud.ts
git commit -m "$(cat <<'EOF'
feat(orbital): diffusion-only compute kernel

TSL Fn updates particle positions on the GPU each frame with a
Brownian-motion step (hash-based randn × diffusion × √dt). No ψ math
or precession yet — verifies the compute → render → uniform plumbing
in isolation before piling on the orbital math.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Sign(ψ) coloring

**Files:**
- Modify: `src/render/components/OrbitalCloud.ts`

Adds a second storage buffer holding `sign(ψ)` per particle. Each frame, the compute kernel evaluates ψ at the particle's current position and writes the sign. The Points material reads the sign and lerps between two colors (default warm-red for positive, cool-blue for negative). This task wires in `evalShTsl` and `evalRadialTsl` and all 16 SH-coefficient + n + radialScale uniforms.

After this task, the cloud should still expand from diffusion, but with two visible lobed color regions corresponding to the active orbital. (The shape doesn't follow the orbital yet — that's Task 7's drift.)

- [ ] **Step 1: Import the math helpers and add uniformArray**

Add to the imports in `OrbitalCloud.ts`:

```ts
import { uniformArray, mix, sign, varying, attribute } from "three/tsl";
import { evalShTsl } from "../orbital/sh-basis";
import { evalRadialTsl } from "../orbital/radial";
```

- [ ] **Step 2: Add the sign storage buffer and the SH uniform array**

In `init()`, alongside the position buffer creation, add:

```ts
    // Per-particle sign(ψ). Initialized to zeros; first frame overwrites.
    const signCpu = new Float32Array(N);
    this.signsStorage = instancedArray(signCpu, "float");
```

Add the corresponding field declaration:

```ts
  private signsStorage: any = null;
```

Then in the uniforms block, add the SH coefficients and the radial uniforms:

```ts
    // 16-element SH coefficient array, n (as float for the shader), radialScale,
    // and the per-axis B vector (added in Task 8). Updated each frame.
    const shCoefs = uniformArray(new Float32Array(SH_COUNT), "float");
    for (let i = 0; i < SH_COUNT; i++) {
      shCoefs.array[i] = this.params[SH_LABELS[i]];
    }

    this.uniforms = {
      dt:           uniform(0.0),
      diffusion:    uniform(this.params.diffusion),
      frame:        uniform(0),
      n:            uniform(this.params.n),
      radialScale:  uniform(this.params.radialScale),
      shCoefs,
    };
```

(Replace the previous `this.uniforms = {...}` from Task 5 — don't leave both.)

- [ ] **Step 3: Update the compute kernel to evaluate ψ and write sign**

Replace the body of the `Fn(() => {...})` in `init()` with:

```ts
    this.updateKernel = Fn(() => {
      const p = positions.element(instanceIndex);

      // --- Diffusion (unchanged from Task 5) ---
      const seed = float(instanceIndex).add(frameU.mul(0x9E3779B1));
      const rx = hash(seed.add(0)).sub(0.5).mul(Math.sqrt(12));
      const ry = hash(seed.add(1)).sub(0.5).mul(Math.sqrt(12));
      const rz = hash(seed.add(2)).sub(0.5).mul(Math.sqrt(12));
      const sigma = diffU.mul(dtU.sqrt());
      const dp = vec3(rx, ry, rz).mul(sigma);
      const pNew = p.add(dp);
      p.assign(pNew);

      // --- Evaluate ψ(pNew) for sign(ψ) coloring ---
      // r in spherical coords (rad), direction = pNew/r.
      const rLen = pNew.length().max(1e-6);
      const rScaled = rLen.div(this.uniforms.radialScale);
      const xh = pNew.x.div(rLen);
      const yh = pNew.y.div(rLen);
      const zh = pNew.z.div(rLen);
      const R = evalRadialTsl(rScaled, this.uniforms.n);
      const Y = evalShTsl(this.uniforms.shCoefs, xh, yh, zh);
      const psi = R.mul(Y);

      this.signsStorage.element(instanceIndex).assign(sign(psi));
    })().compute(this.numParticles);
```

- [ ] **Step 4: Wire the sign into the material color**

Replace the material setup in `init()`:

```ts
    const mat = new PointsNodeMaterial();
    mat.positionNode = this.positionsStorage.toAttribute();
    mat.colorNode = uniform(new Color(1, 1, 1)) as unknown as any;
    mat.sizeNode = uniform(this.params.pointSize);
    mat.transparent = false;
```

with:

```ts
    // Bipolar color: positive lobes warm (red), negative lobes cool (blue).
    // Particles with sign=0 (unevaluated; first frame) render as black —
    // they get overwritten the next frame.
    const POS_COLOR = vec3(0.95, 0.35, 0.25);
    const NEG_COLOR = vec3(0.25, 0.55, 0.95);

    const mat = new PointsNodeMaterial();
    mat.positionNode = this.positionsStorage.toAttribute();
    // sign ∈ {-1, 0, 1}. Map to t ∈ [0, 0.5, 1] for mix(NEG, POS, t).
    const signAttr = this.signsStorage.toAttribute();
    const t = signAttr.mul(0.5).add(0.5);
    mat.colorNode = mix(NEG_COLOR, POS_COLOR, t);
    mat.sizeNode = uniform(this.params.pointSize);
    mat.transparent = false;
```

- [ ] **Step 5: Push SH coefficient updates each frame**

In `update()`, after the existing uniform updates, add:

```ts
    // Push the 16 SH coefficients + n + radialScale into uniforms each frame.
    for (let i = 0; i < SH_COUNT; i++) {
      this.uniforms.shCoefs.array[i] = this.params[SH_LABELS[i]];
    }
    this.uniforms.n.value = this.params.n;
    this.uniforms.radialScale.value = this.params.radialScale;
```

- [ ] **Step 6: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 7: Run full suite**

```bash
npm test
```

Expected: existing tests pass.

- [ ] **Step 8: Manual browser sanity check**

```bash
npm run dev
```

- Enable Orbital Cloud (others off). Default `c_0_0 = 1`, rest zero → cloud should appear in a single color (red, positive, since Y_0^0 > 0 everywhere).
- Slide `c_0_0` to 0, `c_1_0` (p_z) to 1 — cloud should split into a red top half (z > 0) and blue bottom half (z < 0). Brownian diffusion still spreading.
- Slide `c_1_0` to 0, `c_2_0` (d_z²) to 1 — cloud should show three regions: red "donut" in the equatorial plane (negative since 3z²-1 < 0 for small z), red caps at top and bottom (positive since 3z²-1 > 0 for large |z|). Wait — that's mis-stated; Y_2^0 = (1/4)√(5/π)(3z²-1). At equator z=0 it's negative (3·0−1=−1). At pole z=±1 it's positive (3·1−1=2). So we should see BLUE equatorial band, RED polar caps. Verify this matches what you see.
- Slide `n` from 2 to 1 — radial profile changes (no nodes for n=1).
- Slide `n` to 4 — multiple radial node shells visible as concentric color flips.

If colors look wrong (e.g. all red or all blue, no division): the sign buffer isn't getting written. Check that `signs.element(i).assign(sign(psi))` is in the kernel body and that the material's colorNode reads `this.signsStorage.toAttribute()` (NOT `instancedArray.toAttribute()` of a different buffer).

- [ ] **Step 9: Commit**

```bash
git add src/render/components/OrbitalCloud.ts
git commit -m "$(cat <<'EOF'
feat(orbital): ψ evaluation + sign(ψ) bipolar coloring

Compute kernel evaluates ψ = R_n(r/scale) · Σ c_lm Y_l^m(p̂) at each
particle and writes sign(ψ) into a second storage buffer. Points
material reads the sign attr and lerps warm/cool colors. Cloud now
shows orbital lobed regions as the SH coefficient sliders change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Drift up ∇log|ψ|²

**Files:**
- Modify: `src/render/components/OrbitalCloud.ts`

Adds finite-difference gradient of `log|ψ|²` to the compute kernel. Walkers drift toward high-density regions, so the cloud takes on the orbital shape. After this task, sliding SH coefficients should mold the cloud into the lobed shapes (not just color them).

- [ ] **Step 1: Add driftGain uniform**

In `init()`, extend the uniforms object with `driftGain`:

```ts
    this.uniforms = {
      dt:           uniform(0.0),
      diffusion:    uniform(this.params.diffusion),
      frame:        uniform(0),
      n:            uniform(this.params.n),
      radialScale:  uniform(this.params.radialScale),
      driftGain:    uniform(this.params.driftGain),
      shCoefs,
    };
```

And in `update()`, push it each frame:

```ts
    this.uniforms.driftGain.value = this.params.driftGain;
```

- [ ] **Step 2: Add a helper to evaluate ψ inside the compute kernel**

The current kernel evaluates ψ once. The gradient needs 6 more evaluations (central differences). Refactor by inlining ψ into a TSL helper Fn near the top of the file:

```ts
const PSI_EPS = 1e-4;

// TSL Fn that returns ψ(pos). Used both for sign read-out and finite-difference
// gradient inside the compute kernel.
const evalPsi = Fn(([pos, shCoefs, n, radialScale]: [any, any, any, any]) => {
  const rLen = pos.length().max(1e-6);
  const rScaled = rLen.div(radialScale);
  const xh = pos.x.div(rLen);
  const yh = pos.y.div(rLen);
  const zh = pos.z.div(rLen);
  const R = evalRadialTsl(rScaled, n);
  const Y = evalShTsl(shCoefs, xh, yh, zh);
  return R.mul(Y);
});
```

Put this between the `SH_LABELS` constant and the `OrbitalCloud` class declaration. Then inside the compute kernel, replace the ad-hoc ψ evaluation with the helper.

- [ ] **Step 3: Replace the compute kernel body**

Replace the `Fn(() => { ... })` body with:

```ts
    const shCoefs = this.uniforms.shCoefs;
    const nU = this.uniforms.n;
    const rsU = this.uniforms.radialScale;
    const driftU = this.uniforms.driftGain;

    this.updateKernel = Fn(() => {
      const p = positions.element(instanceIndex);

      // --- Evaluate ψ(p) and gradient via central differences ---
      const psiC = evalPsi(p, shCoefs, nU, rsU);

      // 6 offset evaluations for ∇log|ψ|² via central differences.
      const eps = float(PSI_EPS);
      const psiXp = evalPsi(p.add(vec3(eps, 0, 0)), shCoefs, nU, rsU);
      const psiXm = evalPsi(p.add(vec3(eps.negate(), 0, 0)), shCoefs, nU, rsU);
      const psiYp = evalPsi(p.add(vec3(0, eps, 0)), shCoefs, nU, rsU);
      const psiYm = evalPsi(p.add(vec3(0, eps.negate(), 0)), shCoefs, nU, rsU);
      const psiZp = evalPsi(p.add(vec3(0, 0, eps)), shCoefs, nU, rsU);
      const psiZm = evalPsi(p.add(vec3(0, 0, eps.negate())), shCoefs, nU, rsU);

      // log|ψ|² with floor on |ψ|² to avoid log(0).
      const psiFloor = float(1e-6);
      const logSq = (v: any) => v.mul(v).max(psiFloor).log();
      const dx = logSq(psiXp).sub(logSq(psiXm)).div(eps.mul(2));
      const dy = logSq(psiYp).sub(logSq(psiYm)).div(eps.mul(2));
      const dz = logSq(psiZp).sub(logSq(psiZm)).div(eps.mul(2));
      const gradLog = vec3(dx, dy, dz);

      // --- Compose velocity ---
      const drift = gradLog.mul(driftU);

      // --- Diffusion (unchanged) ---
      const seed = float(instanceIndex).add(frameU.mul(0x9E3779B1));
      const rxn = hash(seed.add(0)).sub(0.5).mul(Math.sqrt(12));
      const ryn = hash(seed.add(1)).sub(0.5).mul(Math.sqrt(12));
      const rzn = hash(seed.add(2)).sub(0.5).mul(Math.sqrt(12));
      const sigma = diffU.mul(dtU.sqrt());
      const noiseStep = vec3(rxn, ryn, rzn).mul(sigma);

      // --- Update position ---
      const pNew = p.add(drift.mul(dtU)).add(noiseStep);
      p.assign(pNew);

      // --- Write sign(ψ) for coloring (re-evaluate at new position so color
      //     tracks the lobe the particle just stepped into). ---
      const psiNew = evalPsi(pNew, shCoefs, nU, rsU);
      this.signsStorage.element(instanceIndex).assign(sign(psiNew));
    })().compute(this.numParticles);
```

(Keep the `dtU`, `diffU`, `frameU` aliases at the top of `init()` exactly as in Task 5.)

- [ ] **Step 4: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 5: Run full suite**

```bash
npm test
```

Expected: existing tests pass.

- [ ] **Step 6: Manual browser sanity check**

```bash
npm run dev
```

- Enable Orbital Cloud, defaults (`c_0_0 = 1`, n=2). Within a few seconds the cloud should tighten from the initial uniform ball into the 2s shape (a slightly-puffy sphere with a dim ring at the node).
- Slide `c_0_0` to 0, `c_2_0` (d_z²) to 1. Within ~5 seconds the cloud should reshape into the d_z² peanut-with-donut: two red caps at top and bottom, blue equatorial donut.
- Slide `driftGain` to 0 — drift stops, cloud diffuses out.
- Slide `driftGain` to 5 — drift dominates, sharp orbital shape.
- Slide `c_2_0` back to 0 and turn up `c_3_0` (f, l=3 m=0) — should see a four-lobed dumbbell along z.

If the cloud explodes outward (particles flying to infinity), the gradient is wrong-signed. Check that `drift = gradLog.mul(driftU)` — driftU is the gain, gradLog is ALREADY pointing toward higher density (positive). If you accidentally negated it, particles flee.
If the cloud collapses to a point: the radial floor (`max(rLen, 1e-6)`) didn't kick in, or `radialScale` is being interpreted backwards. Double-check Step 3.

- [ ] **Step 7: Commit**

```bash
git add src/render/components/OrbitalCloud.ts
git commit -m "$(cat <<'EOF'
feat(orbital): drift walkers up ∇log|ψ|²

Finite-difference central gradient of log|ψ|² (6 extra ψ evals per
particle per frame) produces a drift velocity. Cloud now adopts the
orbital shape: sliders mold the geometry, not just the color regions.
driftGain slider controls strength.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Larmor precession around B

**Files:**
- Modify: `src/render/components/OrbitalCloud.ts`

Adds `Bx`, `By`, `Bz`, and `precessionGain` as uniforms. The compute kernel adds `precessionGain · cross(B, p)` to the velocity each step. Cloud rotates around the B axis at rate ≈ precessionGain · |B|.

- [ ] **Step 1: Add B and precessionGain uniforms**

In `init()`, extend the uniforms:

```ts
    this.uniforms = {
      dt:             uniform(0.0),
      diffusion:      uniform(this.params.diffusion),
      frame:          uniform(0),
      n:              uniform(this.params.n),
      radialScale:    uniform(this.params.radialScale),
      driftGain:      uniform(this.params.driftGain),
      precessionGain: uniform(this.params.precessionGain),
      Bx:             uniform(this.params.Bx),
      By:             uniform(this.params.By),
      Bz:             uniform(this.params.Bz),
      shCoefs,
    };
```

In `update()`:

```ts
    this.uniforms.precessionGain.value = this.params.precessionGain;
    this.uniforms.Bx.value = this.params.Bx;
    this.uniforms.By.value = this.params.By;
    this.uniforms.Bz.value = this.params.Bz;
```

- [ ] **Step 2: Add the precession term to the kernel**

In the compute kernel body (Task 7's version), find the `// --- Compose velocity ---` block:

```ts
      // --- Compose velocity ---
      const drift = gradLog.mul(driftU);
```

Replace with:

```ts
      // --- Compose velocity ---
      const drift = gradLog.mul(driftU);
      const B = vec3(this.uniforms.Bx, this.uniforms.By, this.uniforms.Bz);
      const precess = B.cross(p).mul(this.uniforms.precessionGain);
```

And in the `pNew = p.add(...)` line, add the precess contribution:

```ts
      const pNew = p.add(drift.add(precess).mul(dtU)).add(noiseStep);
```

- [ ] **Step 3: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 4: Manual browser sanity check**

```bash
npm run dev
```

- Enable Orbital Cloud, defaults. With `By = 1` (default) and `precessionGain = 1.5`, the cloud should slowly rotate around the y-axis.
- Set up a non-trivial orbital: `c_0_0 = 0`, `c_2_0 = 1` (d_z²). After cloud takes shape, watch — the d_z² peanut should spin around the y-axis. The lobes should sweep past.
- Slide `precessionGain` to 0 — rotation stops.
- Slide `precessionGain` to 10 — fast rotation (multiple revolutions per second).
- Set `By = 0`, `Bx = 1` — rotation axis flips to x-axis.
- Set `Bx = By = Bz = 0` — no rotation regardless of precessionGain.

Issue: cloud "swims" or shears rather than rigidly rotates? The cross product is `B × p`, NOT `p × B`. Check sign.
Issue: cloud spirals outward? Precession should preserve distance from origin (rotation is rigid). If it's spiraling, the precess contribution is being added without the cross product zeroing out the radial component — but cross(B, p) is by construction perpendicular to p, so this shouldn't happen unless something else is amplifying. Look for a missing `.mul(dtU)`.

- [ ] **Step 5: Commit**

```bash
git add src/render/components/OrbitalCloud.ts
git commit -m "$(cat <<'EOF'
feat(orbital): Larmor precession around audio-mappable B vector

precessionGain · (B × p) velocity term in the compute kernel.
Cloud rigidly rotates around the B axis at angular velocity
precessionGain · |B|. Three sliders Bx/By/Bz parameterize B.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Boundary respawn

**Files:**
- Modify: `src/render/components/OrbitalCloud.ts`

Adds a respawn rule: if `|p| > boundaryRadius`, the particle teleports to a uniform random point inside `boundaryRadius / 2`. Prevents the cloud from escaping when slider combinations produce outward drift, or when diffusion exceeds drift.

- [ ] **Step 1: Add boundaryRadius uniform**

In `init()`, extend uniforms:

```ts
      boundaryRadius: uniform(this.params.boundaryRadius),
```

In `update()`:

```ts
    this.uniforms.boundaryRadius.value = this.params.boundaryRadius;
```

- [ ] **Step 2: Add the respawn branch to the kernel**

In the compute kernel, after computing `pNew` but before `p.assign(pNew)`, insert:

```ts
      // Respawn if outside the boundary. Uniform in a ball of radius
      // boundaryRadius / 2 to keep respawned particles away from the wall.
      const bR = this.uniforms.boundaryRadius;
      const outsideMask = pNew.length().greaterThan(bR);
      const rSeed = float(instanceIndex).add(frameU.mul(0x85EBCA6B));
      const u1 = hash(rSeed.add(10));
      const u2 = hash(rSeed.add(11));
      const u3 = hash(rSeed.add(12));
      const newR = bR.mul(0.5).mul(u1.pow(1 / 3));
      const theta = u2.mul(Math.PI * 2);
      const cosPhi = u3.mul(2).sub(1);
      const sinPhi = float(1).sub(cosPhi.mul(cosPhi)).sqrt();
      const reseeded = vec3(
        newR.mul(sinPhi).mul(theta.cos()),
        newR.mul(sinPhi).mul(theta.sin()),
        newR.mul(cosPhi),
      );
      const pFinal = outsideMask.select(reseeded, pNew);
      p.assign(pFinal);
```

Update the sign read-out at the end of the kernel to use `pFinal`:

```ts
      const psiNew = evalPsi(pFinal, shCoefs, nU, rsU);
      this.signsStorage.element(instanceIndex).assign(sign(psiNew));
```

- [ ] **Step 3: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 4: Manual browser sanity check**

```bash
npm run dev
```

- Enable Orbital Cloud. Set `driftGain = 0`, `diffusion = 0.2` — pure diffusion. Previously the cloud kept expanding forever; now it should plateau at a sphere of radius `boundaryRadius` because particles that escape get reseeded inside.
- Slide `boundaryRadius` from 8 down to 2 — cloud should shrink within a couple of seconds as escaping particles are reseeded in the smaller ball.
- Slide back to 8 — cloud relaxes back outward.
- With drift on (`driftGain = 1`, `c_0_0 = 1`, `c_3_3 = 0.5`), set `precessionGain = 0`. With diffusion lowered to 0.01: cloud should hold shape inside the boundary indefinitely.

Issue: cloud disappears? `select` arguments may be swapped (some three versions use `select(cond, ifTrue, ifFalse)`, others `select(cond, ifFalse, ifTrue)`). Try swapping.
Issue: visible "flicker" at the boundary as particles teleport? Expected; with 1M particles the per-particle flicker should be invisible.

- [ ] **Step 5: Commit**

```bash
git add src/render/components/OrbitalCloud.ts
git commit -m "$(cat <<'EOF'
feat(orbital): respawn walkers outside boundaryRadius

If |p_new| > boundaryRadius, teleport the particle to a uniform-in-ball
point of radius boundaryRadius / 2. Hash-derived from (instanceIndex,
frame). Keeps the cloud bounded regardless of diffusion / drift balance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: numParticles live reconfig

**Files:**
- Modify: `src/render/components/OrbitalCloud.ts`

Subscribes to ParamStore for `orbitalCloud.numParticles`. On change: dispose storage buffers + Points mesh + compute kernel, re-init at the new size. Mirrors `ParticleView.rebuildBodies` in spirit.

- [ ] **Step 1: Add a paramStore field and subscribe in init**

In the constructor, uncomment the `paramStore` assignment:

```ts
  private paramStore: ComponentDeps["paramStore"];

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.renderer = deps.renderer;
    this.paramStore = deps.paramStore;
    this.params = params;
    this.numParticles = Math.round(params.numParticles);
    this.init();
  }
```

Add the unsub field:

```ts
  private storeUnsub: (() => void) | null = null;
```

At the very end of `init()` (after the `this.scene.add(pts)`), add:

```ts
    this.storeUnsub = this.paramStore.subscribe((key, value) => {
      if (this.disposed) return;
      if (key === "orbitalCloud.numParticles" && typeof value === "number") {
        const n = Math.round(value);
        if (n !== this.numParticles) {
          this.rebuild(n);
        }
      }
      if (key === "orbitalCloud.pointSize" && typeof value === "number") {
        // pointSize uniform was set via uniform(); update it.
        if (this.material) this.material.sizeNode = uniform(value);
      }
    });
```

- [ ] **Step 2: Add a `rebuild(n)` method**

Add as a private method on the class (after `dispose`):

```ts
  private rebuild(n: number): void {
    // Dispose current GPU resources.
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.material?.dispose();
      this.points = null;
      this.material = null;
    }
    // Drop the kernel handle — instancedArray instances are GC'd when
    // unreferenced.
    this.updateKernel = null;
    this.positionsStorage = null;
    this.signsStorage = null;

    this.numParticles = n;
    this.init();
  }
```

- [ ] **Step 3: Unsubscribe in dispose**

In `dispose()`, before the existing cleanup, add:

```ts
  dispose(): void {
    this.disposed = true;
    this.storeUnsub?.();
    this.storeUnsub = null;
    // (existing cleanup unchanged)
```

- [ ] **Step 4: Type-check**

```bash
npx tsc --noEmit
```

Expected: 2 pre-existing errors only.

- [ ] **Step 5: Manual browser sanity check**

```bash
npm run dev
```

- Enable Orbital Cloud at default 100K. Cloud renders.
- Drop `numParticles` to 10K — cloud visibly thins out. Brief pause for the rebuild is acceptable.
- Bump to 500K — cloud thickens.
- Bump to 1M — note framerate. Should still be playable (>30fps) on a modern GPU.
- Toggle the component off and back on — re-init works at current numParticles.
- Slide `pointSize` to 8 — points get bigger.

Issue: "Cannot read property of null" after rebuild → an old reference to `this.uniforms.shCoefs` etc. was held somewhere; ensure all references go through `this.uniforms.*` after rebuild reconstructs them.

- [ ] **Step 6: Commit**

```bash
git add src/render/components/OrbitalCloud.ts
git commit -m "$(cat <<'EOF'
feat(orbital): live numParticles reconfig + hot pointSize

Subscribe to ParamStore for orbitalCloud.numParticles; on change,
dispose Points + storage buffers + kernel and re-init at the new
size. pointSize updates the sizeNode in place.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final manual verification

**Files:** none (verification only)

End-to-end browser checkout against the spec's parameter table and behavioral expectations.

- [ ] **Step 1: Clean baseline**

```js
localStorage.removeItem("autocorrelation.params.v1")
```

(in the browser devtools, or run in an incognito tab)

- [ ] **Step 2: Start dev server**

```bash
npm run dev
```

Open the URL, click Mic or press T for test source.

- [ ] **Step 3: Default state inspection**

- All four component folders present: DebugView, BoxView, ParticleView, Orbital Cloud.
- Orbital Cloud folder enabled by default with the 28 sliders + 2 discrete selectors.
- Disable other components for focus.
- 100K particles, default c_0_0=1, n=2, By=1 → red-ish sphere slowly rotating around y, with a faint dim ring at the 2s radial node.

- [ ] **Step 4: SH coefficient sweep**

For each (l, m), zero everything else, set that coefficient to 1, observe:
- (0,0) s — uniform red ball (already verified).
- (1,-1) p_y — red top half along +y, blue bottom half. Dumbbell after drift settles.
- (1,0) p_z — red top, blue bottom (along z).
- (1,1) p_x — red along +x, blue along -x.
- (2,0) d_z² — red caps at top/bottom, blue equatorial donut.
- (2,2) d_x²-y² — four-lobed cloverleaf in the equatorial plane.
- (3,3) f, m=3 — six-lobed pinwheel around z.
- (3,0) f, m=0 — alternating red/blue caps along z.

Expected: each slider produces a recognizable orbital shape. Drift takes 2-5 seconds to settle.

- [ ] **Step 5: Radial sweep**

Reset to s only (`c_0_0=1`).
- n=1 → single fuzzy ball, no node.
- n=2 → ball with one dim node ring.
- n=3 → two node rings.
- n=4 → three node rings (you may need to slide `radialScale` down to see them all within the boundary).

Slide `radialScale` from 0.2 to 5.0 — orbital expands and contracts.

- [ ] **Step 6: B-field sweep**

Pick a non-trivial orbital (e.g. `c_2_0 = 1`, n=2). Default `By=1`:
- precessionGain = 0 → no rotation.
- precessionGain = 1.5 (default) → gentle rotation around y.
- precessionGain = 10 → fast rotation; visual blur from temporal aliasing.
- Bx=By=Bz=0 → no rotation regardless of gain.
- Bx=By=Bz=1 → rotation around (1,1,1) axis.

- [ ] **Step 7: Dynamics sliders**

- diffusion = 0 → cloud freezes (no noise), drift alone shapes it; can look "crystalline".
- diffusion = 0.2 → cloud is fuzzy/blurred, dense lobes obscured.
- driftGain = 0, diffusion > 0 → cloud spreads to a featureless sphere bounded by boundaryRadius.
- driftGain = 5 → very tight orbital shape.
- timescale = 0 → kernel runs but with dt=0; cloud frozen in current state.
- timescale = 3 → faster dynamics.

- [ ] **Step 8: numParticles + render**

- 10K → thin cloud, edges visible.
- 100K → default, balanced.
- 500K → dense.
- 1M → solid feel. Note framerate; if <30fps on this machine, that's expected on weaker GPUs.
- pointSize = 0.5 → barely-visible dots.
- pointSize = 8 → fat squares.
- boundaryRadius from 1 to 20 → cloud breathes in and out.

- [ ] **Step 9: Persistence + HMR**

- Reload the page. All slider values you set should persist. Re-enable audio.
- Edit `OrbitalCloud.ts` (e.g. tweak a default color), save. The component should HMR-rebuild without page reload.

- [ ] **Step 10: Coexistence**

- Enable DebugView alongside Orbital Cloud. Both render. DebugView's lines/bars draw over (or under) the cloud depending on z-order; that's expected.
- Enable ParticleView alongside Orbital Cloud. Both run; framerate stays acceptable at lower numParticles values.

- [ ] **Step 11: Edge cases**

- All SH coefficients = 0 → ψ = 0 everywhere → sign(ψ) = 0 → cloud renders in the lerp midpoint color (mix between warm and cool at t=0.5). Should be a neutral gray-ish color.
- Negative coefficients (e.g. c_0_0 = -1) → cloud is uniformly blue (negative s).
- Sum of opposing coefficients (e.g. c_1_0 = 1, c_1_-1 = 1) → tilted dumbbell.

- [ ] **Step 12: Final commit (if any fixes)**

If you fixed anything during this verification:

```bash
git add -p
git commit -m "fix(orbital): <describe>"
```

Otherwise skip.

---

## Done

When all 11 tasks check out:

- `npm test` passes (existing + 18 new tests: 9 SH + 9 radial).
- `npx tsc --noEmit` shows only the 2 pre-existing errors.
- Manual verification 12 steps pass.
- Git log shows ~10 new commits (1 per implementation task, +0/1 from verification).

End state: a fully toggleable `OrbitalCloud` component renders 1M GPU-resident walkers sampling the probability density of a hydrogen-orbital-like wavefunction. 28 sliders mold the cloud's shape, spin, and dynamics. Audio wiring is deferred to the future parameter router — when that lands, route audio features → orbitalCloud.* params.
