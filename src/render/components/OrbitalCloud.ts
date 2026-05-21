import {
  Points, BufferGeometry, BufferAttribute, AdditiveBlending,
} from "three";
import { PointsNodeMaterial, StorageBufferAttribute } from "three/webgpu";
import {
  Fn, instanceIndex, hash, vec3, float, storage,
  uniform, uniformArray, mix, sign,
} from "three/tsl";
import { evalShTsl } from "../orbital/sh-basis";
import { evalRadialTsl } from "../orbital/radial";
import type { Component, ComponentDeps } from "./Component";

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

// ---- coefficient layout (must match sh-basis.ts) ----
const SH_LABELS = [
  "c_0_0",
  "c_1_-1", "c_1_0", "c_1_1",
  "c_2_-2", "c_2_-1", "c_2_0", "c_2_1", "c_2_2",
  "c_3_-3", "c_3_-2", "c_3_-1", "c_3_0", "c_3_1", "c_3_2", "c_3_3",
];
const SH_COUNT = SH_LABELS.length;

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
  opts.lifetime       = { min: 0.5, max: 30, step: 0.1 };
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
  d.lifetime       = 5.0;
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
  private renderer: ComponentDeps["renderer"];
  private paramStore: ComponentDeps["paramStore"];
  private storeUnsub: (() => void) | null = null;

  private numParticles: number;
  private points: Points | null = null;
  private material: PointsNodeMaterial | null = null;
  private scaleUniform: any = null;
  // Storage handles (initialized in init()). Filled in across Tasks 4-6.
  private positionsStorage: any = null;
  private signsStorage: any = null;
  private lifetimesStorage: any = null;
  private uniforms: any = null;
  private updateKernel: any = null;
  private frameCounter = 0;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.renderer = deps.renderer;
    this.paramStore = deps.paramStore;
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

    // Wrap as a TSL storage buffer. StorageBufferAttribute marks the buffer
    // for GPU storage binding. The compute kernel and the Points mesh share
    // the same backing allocation via toAttribute().
    const posAttr = new StorageBufferAttribute(positionsCpu, 3);
    this.positionsStorage = storage(posAttr, "vec3", N);

    // Per-particle sign(ψ). Initialized to zeros; first frame overwrites.
    const signCpu = new Float32Array(N);
    const signAttr = new StorageBufferAttribute(signCpu, 1);
    this.signsStorage = storage(signAttr, "float", N);

    // Per-particle lifetime counter (seconds). Staggered so respawns spread
    // across time rather than all firing on the same frame.
    const maxLt = this.params.lifetime;
    const lifetimesCpu = new Float32Array(N);
    for (let i = 0; i < N; i++) lifetimesCpu[i] = Math.random() * maxLt;
    const lifetimesAttr = new StorageBufferAttribute(lifetimesCpu, 1);
    this.lifetimesStorage = storage(lifetimesAttr, "float", N);

    // 16-element SH coefficient array, n (as float for the shader), radialScale.
    // Updated each frame from the params bag.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const shCoefs = uniformArray(new Float32Array(SH_COUNT) as unknown as any[], "float");
    for (let i = 0; i < SH_COUNT; i++) {
      shCoefs.array[i] = this.params[SH_LABELS[i]];
    }

    // Uniforms updated each frame from the params bag in update().
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
      boundaryRadius: uniform(this.params.boundaryRadius),
      lifetime:       uniform(this.params.lifetime),
    };

    // Compute kernel: drift up ∇log|ψ|² + diffusion * randn(3) * sqrt(dt).
    // Drift uses central finite differences (6 extra ψ evals per particle).
    // randn produced via hash() of (instanceIndex, frame) per axis,
    // Box-Muller-approximated: uniform [0,1) → [-0.5, 0.5) × √12 gives
    // variance 1. Visually indistinguishable from gaussian at these magnitudes.
    const positions = this.positionsStorage;
    const lifetimes = this.lifetimesStorage;
    const dtU = this.uniforms.dt;
    const diffU = this.uniforms.diffusion;
    const frameU = this.uniforms.frame;
    const shCoefsU = this.uniforms.shCoefs;
    const nU = this.uniforms.n;
    const rsU = this.uniforms.radialScale;
    const driftU = this.uniforms.driftGain;
    const lifetimeU = this.uniforms.lifetime;

    // Cast the callback to `any` — Fn's TS overloads require a Node return, but
    // compute kernels are side-effecting and return void at the JS level.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.updateKernel = (Fn as any)(() => {
      const p = positions.element(instanceIndex);

      // --- Evaluate ψ(p) and gradient via central differences ---
      // 6 offset evaluations for ∇log|ψ|² via central differences.
      const eps = float(PSI_EPS);
      const psiXp = evalPsi(p.add(vec3(eps, 0, 0)), shCoefsU, nU, rsU);
      const psiXm = evalPsi(p.add(vec3(eps.negate(), 0, 0)), shCoefsU, nU, rsU);
      const psiYp = evalPsi(p.add(vec3(0, eps, 0)), shCoefsU, nU, rsU);
      const psiYm = evalPsi(p.add(vec3(0, eps.negate(), 0)), shCoefsU, nU, rsU);
      const psiZp = evalPsi(p.add(vec3(0, 0, eps)), shCoefsU, nU, rsU);
      const psiZm = evalPsi(p.add(vec3(0, 0, eps.negate())), shCoefsU, nU, rsU);

      // log|ψ|² with floor on |ψ|² to avoid log(0).
      const psiFloor = float(1e-6);
      const logSq = (v: any) => v.mul(v).max(psiFloor).log();
      const dx = logSq(psiXp).sub(logSq(psiXm)).div(eps.mul(2));
      const dy = logSq(psiYp).sub(logSq(psiYm)).div(eps.mul(2));
      const dz = logSq(psiZp).sub(logSq(psiZm)).div(eps.mul(2));
      const gradLog = vec3(dx, dy, dz);

      // --- Compose velocity ---
      const drift = gradLog.mul(driftU);
      const B = vec3(this.uniforms.Bx, this.uniforms.By, this.uniforms.Bz);
      const precess = B.cross(p).mul(this.uniforms.precessionGain);

      // --- Diffusion ---
      // Seed mixing: instanceIndex (up to ~1e6) + frame * small prime kept in
      // float32 precision range. Large prime multipliers (0x9E3779B1 etc.)
      // produce >1e9 magnitudes per frame; float32 mantissa loses the
      // instanceIndex contribution and every particle gets the same hash
      // input. We wrap frame at 65536 and multiply by a small prime so the
      // combined seed stays under ~1e7.
      const fWrap = frameU.mod(65536);
      const seed = float(instanceIndex).add(fWrap.mul(13.37));
      const rxn = hash(seed.add(0)).sub(0.5).mul(Math.sqrt(12));
      const ryn = hash(seed.add(1)).sub(0.5).mul(Math.sqrt(12));
      const rzn = hash(seed.add(2)).sub(0.5).mul(Math.sqrt(12));
      const sigma = diffU.mul(dtU.sqrt());
      const noiseStep = vec3(rxn, ryn, rzn).mul(sigma);

      // --- Update position ---
      const pNew = p.add(drift.add(precess).mul(dtU)).add(noiseStep);

      // --- Shared respawn position (used for both boundary and lifetime respawn).
      // Uniform in a ball of radius boundaryRadius / 2 to keep respawned
      // particles away from the wall.
      const bR = this.uniforms.boundaryRadius;
      // Decorrelate from the diffusion seed by using a different small prime
      // and a different frame-wrap modulus. Same precision rationale as above.
      const fWrapR = frameU.mod(32768);
      const rSeed = float(instanceIndex).mul(1.013).add(fWrapR.mul(7.919));
      const u1 = hash(rSeed.add(10));
      const u2 = hash(rSeed.add(11));
      const u3 = hash(rSeed.add(12));
      const newR = bR.mul(0.5).mul(u1.pow(float(1 / 3)));
      const theta = u2.mul(Math.PI * 2);
      const cosPhi = u3.mul(2).sub(1);
      const sinPhi = float(1).sub(cosPhi.mul(cosPhi)).sqrt();
      const reseeded = vec3(
        newR.mul(sinPhi).mul(theta.cos()),
        newR.mul(sinPhi).mul(theta.sin()),
        newR.mul(cosPhi),
      );

      // --- Lifetime tick ---
      const lt = lifetimes.element(instanceIndex);
      const ltNew = lt.sub(dtU);
      const expired = ltNew.lessThan(0);
      // Fresh lifetime: lifetime * (0.5..1.0) jitter, decorrelated from position seeds.
      const ltSeed = hash(seed.add(100));
      const freshLifetime = lifetimeU.mul(ltSeed.mul(0.5).add(0.5));

      // --- Compose final position and lifetime ---
      const outsideMask = pNew.length().greaterThan(bR);
      // TSL bool OR via float cast: WGSL rejects (bool + bool), so each
      // boolean is cast to f32 individually before the add.
      const shouldRespawn = outsideMask.select(float(1), float(0))
        .add(expired.select(float(1), float(0)))
        .greaterThan(0);
      const pFinal = shouldRespawn.select(reseeded, pNew);
      const ltFinal = shouldRespawn.select(freshLifetime, ltNew);

      p.assign(pFinal);
      lifetimes.element(instanceIndex).assign(ltFinal);

      // --- Write sign(ψ) for coloring (re-evaluate at new position so color
      //     tracks the lobe the particle just stepped into). ---
      const psiNew = evalPsi(pFinal, shCoefsU, nU, rsU);
      this.signsStorage.element(instanceIndex).assign(sign(psiNew));
    })().compute(this.numParticles);

    // Bipolar color: positive lobes warm (red), negative lobes cool (blue).
    // Particles with sign=0 (unevaluated; first frame) render as black —
    // they get overwritten the next frame.
    const POS_COLOR = vec3(0.95, 0.35, 0.25);
    const NEG_COLOR = vec3(0.25, 0.55, 0.95);

    // THREE.Points: one vertex per particle. The storage buffer has N
    // entries and the geometry has N vertices, so vertex-index IS particle-
    // index — `positionsStorage.toAttribute()` reads positions[i] for the
    // i-th point without any instancing confusion.
    //
    // Trade: WebGPU point primitives are clamped to 1px on Apple Silicon
    // and many other GPUs, so pointSize doesn't visibly resize particles.
    // The visual density comes from sheer particle count + additive
    // blending: many overlapping 1px hits → bright glow where density is
    // high.
    const mat = new PointsNodeMaterial();
    mat.positionNode = this.positionsStorage.toAttribute();
    const signAttrNode = this.signsStorage.toAttribute();
    const t = signAttrNode.mul(0.5).add(0.5);
    mat.colorNode = mix(NEG_COLOR, POS_COLOR, t) as unknown as any;
    // Keep scaleUniform for API symmetry (the live-update subscriber
    // mutates it); has no visible effect with point primitives.
    this.scaleUniform = uniform(this.params.pointSize * 0.01);

    // Additive blending: stacked particles brighten into a glow rather than
    // overwrite. depthWrite off so transparent draw order doesn't matter.
    // 1px point fragments keep total fragment work bounded even at 1M, so
    // this doesn't trigger the GTAO/MRT freeze we hit with billboards.
    mat.transparent = true;
    mat.depthWrite = false;
    mat.blending = AdditiveBlending;

    // Geometry: a dummy position attribute is required so three knows the
    // draw count. Actual positions come from positionNode above.
    const geom = new BufferGeometry();
    geom.setAttribute("position", new BufferAttribute(new Float32Array(N * 3), 3));
    geom.setDrawRange(0, N);
    const pts = new Points(geom, mat);
    pts.frustumCulled = false; // particles roam past the initial bounds
    this.points = pts;
    this.material = mat;
    this.scene.add(pts);

    this.storeUnsub = this.paramStore.subscribe((key, value) => {
      if (this.disposed) return;
      if (key === "orbitalCloud.numParticles" && typeof value === "number") {
        const n = Math.round(value);
        if (n !== this.numParticles) {
          this.rebuild(n);
        }
      }
      if (key === "orbitalCloud.pointSize" && typeof value === "number") {
        // Update the scale uniform in-place. Rebuilding the node would
        // require re-binding the shader; mutating .value is hot.
        if (this.scaleUniform) this.scaleUniform.value = value * 0.01;
      }
    });
  }

  update(): void {
    if (!this.updateKernel || this.disposed) return;
    // Frame-locked dt; clock-locked dt would be more precise but irrelevant
    // since the kernel just adds randn(3) per particle (statistical, not
    // deterministic).
    const dt = (1 / 60) * this.params.timescale;
    this.uniforms.dt.value = dt;
    this.uniforms.diffusion.value = this.params.diffusion;
    this.uniforms.frame.value = ++this.frameCounter;
    // Push the 16 SH coefficients + n + radialScale into uniforms each frame.
    for (let i = 0; i < SH_COUNT; i++) {
      this.uniforms.shCoefs.array[i] = this.params[SH_LABELS[i]];
    }
    this.uniforms.n.value = this.params.n;
    this.uniforms.radialScale.value = this.params.radialScale;
    this.uniforms.driftGain.value = this.params.driftGain;
    this.uniforms.precessionGain.value = this.params.precessionGain;
    this.uniforms.Bx.value = this.params.Bx;
    this.uniforms.By.value = this.params.By;
    this.uniforms.Bz.value = this.params.Bz;
    this.uniforms.boundaryRadius.value = this.params.boundaryRadius;
    this.uniforms.lifetime.value = this.params.lifetime;
    void this.renderer.computeAsync(this.updateKernel);
  }

  private rebuild(n: number): void {
    // Dispose current GPU resources. THREE.Points has no own .dispose();
    // its geometry and material do.
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
    this.lifetimesStorage = null;
    this.scaleUniform = null;

    this.numParticles = n;
    this.init();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.storeUnsub?.();
    this.storeUnsub = null;
    this.uniforms = null;
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.material?.dispose();
      this.points = null;
      this.material = null;
    }
  }
}
