import {
  BufferGeometry,
  BufferAttribute,
  Points,
} from "three";
import { PointsNodeMaterial, StorageBufferAttribute } from "three/webgpu";
import { Fn, instanceIndex, hash, vec3, float, storage, uniform, uniformArray, mix, sign } from "three/tsl";
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
  private renderer: ComponentDeps["renderer"];
  // private paramStore: ComponentDeps["paramStore"]; // used in Task 10

  private numParticles: number;
  private points: Points | null = null;
  private material: PointsNodeMaterial | null = null;
  // Storage handles (initialized in init()). Filled in across Tasks 4-6.
  private positionsStorage: any = null;
  private signsStorage: any = null;
  private uniforms: any = null;
  private updateKernel: any = null;
  private frameCounter = 0;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.renderer = deps.renderer;
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

    // Wrap as a TSL storage buffer. StorageBufferAttribute marks the buffer
    // for GPU storage binding. The compute kernel and the Points mesh share
    // the same backing allocation via toAttribute().
    const posAttr = new StorageBufferAttribute(positionsCpu, 3);
    this.positionsStorage = storage(posAttr, "vec3", N);

    // Per-particle sign(ψ). Initialized to zeros; first frame overwrites.
    const signCpu = new Float32Array(N);
    const signAttr = new StorageBufferAttribute(signCpu, 1);
    this.signsStorage = storage(signAttr, "float", N);

    // 16-element SH coefficient array, n (as float for the shader), radialScale.
    // Updated each frame from the params bag.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const shCoefs = uniformArray(new Float32Array(SH_COUNT) as unknown as any[], "float");
    for (let i = 0; i < SH_COUNT; i++) {
      shCoefs.array[i] = this.params[SH_LABELS[i]];
    }

    // Uniforms updated each frame from the params bag in update().
    this.uniforms = {
      dt:          uniform(0.0),
      diffusion:   uniform(this.params.diffusion),
      frame:       uniform(0),
      n:           uniform(this.params.n),
      radialScale: uniform(this.params.radialScale),
      driftGain:   uniform(this.params.driftGain),
      shCoefs,
    };

    // Compute kernel: drift up ∇log|ψ|² + diffusion * randn(3) * sqrt(dt).
    // Drift uses central finite differences (6 extra ψ evals per particle).
    // randn produced via hash() of (instanceIndex, frame) per axis,
    // Box-Muller-approximated: uniform [0,1) → [-0.5, 0.5) × √12 gives
    // variance 1. Visually indistinguishable from gaussian at these magnitudes.
    const positions = this.positionsStorage;
    const dtU = this.uniforms.dt;
    const diffU = this.uniforms.diffusion;
    const frameU = this.uniforms.frame;
    const shCoefsU = this.uniforms.shCoefs;
    const nU = this.uniforms.n;
    const rsU = this.uniforms.radialScale;
    const driftU = this.uniforms.driftGain;

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
      const psiNew = evalPsi(pNew, shCoefsU, nU, rsU);
      this.signsStorage.element(instanceIndex).assign(sign(psiNew));
    })().compute(this.numParticles);

    // Build the points geometry. The position attribute is bound from the
    // storage buffer via `toAttribute()` so the Points mesh and the compute
    // kernel share the same memory.
    const geom = new BufferGeometry();
    // A dummy attribute is required to satisfy three's draw count detection;
    // the actual positions come from positionNode below.
    geom.setAttribute("position", new BufferAttribute(new Float32Array(N * 3), 3));
    geom.setDrawRange(0, N);

    // Bipolar color: positive lobes warm (red), negative lobes cool (blue).
    // Particles with sign=0 (unevaluated; first frame) render as black —
    // they get overwritten the next frame.
    const POS_COLOR = vec3(0.95, 0.35, 0.25);
    const NEG_COLOR = vec3(0.25, 0.55, 0.95);

    const mat = new PointsNodeMaterial();
    // toAttribute() is method-chained at runtime via addMethodChaining;
    // positionsStorage is typed `any` to avoid the missing TS declaration.
    mat.positionNode = this.positionsStorage.toAttribute();
    // sign ∈ {-1, 0, 1}. Map to t ∈ [0, 0.5, 1] for mix(NEG, POS, t).
    const signAttrNode = this.signsStorage.toAttribute();
    const t = signAttrNode.mul(0.5).add(0.5);
    mat.colorNode = mix(NEG_COLOR, POS_COLOR, t) as unknown as any;
    // sizeNode exists at runtime but @types/three omits it from PointsNodeMaterial.
    (mat as any).sizeNode = uniform(this.params.pointSize);
    mat.transparent = false;

    const pts = new Points(geom, mat);
    pts.frustumCulled = false; // particles can roam past initial bounds
    this.points = pts;
    this.material = mat;
    this.scene.add(pts);
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
    void this.renderer.computeAsync(this.updateKernel);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
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
