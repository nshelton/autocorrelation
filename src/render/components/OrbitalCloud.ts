import {
  Points, BufferGeometry, BufferAttribute, AdditiveBlending,
  InstancedMesh, BoxGeometry, PlaneGeometry,
} from "three";
import {
  PointsNodeMaterial, MeshBasicNodeMaterial, StorageBufferAttribute,
} from "three/webgpu";
import {
  Fn, instanceIndex, hash, vec3, vec4, float, storage,
  uniform, uniformArray, mix, positionLocal, normalWorld, dot, max,
} from "three/tsl";
import { evalShTsl } from "../orbital/sh-basis";
import { evalRadialTsl } from "../orbital/radial";
import type { Component, ComponentDeps } from "./Component";
import type { ParamStore } from "../../params/ParamStore";

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
// Low end is for experimenting with the heavier render modes (oriented splats,
// instanced cubes) without GTAO/MRT stalls. Default kept at 10000 for first-
// run experience; existing users keep their saved values.
const PARTICLE_COUNTS = [1000, 5000, 10000, 50000, 100000, 500000, 1000000] as const;

// ---- render mode options ----
// 0 = Points (1px additive, current default, cheapest)
// 1 = Oriented splats (PlaneGeometry quads aligned with ∇|ψ|², soft Gaussian
//     falloff, additive — tiles probability-density isosurfaces)
// 2 = Instanced cubes (opaque BoxGeometry with hand-rolled lambert; AO from
//     scene MRT pass darkens crevices, gives strongest 3D-structure cue)
const RENDER_MODES = [0, 1, 2] as const;

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
  opts.renderMode     = { min: 0, max: 0, step: 0 };  // discrete; ignored
  // Drives splat (mode 1) and cube (mode 2) size live via scaleUniform;
  // Points (mode 0) ignores it (WebGPU clamps to 1px). Range extends well
  // below 0.5 because splats stack additively — even sub-pixel splats are
  // visible at high particle counts.
  opts.pointSize      = { min: 0.05, max: 8, step: 0.05 };
  opts.boundaryRadius = { min: 1, max: 20, step: 0.1 };
  opts.lifetime       = { min: 0.5, max: 30, step: 0.1 };
  opts.colorScale     = { min: 0.1, max: 200, step: 0.1 };
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
  d.numParticles   = 10000;
  d.renderMode     = 0;
  d.pointSize      = 2.0;
  d.boundaryRadius = 8.0;
  d.lifetime       = 5.0;
  d.colorScale     = 10.0;
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
    renderMode: "discrete" as const,
  };
  static paramDiscreteOptions = {
    numParticles: PARTICLE_COUNTS as unknown as number[],
    n: [1, 2, 3, 4],
    renderMode: RENDER_MODES as unknown as number[],
  };
  static paramDiscreteLabels = {
    renderMode: ["Points", "Splats", "Cubes"],
    n: ["1 (1s)", "2 (2s/2p)", "3 (3s/3p/3d)", "4 (4s/4p/4d/4f)"],
  };
  // Each c_l_m: p=1/16 to be non-zero; if non-zero, 50/50 ±1. All-zero falls
  // back to c_0_0=1 (pure 1s — most common natural state anyway).
  static paramButtons = [
    {
      title: "Randomize SH",
      onClick: (store: ParamStore) => {
        const vals: Record<string, number> = {};
        let any = false;
        for (const k of SH_LABELS) {
          if (Math.random() < 1 / 16) {
            vals[k] = Math.random() < 0.5 ? -1 : 1;
            any = true;
          } else {
            vals[k] = 0;
          }
        }
        if (!any) vals.c_0_0 = 1;
        for (const k of SH_LABELS) store.set(`orbitalCloud.${k}`, vals[k]);
      },
    },
  ];

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private renderer: ComponentDeps["renderer"];
  private paramStore: ComponentDeps["paramStore"];
  private storeUnsub: (() => void) | null = null;

  private numParticles: number;
  // Holds Points (mode 0) or InstancedMesh (modes 1, 2). Both extend Object3D
  // so the scene-add / scene-remove plumbing is uniform; dispose differs and
  // is guarded by instanceof in rebuild()/dispose().
  private points: Points | InstancedMesh | null = null;
  private material: PointsNodeMaterial | MeshBasicNodeMaterial | null = null;
  private scaleUniform: any = null;
  // Storage handles (initialized in init()).
  private positionsStorage: any = null;
  private psiStorage: any = null;
  // Per-particle normalized ∇|ψ|² direction. Only used by mode 1 (oriented
  // splats) — the splat plane is laid perpendicular to this so quads tile
  // probability-density isosurfaces. Always written by the kernel; ignored
  // by modes 0 and 2.
  private normalsStorage: any = null;
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

    // Per-particle ψ value (signed, continuous). Initialized to zeros; first
    // frame overwrites. Material maps it through a tanh-normalized diverging
    // colormap, so amplitude — not just sign — drives the per-particle color.
    const psiCpu = new Float32Array(N);
    const psiAttr = new StorageBufferAttribute(psiCpu, 1);
    this.psiStorage = storage(psiAttr, "float", N);

    // Per-particle normalized ∇|ψ|² direction (vec3). Initialized to zero;
    // kernel writes safe.select(normalize(gradLog), +Z) each frame.
    const normalsCpu = new Float32Array(N * 3);
    const normalsAttr = new StorageBufferAttribute(normalsCpu, 3);
    this.normalsStorage = storage(normalsAttr, "vec3", N);

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
      colorScale:     uniform(this.params.colorScale),
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

      // --- Write normal (mode 1 splat orientation) ---
      // Same direction as ∇|ψ|² (log is monotonic, gradient direction is
      // preserved). Stored normalized; near-zero gradient falls back to +Z.
      // Captured at p (not pFinal) to avoid 6 extra evalPsi calls — the per-
      // frame position step is small relative to particle spacing, so the
      // visual lag is invisible.
      const gradLen = gradLog.length();
      const gradDir = gradLen.greaterThan(float(1e-6))
        .select(gradLog.div(gradLen.max(float(1e-6))), vec3(0, 0, 1));
      this.normalsStorage.element(instanceIndex).assign(gradDir);

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

      // --- Write ψ for coloring (re-evaluate at new position so color tracks
      //     the lobe the particle just stepped into). Signed, continuous —
      //     the material normalizes through tanh(colorScale * ψ). ---
      const psiNew = evalPsi(pFinal, shCoefsU, nU, rsU);
      this.psiStorage.element(instanceIndex).assign(psiNew);
    })().compute(this.numParticles);

    // Diverging "hot-cold" colormap (matplotlib coolwarm endpoints): deep
    // blue at -1, near-white at 0, deep red at +1. Normalization is
    // algebraic-sigmoid (k·ψ)/(1+|k·ψ|), same smooth-saturate shape as tanh
    // but built from TSL primitives (three's TSL has no tanh export).
    // colorScale controls how aggressively values saturate; ψ amplitudes
    // vary wildly across orbitals and shells, hence the live-tunable knob.
    const COOL = vec3(0.230, 0.299, 0.754);
    const MID  = vec3(0.865, 0.865, 0.865);
    const WARM = vec3(0.706, 0.016, 0.150);

    this.scaleUniform = uniform(this.params.pointSize * 0.01);

    const mode = Math.round(this.params.renderMode);
    let mesh: Points | InstancedMesh;
    let mat: PointsNodeMaterial | MeshBasicNodeMaterial;

    // Storage→attribute conversion for the render path. Points (mode 0) wants
    // PER-VERTEX (vertex index = particle index). InstancedMesh (modes 1, 2)
    // wants PER-INSTANCE (geometry has G vertices × N instances, and we want
    // each instance to read storage[instanceIndex], NOT storage[vIndex]).
    // `.toAttribute()` defaults to per-vertex; `.setInstanced(true)` is what
    // flags it as per-instance — without it cubes/splats pull their 8 (or 4)
    // vertices from 8 (or 4) random particle slots, producing garbage blobs.
    const wantInstanced = mode !== 0;
    const psiAttrNode = wantInstanced
      ? this.psiStorage.toAttribute().setInstanced(true)
      : this.psiStorage.toAttribute();

    // Hot-cold mapping shared by all 3 modes; only the underlying attribute's
    // instanced flag differs.
    const xCol = psiAttrNode.mul(this.uniforms.colorScale);
    const tNorm = xCol.div(xCol.abs().add(1));
    const absT = tNorm.abs();
    const warmSide = mix(MID, WARM, absT);
    const coolSide = mix(MID, COOL, absT);
    const rgbColor = tNorm.greaterThan(0).select(warmSide, coolSide);

    if (mode === 1) {
      // --- Mode 1: Oriented splats ---
      // PlaneGeometry quads laid in the plane perpendicular to ∇|ψ|², so
      // they tile probability-density isosurfaces. Soft Gaussian alpha
      // falloff; additive blending stacks them into volume-looking clouds.
      // No depth write, so GTAO doesn't see them (and the +Z geometry
      // normal would be wrong post-rotation anyway).
      const bMat = new MeshBasicNodeMaterial();
      // Per-instance reads (see psiAttrNode comment above for why .setInstanced).
      const center = this.positionsStorage.toAttribute().setInstanced(true);
      const normalAttr = this.normalsStorage.toAttribute().setInstanced(true);
      // Tangent basis from the per-instance normal. Cross with X axis,
      // falling back to Y when |normal.x| > 0.9 (cross with X degenerates).
      const useY = normalAttr.x.abs().greaterThan(0.9);
      const helper = useY.select(vec3(0, 1, 0), vec3(1, 0, 0));
      const tangent = normalAttr.cross(helper).normalize();
      const bitangent = normalAttr.cross(tangent);
      // PlaneGeometry vertex local is [-0.5, 0.5]² × {0}; map onto the
      // tangent plane around the per-instance center, scaled by size.
      const offset = tangent.mul(positionLocal.x.mul(this.scaleUniform))
        .add(bitangent.mul(positionLocal.y.mul(this.scaleUniform)));
      bMat.positionNode = center.add(offset);
      // Gaussian alpha falloff: r² ∈ [0, 0.5]; k=12 puts corners at ~e⁻⁶.
      const r2 = positionLocal.x.mul(positionLocal.x)
        .add(positionLocal.y.mul(positionLocal.y));
      const alpha = r2.mul(-12).exp();
      bMat.colorNode = vec4(rgbColor, alpha) as unknown as any;
      bMat.transparent = true;
      bMat.depthWrite = false;
      bMat.blending = AdditiveBlending;
      const planeGeom = new PlaneGeometry(1, 1);
      const im = new InstancedMesh(planeGeom, bMat, N);
      im.frustumCulled = false;
      mat = bMat;
      mesh = im;
    } else if (mode === 2) {
      // --- Mode 2: Instanced cubes ---
      // Opaque BoxGeometry, hand-rolled lambert from BoxView.ts:80 (the
      // comment there documents why MeshStandardNodeMaterial doesn't work
      // with InstancedMesh on r170). Cubes write proper geometry normals
      // to the MRT, so GTAO darkens dense regions — that's the 3D-structure
      // cue we're after. Particle count should stay modest (< 100k) to
      // avoid the GTAO/MRT stall hit by prior billboard attempts.
      const cMat = new MeshBasicNodeMaterial();
      // Per-instance read (see psiAttrNode comment above for why .setInstanced).
      const center = this.positionsStorage.toAttribute().setInstanced(true);
      // Unit cube scaled by scaleUniform in-shader → pointSize is live.
      cMat.positionNode = center.add(positionLocal.mul(this.scaleUniform));
      const lightDir = vec3(0.408, 0.866, 0.306);
      const ndotl = max(dot(normalWorld, lightDir), float(0.0));
      const lit = ndotl.mul(0.7).add(0.3);
      cMat.colorNode = vec4(rgbColor.mul(lit), 1.0) as unknown as any;
      const cubeGeom = new BoxGeometry(1, 1, 1);
      const im = new InstancedMesh(cubeGeom, cMat, N);
      im.frustumCulled = false;
      mat = cMat;
      mesh = im;
    } else {
      // --- Mode 0: Points (default) ---
      // THREE.Points: vertex-index IS particle-index (no instancing).
      // WebGPU clamps point primitives to 1px on Apple Silicon and many
      // other GPUs, so pointSize doesn't visibly resize — visual density
      // comes from sheer particle count + additive stacking. 1px fragments
      // keep total fragment work bounded even at 1M, so GTAO is happy.
      const pMat = new PointsNodeMaterial();
      pMat.positionNode = this.positionsStorage.toAttribute();
      pMat.colorNode = rgbColor as unknown as any;
      pMat.transparent = true;
      pMat.depthWrite = false;
      pMat.blending = AdditiveBlending;
      const geom = new BufferGeometry();
      geom.setAttribute("position", new BufferAttribute(new Float32Array(N * 3), 3));
      geom.setDrawRange(0, N);
      const pts = new Points(geom, pMat);
      pts.frustumCulled = false;
      mat = pMat;
      mesh = pts;
    }

    this.points = mesh;
    this.material = mat;
    this.scene.add(mesh);

    this.storeUnsub = this.paramStore.subscribe((key, value) => {
      if (this.disposed) return;
      if (key === "orbitalCloud.numParticles" && typeof value === "number") {
        const n = Math.round(value);
        if (n !== this.numParticles) {
          this.rebuild(n);
        }
      }
      if (key === "orbitalCloud.renderMode" && typeof value === "number") {
        // Render mode requires rebuilding the mesh + material; storage
        // buffers and the compute kernel are reused via init().
        this.rebuild(this.numParticles);
      }
      if (key === "orbitalCloud.pointSize" && typeof value === "number") {
        // Update the scale uniform in-place. Rebuilding the node would
        // require re-binding the shader; mutating .value is hot. Mode 1
        // and Mode 2 read scaleUniform live; Mode 0 (Points) ignores it.
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
    this.uniforms.colorScale.value = this.params.colorScale;
    void this.renderer.computeAsync(this.updateKernel);
  }

  private rebuild(n: number): void {
    this.teardown();
    this.numParticles = n;
    this.init();
  }

  // Shared teardown for rebuild() and dispose(). InstancedMesh has its own
  // .dispose() (releases instance matrix buffer); Points doesn't (just
  // inherits from Object3D) — guard with instanceof.
  //
  // Critical: also unsubscribe the param-store listener. init() re-subscribes,
  // so without this teardown each rebuild() (e.g. on renderMode change) leaks
  // a subscriber. Two+ subscribers all fire on the next renderMode change →
  // cascading rebuilds → WebGPU pipeline thrash → freeze.
  private teardown(): void {
    this.storeUnsub?.();
    this.storeUnsub = null;
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      if (this.points instanceof InstancedMesh) this.points.dispose();
      this.material?.dispose();
      this.points = null;
      this.material = null;
    }
    this.updateKernel = null;
    this.positionsStorage = null;
    this.psiStorage = null;
    this.normalsStorage = null;
    this.lifetimesStorage = null;
    this.scaleUniform = null;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.uniforms = null;
    this.teardown();
  }
}
