import {
  BufferGeometry,
  BufferAttribute,
  Points,
  Color,
} from "three";
import { PointsNodeMaterial, StorageBufferAttribute } from "three/webgpu";
import { storage, uniform } from "three/tsl";
import type { Component, ComponentDeps } from "./Component";

// ---- coefficient layout (must match sh-basis.ts) ----
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
  // Storage handles (initialized in init()). Filled in across Tasks 4-6.
  private positionsStorage: any = null;
  private uniforms: Record<string, unknown> | null = null;
  private disposed = false;

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

    // Wrap as a TSL storage buffer. StorageBufferAttribute marks the buffer
    // for GPU storage binding. (In Task 5 we drive this from the compute
    // shader; for now it just feeds the points geometry.)
    const posAttr = new StorageBufferAttribute(positionsCpu, 3);
    this.positionsStorage = storage(posAttr, "vec3", N);

    // Build the points geometry. The position attribute is bound from the
    // storage buffer via `toAttribute()` so the Points mesh and the compute
    // kernel share the same memory.
    const geom = new BufferGeometry();
    // A dummy attribute is required to satisfy three's draw count detection;
    // the actual positions come from positionNode below.
    geom.setAttribute("position", new BufferAttribute(new Float32Array(N * 3), 3));
    geom.setDrawRange(0, N);

    const mat = new PointsNodeMaterial();
    // toAttribute() is method-chained at runtime via addMethodChaining;
    // positionsStorage is typed `any` to avoid the missing TS declaration.
    mat.positionNode = this.positionsStorage.toAttribute();
    mat.colorNode = uniform(new Color(1, 1, 1)) as unknown as any;
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
    // No compute yet — particles are static. Task 5 wires the kernel and
    // populates this.uniforms for hot-param updates.
    void this.uniforms;
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
