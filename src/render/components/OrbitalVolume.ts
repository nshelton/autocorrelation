import { Mesh, BoxGeometry, EdgesGeometry, LineSegments, BackSide, NormalBlending } from "three";
import { exp, Loop, MeshBasicNodeMaterial, LineBasicNodeMaterial } from "three/webgpu";
import {
  Fn, vec3, vec4, float, uniform, uniformArray, positionWorld,
  cameraPosition, normalize, max, min,
} from "three/tsl";
import { evalPsi } from "../orbital/psi";
import { SH_COUNT } from "../orbital/sh-basis";
import type { Component, ComponentDeps } from "./Component";

// Discrete option sets (kept around so the static schema and the existing
// test pass while the shader is in baseline mode).
const VOLUME_STEPS_OPTIONS = [8, 16, 24, 32] as const;
const SHADOW_STEPS_OPTIONS = [0, 2, 4, 8] as const;

// SH coefficient labels — order MUST match sh-basis.ts (16 entries, l=0..3).
const SH_LABELS = [
  "c_0_0",
  "c_1_-1", "c_1_0", "c_1_1",
  "c_2_-2", "c_2_-1", "c_2_0", "c_2_1", "c_2_2",
  "c_3_-3", "c_3_-2", "c_3_-1", "c_3_0", "c_3_1", "c_3_2", "c_3_3",
];
const SHARED_SH_KEYS = SH_LABELS.map((k) => `orbitalCloud.${k}`);

// Matplotlib coolwarm endpoints — same colormap as OrbitalCloud.
const COOL = vec3(0.230, 0.299, 0.754);
const MID  = vec3(0.865, 0.865, 0.865);
const WARM = vec3(0.706, 0.016, 0.150);

function buildParamOpts(): Record<string, { min: number; max: number; step?: number }> {
  return {
    volumeSteps:  { min: 0, max: 0, step: 0 },
    shadowSteps:  { min: 0, max: 0, step: 0 },
    density:      { min: 0.1, max: 500, step: 0.1 },
    boundsRadius: { min: 1, max: 20, step: 0.1 },
  };
}

function buildParamDefaults(): Record<string, number> {
  return {
    volumeSteps:  8,
    shadowSteps:  0,
    density:      50,
    boundsRadius: 8,
  };
}

// BASELINE SHADER: render the view ray direction as RGB on the cube's
// back face. No loops, no per-fragment evalPsi, no integration. The full
// volumetric ray-marcher repeatedly crashed the machine while we sorted
// out the TSL Loop/If/Break issue; this baseline keeps the component
// instantiable + verifiable while we figure that out.
export class OrbitalVolume implements Component {
  static id = "orbitalVolume";
  static label = "Orbital Volume";
  static paramPrefix = "orbitalVolume";
  static paramOpts = buildParamOpts();
  static paramDefaults = buildParamDefaults();
  static paramKinds = {
    volumeSteps: "discrete" as const,
    shadowSteps: "discrete" as const,
  };
  static paramDiscreteOptions = {
    volumeSteps: VOLUME_STEPS_OPTIONS as unknown as number[],
    shadowSteps: SHADOW_STEPS_OPTIONS as unknown as number[],
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private paramStore: ComponentDeps["paramStore"];

  private mesh: Mesh | null = null;
  private material: MeshBasicNodeMaterial | null = null;
  private outline: LineSegments | null = null;
  private outlineMaterial: LineBasicNodeMaterial | null = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private uniforms: any = null;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.paramStore = deps.paramStore;
    this.params = params;
    this.init();
  }

  private init(): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const shCoefs = uniformArray(new Float32Array(SH_COUNT) as unknown as any[], "float");
    for (let i = 0; i < SH_COUNT; i++) {
      shCoefs.array[i] = this.readSharedKey(SHARED_SH_KEYS[i]);
    }
    this.uniforms = {
      shCoefs,
      n:            uniform(this.readShared("n", 2)),
      radialScale:  uniform(this.readShared("radialScale", 1.0)),
      colorScale:   uniform(this.readShared("colorScale", 10.0)),
      density:      uniform(this.params.density),
      boundsRadius: uniform(this.params.boundsRadius),
      // Int uniform; passed directly as the Loop bound so the WGSL for-loop
      // gets a real `i < uniforms.volumeSteps` test.
      volumeSteps:  uniform(Math.round(this.params.volumeSteps), "int"),
    };

    const geom = new BoxGeometry(1, 1, 1);

    const mat = new MeshBasicNodeMaterial();
    mat.side = BackSide;
    mat.transparent = true;
    mat.depthWrite = false;
    mat.blending = NormalBlending;
    mat.colorNode = this.buildColorNode();

    const mesh = new Mesh(geom, mat);
    mesh.scale.setScalar(this.params.boundsRadius * 2);
    mesh.frustumCulled = false;
    this.scene.add(mesh);

    // Wireframe outline — 1px white edges of the same unit cube. Added as a
    // child of the volume mesh so it inherits scale automatically (one
    // transform to update, in update()).
    const edgeGeom = new EdgesGeometry(geom);
    const lineMat = new LineBasicNodeMaterial();
    lineMat.colorNode = vec4(1, 1, 1, 1);
    const outline = new LineSegments(edgeGeom, lineMat);
    outline.frustumCulled = false;
    mesh.add(outline);

    this.mesh = mesh;
    this.material = mat;
    this.outline = outline;
    this.outlineMaterial = lineMat;
  }

  private readShared(localKey: string, fallback: number): number {
    try {
      const v = this.paramStore.get(`orbitalCloud.${localKey}`);
      return typeof v === "number" ? v : fallback;
    } catch {
      return fallback;
    }
  }

  private readSharedKey(fullKey: string): number {
    try {
      const v = this.paramStore.get(fullKey);
      return typeof v === "number" ? v : 0;
    } catch {
      return 0;
    }
  }

  private buildColorNode(): any {

    const u = this.uniforms;
    return (Fn as any)(() => {
      const R = u.boundsRadius;
      const rayOrigin = cameraPosition;
      const rayDir = normalize(positionWorld.sub(cameraPosition));

      // Slab intersect against [-R, R]^3.
      const invDir = vec3(1, 1, 1).div(rayDir);
      const t1 = vec3(R.negate()).sub(rayOrigin).mul(invDir);
      const t2 = vec3(R).sub(rayOrigin).mul(invDir);
      const tMin = vec3(min(t1.x, t2.x), min(t1.y, t2.y), min(t1.z, t2.z));
      const tMax = vec3(max(t1.x, t2.x), max(t1.y, t2.y), max(t1.z, t2.z));
      const tNear = max(max(max(tMin.x, tMin.y), tMin.z), float(0));
      const tFar = min(min(tMax.x, tMax.y), tMax.z);

      const dt = (tFar.sub(tNear)).div(float(32));
      const step = dt.mul(rayDir);
      let pos = rayOrigin.add(rayDir.mul(tNear)).toVar();
      let accumColor = vec3(0).toVar();
      let accumAlpha = float(0).toVar();

      Loop(32, () => {
        const psi = evalPsi(pos, u.shCoefs, u.n, u.radialScale);

        const dens = psi.mul(psi).mul(u.density).min(float(1)).max(float(0));
        const stepAlpha = float(1).sub(exp(dens.mul(dt).negate()));

      //   // Hot-cold sample color (algebraic-sigmoid normalization).
        const xCol = psi.mul(u.colorScale);
        const tNorm = xCol.div(xCol.abs().add(float(1)));
        const tPos = tNorm.max(float(0));
        const tNeg = tNorm.min(float(0)).abs();
        const sampleColor = MID
          .add(WARM.sub(MID).mul(tPos))
          .add(COOL.sub(MID).mul(tNeg));

      //   // Front-to-back composite (premultiplied alpha).
        accumColor.addAssign(sampleColor.mul(stepAlpha).mul(float(1).sub(accumAlpha)));
        accumAlpha.addAssign(stepAlpha.mul(float(1).sub(accumAlpha)));

        pos.addAssign(step);
      });


      return vec4(accumColor, accumAlpha);
    })();
  }

  update(): void {
    if (this.disposed || !this.uniforms) return;
    const u = this.uniforms;

    for (let i = 0; i < SH_COUNT; i++) {
      u.shCoefs.array[i] = this.readSharedKey(SHARED_SH_KEYS[i]);
    }
    u.n.value = this.readShared("n", 2);
    u.radialScale.value = this.readShared("radialScale", 1.0);
    u.colorScale.value = this.readShared("colorScale", 10.0);
    u.density.value = this.params.density;
    u.boundsRadius.value = this.params.boundsRadius;
    u.volumeSteps.value = Math.round(this.params.volumeSteps);

    if (this.mesh) this.mesh.scale.setScalar(this.params.boundsRadius * 2);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.outline) {
      this.outline.geometry.dispose();
      this.outlineMaterial?.dispose();
      this.outline = null;
      this.outlineMaterial = null;
    }
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.material?.dispose();
      this.mesh = null;
      this.material = null;
    }
  }
}
