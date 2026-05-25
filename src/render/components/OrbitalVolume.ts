import { Mesh, BoxGeometry, BackSide, NormalBlending } from "three";
import { MeshBasicNodeMaterial } from "three/webgpu";
import {
  Fn, vec3, vec4, float, int, uniform, uniformArray, positionWorld,
  cameraPosition, Loop, If, max, min, exp, normalize,
} from "three/tsl";
import { evalPsi } from "../orbital/psi";
import type { Component, ComponentDeps } from "./Component";

// Discrete option sets.
const VOLUME_STEPS_OPTIONS = [16, 32, 48, 64, 96, 128] as const;
const SHADOW_STEPS_OPTIONS = [0, 4, 8, 16, 24] as const;

// Largest entry in each set. The TSL Loop is compiled with this upper bound;
// the actual per-frame iteration count is a uniform we break out of early.
// Compile-time bound keeps the shader stable so volumeSteps/shadowSteps can
// change live without a material rebuild.
const MAX_VOLUME_STEPS = 128;
const MAX_SHADOW_STEPS = 24;

// SH coefficient layout — MUST match sh-basis.ts SH_COUNT (16). Duplicated
// here as a constant so we can build the uniform array; not re-exported
// because callers don't need it.
const SH_COUNT = 16;
const SH_LABELS = [
  "c_0_0",
  "c_1_-1", "c_1_0", "c_1_1",
  "c_2_-2", "c_2_-1", "c_2_0", "c_2_1", "c_2_2",
  "c_3_-3", "c_3_-2", "c_3_-1", "c_3_0", "c_3_1", "c_3_2", "c_3_3",
];

// Hardcoded light direction. Same axis as the cubes' lambert in
// OrbitalCloud.ts. Promoting to a shared param is out of scope for v1.
const LIGHT_DIR = vec3(0.408, 0.866, 0.306);

// Matplotlib coolwarm endpoints — same colormap as OrbitalCloud cubes/splats
// so volume + particle views read as the same orbital, just rendered
// differently.
const COOL = vec3(0.230, 0.299, 0.754);
const MID  = vec3(0.865, 0.865, 0.865);
const WARM = vec3(0.706, 0.016, 0.150);

function buildParamOpts(): Record<string, { min: number; max: number; step?: number }> {
  return {
    volumeSteps:  { min: 0, max: 0, step: 0 },  // discrete; ignored
    shadowSteps:  { min: 0, max: 0, step: 0 },  // discrete; ignored
    density:      { min: 0.1, max: 500, step: 0.1 },
    boundsRadius: { min: 1, max: 20, step: 0.1 },
  };
}

function buildParamDefaults(): Record<string, number> {
  return {
    volumeSteps:  48,
    shadowSteps:  8,
    density:      50,
    boundsRadius: 8,
  };
}

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
    // ---- Uniforms ----
    // SH coefs are loaded each frame from the OrbitalCloud param namespace.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const shCoefs = uniformArray(new Float32Array(SH_COUNT) as unknown as any[], "float");
    for (let i = 0; i < SH_COUNT; i++) {
      const v = this.paramStore.get(`orbitalCloud.${SH_LABELS[i]}`);
      shCoefs.array[i] = typeof v === "number" ? v : 0;
    }

    const nU = uniform(this.readShared("n", 2));
    const radialScaleU = uniform(this.readShared("radialScale", 1.0));
    const colorScaleU = uniform(this.readShared("colorScale", 10.0));

    const boundsRadiusU = uniform(this.params.boundsRadius);
    const densityU = uniform(this.params.density);
    // int uniforms drive the early-out gate inside the fragment loop.
    const volumeStepsU = uniform(Math.round(this.params.volumeSteps), "int");
    const shadowStepsU = uniform(Math.round(this.params.shadowSteps), "int");

    this.uniforms = {
      shCoefs, n: nU, radialScale: radialScaleU, colorScale: colorScaleU,
      boundsRadius: boundsRadiusU, density: densityU,
      volumeSteps: volumeStepsU, shadowSteps: shadowStepsU,
    };

    // ---- Geometry ----
    // Unit cube, scaled at runtime via mesh.scale so boundsRadius is live.
    // Rebuilding the geometry on every change would be wasteful.
    const geom = new BoxGeometry(1, 1, 1);

    // ---- Material ----
    const mat = new MeshBasicNodeMaterial();
    mat.side = BackSide;            // see fragment-shader comment below
    mat.transparent = true;
    mat.depthWrite = false;
    mat.blending = NormalBlending;
    mat.colorNode = this.buildColorNode();

    // ---- Mesh ----
    const mesh = new Mesh(geom, mat);
    mesh.scale.setScalar(this.params.boundsRadius * 2);
    mesh.frustumCulled = false;     // shader does its own bounds clip
    this.scene.add(mesh);

    this.mesh = mesh;
    this.material = mat;
  }

  // Read an orbitalCloud.* shared param, falling back if the key is not
  // registered (e.g. OrbitalCloud was removed from COMPONENTS).
  private readShared(localKey: string, fallback: number): number {
    try {
      const v = this.paramStore.get(`orbitalCloud.${localKey}`);
      return typeof v === "number" ? v : fallback;
    } catch {
      return fallback;
    }
  }

  // Fragment shader body.
  //
  // Strategy:
  //   - BackSide rendering: `positionWorld` is the far end of the ray; the
  //     near end is the camera. Clamping tNear to 0 starts the march at the
  //     camera when it sits inside the cube.
  //   - Slab-method ray-box intersect against [-R, R]^3 in world space (the
  //     unit cube is scaled by 2R at mesh level — see init()).
  //   - Step the ray, evaluate |ψ|² as density, front-to-back composite with
  //     premultiplied alpha. Saturated alpha breaks the work loop early.
  //   - Per-step shadow ray toward the light accumulates optical depth.
  //     shadowSteps == 0 skips the shadow loop entirely.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private buildColorNode(): any {
    const u = this.uniforms;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (Fn as any)(() => {
      const R = u.boundsRadius;
      const rayOrigin = cameraPosition;
      const rayDir = normalize(positionWorld.sub(cameraPosition));

      // Slab intersection with [-R, R]^3.
      const invDir = vec3(1, 1, 1).div(rayDir);
      const t1 = vec3(R.negate()).sub(rayOrigin).mul(invDir);
      const t2 = vec3(R).sub(rayOrigin).mul(invDir);
      const tMin = vec3(min(t1.x, t2.x), min(t1.y, t2.y), min(t1.z, t2.z));
      const tMax = vec3(max(t1.x, t2.x), max(t1.y, t2.y), max(t1.z, t2.z));
      const tNearRaw = max(max(tMin.x, tMin.y), tMin.z);
      const tFar = min(min(tMax.x, tMax.y), tMax.z);
      const tNear = max(tNearRaw, float(0));

      const segLen = (tFar.sub(tNear)).max(float(0));
      const stepsFloat = float(u.volumeSteps).max(float(1));
      const dt = segLen.div(stepsFloat);

      const accumColor = vec3(0, 0, 0).toVar();
      const accumAlpha = float(0).toVar();
      const tCurr = tNear.toVar();

      // Compile-time loop bound is MAX_VOLUME_STEPS; the body gates on the
      // runtime `volumeSteps` uniform and on accumAlpha < 0.99 so the step
      // count can change live without a material rebuild.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      Loop(int(MAX_VOLUME_STEPS), ({ i }: { i: any }) => {
        If(i.lessThan(u.volumeSteps).and(accumAlpha.lessThan(float(0.99))), () => {
          const p = rayOrigin.add(rayDir.mul(tCurr));
          const psi = evalPsi(p, u.shCoefs, u.n, u.radialScale);

          const dens = psi.mul(psi).mul(u.density).min(float(1)).max(float(0));
          const stepAlpha = float(1).sub(exp(dens.mul(dt).negate()));

          // Hot-cold sample color (algebraic-sigmoid normalization). Matches
          // OrbitalCloud's diverging colormap.
          const xCol = psi.mul(u.colorScale);
          const tNorm = xCol.div(xCol.abs().add(float(1)));
          const tPos = tNorm.max(float(0));
          const tNeg = tNorm.min(float(0)).abs();
          const sampleColor = MID
            .add(WARM.sub(MID).mul(tPos))
            .add(COOL.sub(MID).mul(tNeg))
            .toVar();

          // Self-shadow: march from p along LIGHT_DIR, accumulate optical
          // depth. shadowSteps == 0 falls through with no darkening.
          If(u.shadowSteps.greaterThan(int(0)), () => {
            const shadowDt = R.div(float(u.shadowSteps).max(float(1)));
            const shadowDens = float(0).toVar();
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            Loop(int(MAX_SHADOW_STEPS), ({ i: j }: { i: any }) => {
              If(j.lessThan(u.shadowSteps), () => {
                const q = p.add(LIGHT_DIR.mul(shadowDt.mul(float(j).add(float(1)))));
                const psiS = evalPsi(q, u.shCoefs, u.n, u.radialScale);
                shadowDens.addAssign(psiS.mul(psiS).mul(u.density).mul(shadowDt));
              });
            });
            const transmittance = exp(shadowDens.negate());
            // 60% shadowed / 40% ambient floor. Volume integration darkens
            // further on its own, so the ambient base is generous.
            sampleColor.assign(sampleColor.mul(transmittance.mul(float(0.6)).add(float(0.4))));
          });

          accumColor.addAssign(sampleColor.mul(stepAlpha).mul(float(1).sub(accumAlpha)));
          accumAlpha.addAssign(stepAlpha.mul(float(1).sub(accumAlpha)));
          tCurr.addAssign(dt);
        });
      });

      return vec4(accumColor, accumAlpha);
    })();
  }

  update(): void {
    if (this.disposed || !this.uniforms) return;
    const u = this.uniforms;

    for (let i = 0; i < SH_COUNT; i++) {
      const v = this.paramStore.get(`orbitalCloud.${SH_LABELS[i]}`);
      u.shCoefs.array[i] = typeof v === "number" ? v : 0;
    }
    u.n.value = this.readShared("n", 2);
    u.radialScale.value = this.readShared("radialScale", 1.0);
    u.colorScale.value = this.readShared("colorScale", 10.0);

    u.density.value = this.params.density;
    u.boundsRadius.value = this.params.boundsRadius;
    u.volumeSteps.value = Math.round(this.params.volumeSteps);
    u.shadowSteps.value = Math.round(this.params.shadowSteps);

    // Mesh scale tracks boundsRadius — geometry is unit-sized, see init().
    if (this.mesh) this.mesh.scale.setScalar(this.params.boundsRadius * 2);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.material?.dispose();
      this.mesh = null;
      this.material = null;
    }
    this.uniforms = null;
  }
}
