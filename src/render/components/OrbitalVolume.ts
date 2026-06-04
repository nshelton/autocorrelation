import {
  Mesh, BoxGeometry, EdgesGeometry, LineSegments, BackSide, NormalBlending,
  PerspectiveCamera, RenderTarget, Vector2, Color, HalfFloatType,
} from "three";
import {
  exp, Loop, MeshBasicNodeMaterial, LineBasicNodeMaterial, QuadMesh,
} from "three/webgpu";
import {
  Fn, vec3, vec4, float, uniform, uniformArray, positionWorld,
  cameraPosition, normalize, max, min, mix, fract, texture, screenUV,
  screenCoordinate,
} from "three/tsl";
import { evalPsi } from "../orbital/psi";
import { SH_COUNT } from "../orbital/sh-basis";
import { LIGHT_DIR } from "../orbital/colormap";
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
const GREEN = /*@__PURE__*/ vec3(0.230, 0.99, 0.14);
const RED = /*@__PURE__*/ vec3(0.706, 0.016, 0.950);
const BLACK = /*@__PURE__*/ vec3(0, 0, 0);

// Halton(2, 3) sequence — the standard TAA / TAAU sub-pixel jitter offsets,
// centered to [-0.5, 0.5]. 16 samples; the camera projection cycles through
// them so successive frames sample different positions within each pixel.
// Low-discrepancy = quasi-blue-noise = good spatial spread without clumps.
const HALTON_LEN = 16;
const HALTON_JITTER = /*@__PURE__*/ (() => {
  const halton = (i: number, b: number): number => {
    let r = 0, f = 1 / b, idx = i;
    while (idx > 0) { r += f * (idx % b); idx = Math.floor(idx / b); f /= b; }
    return r;
  };
  const arr = new Float32Array(HALTON_LEN * 2);
  for (let i = 0; i < HALTON_LEN; i++) {
    arr[i * 2]     = halton(i + 1, 2) - 0.5;
    arr[i * 2 + 1] = halton(i + 1, 3) - 0.5;
  }
  return arr;
})();

function buildParamOpts(): Record<string, { min: number; max: number; step?: number }> {
  return {
    volumeSteps:  { min: 0, max: 0, step: 0 },
    shadowSteps:  { min: 0, max: 0, step: 0 },
    density:      { min: 0.1, max: 500, step: 0.1 },
    boundsRadius: { min: 1, max: 20, step: 0.1 },
    // Accumulator blend weight — direct α for the shader-side mix.
    // 1.0 → no blending (always show current frame, no smear).
    // 0.5 → 50/50 with previous frame.
    // 0.05 → heavy persistence-of-vision smear.
    // 0.01 → nearly frozen, slow ghost trails.
    accumBlend:   { min: 0.01, max: 1.0, step: 0.01 },
  };
}

function buildParamDefaults(): Record<string, number> {
  return {
    volumeSteps:  8,
    shadowSteps:  0,
    density:      50,
    boundsRadius: 8,
    accumBlend:   0.1,
  };
}

// Renders a hydrogen-orbital ψ field as a volumetric ray-march, with
// per-frame sub-pixel Halton jitter on the camera + IGN ray-start jitter
// in the shader, accumulated via a shader-side EMA across two ping-pong
// full-res render targets. Reads the orbital shape from `orbitalCloud.*`
// params so toggling between this and the particle view shows the same
// orbital.
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
  private renderer: ComponentDeps["renderer"];
  private camera: ComponentDeps["camera"];

  // Volume mesh — on layer 1, only renders into the half-res RT (NOT into
  // the main scenePass). Holds the ray-march material.
  private mesh: Mesh | null = null;
  private material: MeshBasicNodeMaterial | null = null;

  // Presenter mesh — on layer 0, rendered by the main scenePass. Its
  // material samples the RT at screenUV, so the visible "volume" you see
  // in the scene is the bilinear-upsampled half-res ray-march output. Cost
  // per fragment is one texture lookup.
  private presenter: Mesh | null = null;
  private presenterMaterial: MeshBasicNodeMaterial | null = null;

  // Wireframe outline — child of the presenter so it shows up in the main
  // scene at full res (crisp), not at half-res via the RT.
  private outline: LineSegments | null = null;
  private outlineMaterial: LineBasicNodeMaterial | null = null;

  // Half-res render target + a clone of the main camera with layers.set(1)
  // so only the volume mesh renders into it.
  private rt: RenderTarget | null = null;
  private rtCamera: PerspectiveCamera | null = null;
  private rtSize: Vector2 = new Vector2(0, 0);

  // Full-res ping-pong accumulator. Three's WebGPU rejects ConstantColor
  // blend factors as "Blend factor not supported" (silently → black
  // pipeline), so we do the EMA blend in the shader instead:
  //   accum = mix(prev_accum, sample, accumBlend)
  // Each frame samples one RT and writes to the other; the next frame
  // swaps roles. Presenter samples whichever was most recently written.
  private accumA: RenderTarget | null = null;
  private accumB: RenderTarget | null = null;
  private accumQuad: QuadMesh | null = null;
  private accumMaterial: MeshBasicNodeMaterial | null = null;
  // TextureNode whose .value is swapped each frame between accumA.texture
  // and accumB.texture — drives which target the quad shader reads as "prev".
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private accumPrevTex: any = null;
  // Same trick for the presenter — points at whichever RT was just written.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private presenterTex: any = null;
  // EMA blend weight (= params.accumBlend). Updated each frame.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private accumAlphaU: any = null;
  // Increments once per frame; drives Halton table index for camera jitter
  // and the shader's temporal jitter increment.
  private accumFrame = 0;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private uniforms: any = null;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.paramStore = deps.paramStore;
    this.renderer = deps.renderer;
    this.camera = deps.camera;
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
      // Per-frame counter for IGN-based ray-start jitter. Animated by
      // accumFrame so the noise pattern shifts each frame; the accumulator
      // averages them out.
      accumFrameU:  uniform(0),
    };

    const geom = new BoxGeometry(1, 1, 1);

    // ---- Half-res RT + ping-pong full-res accumulators + clone camera ----
    // HalfFloat (RGBA16F) — 8-bit RGB banded badly under the HDR post stack,
    // and the EMA accumulator amplifies quantization (residuals < 1/255
    // round to zero and the running mean snaps to step values).
    const rtOpts = { type: HalfFloatType };
    this.rt = new RenderTarget(2, 2, rtOpts);  // sized by resizeRT()
    this.accumA = new RenderTarget(2, 2, rtOpts);
    this.accumB = new RenderTarget(2, 2, rtOpts);
    this.resizeRT();

    this.rtCamera = this.camera.clone();
    this.rtCamera.layers.set(1);

    // ---- Accumulator fullscreen-quad ----
    // Shader: mix(prev_accum, low_res_sample, α). Both texture nodes are
    // bound to specific Texture instances at material creation; we swap
    // accumPrevTex.value each frame to point at whichever RT holds the
    // previous running mean. Replaces CustomBlending entirely.
    this.accumAlphaU = uniform(1);
    this.accumPrevTex = texture(this.accumA.texture, screenUV);
    const sampleTex = texture(this.rt.texture, screenUV);
    const accumMat = new MeshBasicNodeMaterial();
    accumMat.colorNode = mix(this.accumPrevTex, sampleTex, this.accumAlphaU);
    accumMat.transparent = false;
    accumMat.depthTest = false;
    accumMat.depthWrite = false;
    this.accumMaterial = accumMat;
    this.accumQuad = new QuadMesh(accumMat);

    // ---- Volume mesh — on layer 1, ray-marches into the RT ----
    const mat = new MeshBasicNodeMaterial();
    mat.side = BackSide;
    mat.transparent = true;
    mat.depthWrite = false;
    mat.blending = NormalBlending;
    mat.colorNode = this.buildColorNode();

    const mesh = new Mesh(geom, mat);
    mesh.scale.setScalar(this.params.boundsRadius * 2);
    mesh.frustumCulled = false;
    mesh.layers.set(1);   // not rendered by main camera
    this.scene.add(mesh);

    // ---- Presenter mesh — on layer 0, samples the RT at full res ----
    // Same cube geometry + scale as the volume mesh, but its fragment shader
    // is just a texture lookup → cheap. The visible volume in the scene is
    // the bilinear-upsampled half-res output.
    const presenterMat = new MeshBasicNodeMaterial();
    presenterMat.side = BackSide;
    presenterMat.transparent = true;
    presenterMat.depthWrite = false;
    presenterMat.blending = NormalBlending;
    // Presenter samples whichever ping-pong accumulator was most recently
    // written. presenterTex.value gets swapped to point at the active RT
    // each frame in update(). Initial binding is accumB because frame 0
    // writes to accumB (we read accumA's initial zeros, blend with α=1 to
    // discard them, output to accumB).
    this.presenterTex = texture(this.accumB.texture, screenUV);
    presenterMat.colorNode = this.presenterTex;
    // Opt out of GTAO darkening — same trick as before. setupNormal returns
    // a forward-facing view-space normal so the depth/normal mismatch at our
    // pixels (we don't write depth) doesn't cause GTAO to compute spurious
    // occlusion. See the spec'd comment from earlier work.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (presenterMat as any).setupNormal = () => vec3(0, 0, 1);

    const presenter = new Mesh(geom, presenterMat);
    presenter.scale.setScalar(this.params.boundsRadius * 2);
    presenter.frustumCulled = false;
    // No layers.set() — stays on default layer 0, rendered by main camera.
    this.scene.add(presenter);

    // ---- Wireframe outline — child of presenter so it renders at full res ----
    const edgeGeom = new EdgesGeometry(geom);
    const lineMat = new LineBasicNodeMaterial();
    lineMat.colorNode = vec4(0.1, 0.1, 0.1, 0.1);
    const outline = new LineSegments(edgeGeom, lineMat);
    outline.frustumCulled = false;
    presenter.add(outline);

    this.mesh = mesh;
    this.material = mat;
    this.presenter = presenter;
    this.presenterMaterial = presenterMat;
    this.outline = outline;
    this.outlineMaterial = lineMat;
  }

  // Resize the low-res RT to canvas × pixelRatio × renderScale, and the
  // accumulator pair to canvas × pixelRatio (full res). Called on init and
  // each frame from update() (no-op if size hasn't changed). The EMA will
  // quickly fade any garbage left over from the previous size.
  private static readonly RENDER_SCALE = 0.25;
  private static readonly CLEAR_COLOR = /*@__PURE__*/ new Color(0, 0, 0);
  private resizeRT(): void {
    if (!this.rt || !this.accumA || !this.accumB) return;
    const sz = this.renderer.getSize(this.rtSize);
    const pr = this.renderer.getPixelRatio();
    const lowW = Math.max(1, Math.floor(sz.width * pr * OrbitalVolume.RENDER_SCALE));
    const lowH = Math.max(1, Math.floor(sz.height * pr * OrbitalVolume.RENDER_SCALE));
    const fullW = Math.max(1, Math.floor(sz.width * pr));
    const fullH = Math.max(1, Math.floor(sz.height * pr));
    if (this.rt.width !== lowW || this.rt.height !== lowH) this.rt.setSize(lowW, lowH);
    if (this.accumA.width !== fullW || this.accumA.height !== fullH) this.accumA.setSize(fullW, fullH);
    if (this.accumB.width !== fullW || this.accumB.height !== fullH) this.accumB.setSize(fullW, fullH);
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

      // Per-pixel, per-frame ray-start jitter. Interleaved Gradient Noise
      // (Jimenez 2014) gives a cheap blue-noise-like spatial pattern; the
      // golden-ratio temporal increment shifts the pattern each frame so
      // every pixel cycles through different sub-step offsets. The
      // accumulator's running mean integrates these → no step-count banding.
      const sx = screenCoordinate.x;
      const sy = screenCoordinate.y;
      const ign = fract(
        float(52.9829189).mul(
          fract(float(0.06711056).mul(sx).add(float(0.00583715).mul(sy)))
        )
      );
      const jitter = fract(ign.add(u.accumFrameU.mul(float(0.61803398875))));
      let pos = rayOrigin.add(rayDir.mul(tNear.add(dt.mul(jitter)))).toVar();
      let accumColor = vec3(0).toVar();
      let accumAlpha = float(0).toVar();

      Loop(32, () => {
        const psi = evalPsi(pos, u.shCoefs, u.n, u.radialScale);

        const dens = psi.mul(psi).mul(u.density).min(float(1)).max(float(0));
        if (dens.greaterThan(float(0.01))) {
            const stepAlpha = float(1).sub(exp(dens.mul(dt).negate()));

            // Hot-cold sample color (algebraic-sigmoid normalization).
            const xCol = psi.mul(u.colorScale);
            const tNorm = xCol.div(xCol.abs().add(float(1)));
            const tPos = tNorm.max(float(0));
            const tNeg = tNorm.min(float(0)).abs();
            const sampleColor = BLACK
              .add(GREEN.sub(BLACK).mul(tPos))
              .add(RED.sub(BLACK).mul(tNeg));

            // Shadow march from pos along LIGHT_DIR. Fixed 8 iterations with a
            // fixed shadowDt of 0.2; shadowSteps slider is currently a no-op.
            const shadowDt = float(0.5);
            const shadowDens = float(0).toVar();
            let shadowPos = pos.toVar();
            const shadowStep = LIGHT_DIR.mul(shadowDt);

            Loop(16, () => {

              shadowPos.addAssign(shadowStep);
              const psiS = evalPsi(shadowPos, u.shCoefs, u.n, u.radialScale);
              shadowDens.addAssign(psiS.mul(psiS).mul(u.density).mul(shadowDt));
            });

            // Transmittance with a 40% ambient floor so back-lit fragments don't
            // crush to near-zero.
            const transmittance = exp(shadowDens.negate());
            const lit = transmittance.mul(float(0.8)).add(float(0.2));

            // Front-to-back composite (premultiplied alpha).
            accumColor.addAssign(sampleColor.mul(lit).mul(stepAlpha).mul(float(1).sub(accumAlpha)));
            accumAlpha.addAssign(stepAlpha.mul(float(1).sub(accumAlpha)));

        }
     
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

    const scale = this.params.boundsRadius * 2;
    if (this.mesh) this.mesh.scale.setScalar(scale);
    if (this.presenter) this.presenter.scale.setScalar(scale);

    // ---- Per-frame accumulator dance ----
    //   1. resizeRT(): handle canvas resize (RTs reallocate; EMA fades any
    //      garbage in a few frames)
    //   2. Sync rtCamera + apply sub-pixel Halton jitter to the projection
    //   3. Render volume into the low-res RT
    //   4. Swap ping-pong roles, point accumPrevTex at the read RT, render
    //      the quad into the write RT — shader does mix(prev, sample, blend)
    //   5. Point presenterTex at the write RT so main scene shows freshest
    //   6. Increment accumFrame (drives Halton index + temporal noise)
    if (!this.rt || !this.rtCamera || !this.accumA || !this.accumB || !this.accumQuad ||
        !this.accumMaterial || !this.accumPrevTex || !this.presenterTex || !this.accumAlphaU) {
      return;
    }

    this.resizeRT();

    this.rtCamera.copy(this.camera);
    this.rtCamera.layers.set(1);   // .copy() clobbers the layer mask

    // Sub-pixel camera jitter — Halton(2, 3) offset in [-0.5, 0.5]² applied
    // to the projection matrix's (cx, cy) translation. NDC unit = 2/W, so
    // jx · (2/lowResW) shifts the camera by jx low-res-pixels. Over
    // HALTON_LEN frames the accumulator sees the volume from a range of
    // sub-pixel positions → quasi-supersampled image.
    const idx = (this.accumFrame % HALTON_LEN) * 2;
    const jx = HALTON_JITTER[idx];
    const jy = HALTON_JITTER[idx + 1];
    const proj = this.rtCamera.projectionMatrix.elements;
    proj[8]  += jx * (2 / this.rt.width);
    proj[9]  += jy * (2 / this.rt.height);

    // Drive the shader's per-frame jitter increment.
    u.accumFrameU.value = this.accumFrame;

    // setClearColor is a RENDERER-GLOBAL setter. If we don't restore it the
    // main scene's clear color (from Scene.ts) gets overwritten and every
    // frame after ours clears to (0,0,0,0) → entire app goes black. Save
    // and restore around our own renders.
    const prevClearColor = OrbitalVolume._scratchColor;
    // getClearColor's strict signature expects a Color4 (rgba) — the runtime
    // implementation just writes r/g/b into the target. Cast to bypass.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.renderer.getClearColor(prevClearColor as any);
    const prevClearAlpha = this.renderer.getClearAlpha();

    // Stage 1: render volume into the low-res RT.
    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.rt);
    this.renderer.setClearColor(OrbitalVolume.CLEAR_COLOR, 0);
    this.renderer.clear();
    void this.renderer.renderAsync(this.scene, this.rtCamera);

    // Stage 2: ping-pong shader blend. Pick read (prev) + write (curr) based
    // on frame parity, point the accumulator material's TextureNode at prev,
    // render the quad into curr. Shader does mix(prev, sample, accumBlend).
    const aOnEven = (this.accumFrame & 1) === 0;
    const readRT = aOnEven ? this.accumA : this.accumB;
    const writeRT = aOnEven ? this.accumB : this.accumA;

    this.accumAlphaU.value = this.params.accumBlend;
    this.accumPrevTex.value = readRT.texture;

    this.renderer.setRenderTarget(writeRT);
    void this.accumQuad.renderAsync(this.renderer);

    // Presenter samples the JUST-WRITTEN buffer.
    this.presenterTex.value = writeRT.texture;

    this.accumFrame += 1;

    // Restore renderer-global state for the main scenePass.
    this.renderer.setRenderTarget(prevTarget);
    this.renderer.setClearColor(prevClearColor, prevClearAlpha);
  }

  // Scratch Color used by update() to read out the renderer's previous clear
  // color before mutating it. Module-level static so we don't allocate per
  // frame.
  private static readonly _scratchColor = /*@__PURE__*/ new Color();

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.outline) {
      this.outline.geometry.dispose();
      this.outlineMaterial?.dispose();
      this.outline = null;
      this.outlineMaterial = null;
    }
    if (this.presenter) {
      this.scene.remove(this.presenter);
      this.presenterMaterial?.dispose();
      this.presenter = null;
      this.presenterMaterial = null;
    }
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.material?.dispose();
      this.mesh = null;
      this.material = null;
    }
    if (this.rt) {
      this.rt.dispose();
      this.rt = null;
    }
    if (this.accumA) {
      this.accumA.dispose();
      this.accumA = null;
    }
    if (this.accumB) {
      this.accumB.dispose();
      this.accumB = null;
    }
    if (this.accumMaterial) {
      this.accumMaterial.dispose();
      this.accumMaterial = null;
    }
    this.accumQuad = null;
    this.accumPrevTex = null;
    this.presenterTex = null;
    this.accumAlphaU = null;
    this.rtCamera = null;
  }
}
