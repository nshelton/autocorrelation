import {
  InstancedMesh,
  BoxGeometry,
  CylinderGeometry,
  IcosahedronGeometry,
  TorusGeometry,
  ConeGeometry,
  BufferGeometry,
  Object3D,
  Color,
} from "three";
import type { MeshStandardNodeMaterial } from "three/webgpu";
import { vec4, uniform } from "three/tsl";
import { makeLitMaterial, releaseLitMaterial } from "./litMaterial";
import type { Component, ComponentDeps } from "./Component";

// Per-layer instance ceiling (steps x arms). Meshes are sized to the exact live
// count, never this — see ParticleView for why over-allocating the instance
// buffer silently kills the draw on the WebGPU backend.
const MAX_PER_LAYER = 2048;
const TWO_PI = Math.PI * 2;
const DEG = Math.PI / 180;

// Unit-sized primitives: every shape is 1 unit across in each axis so the
// per-instance sizeX/Y/Z scale means the same thing whichever shape is picked.
// Long-axis shapes are pre-rotated to Z so sizeZ is always "length down the
// tunnel", matching the box.
const SHAPES: Array<() => BufferGeometry> = [
  () => new BoxGeometry(1, 1, 1),
  () => new CylinderGeometry(0.5, 0.5, 1, 16).rotateX(Math.PI / 2),
  () => new IcosahedronGeometry(0.5, 2),
  () => new TorusGeometry(0.35, 0.15, 8, 24),
  () => new ConeGeometry(0.5, 1, 16).rotateX(Math.PI / 2),
];

const SHAPE_LABELS = ["Box", "Tube", "Sphere", "Torus", "Cone"];

const LAYER_KEYS = ["l1", "l2", "l3"] as const;

// Live state for one of the three layers. Everything else is read from params.
class Layer {
  mesh: InstancedMesh | null = null;
  mat: MeshStandardNodeMaterial;
  // Held by reference inside the material's uniform node — mutate via setHex,
  // never reassign, or the shader keeps reading the old object.
  color = new Color();
  emissiveU = uniform(0);
  // Accumulated rather than derived from a global clock, so moving the speed or
  // spin slider changes the rate without teleporting the whole layer.
  travel = 0;
  spinPhase = 0;
  count = 0;
  lastShape = NaN;
  lastColor = NaN;

  constructor(public key: string) {
    this.mat = makeLitMaterial();
    const c = uniform(this.color);
    this.mat.colorNode = vec4(c, 1.0);
    // Emission is the layer's own color scaled past 1 so bloom has something to
    // clip; albedo stays the same color so unlit faces still read as the layer.
    this.mat.emissiveNode = c.mul(this.emissiveU);
  }
}

// Positive modulo — JS % keeps the sign of the dividend, and `travel` runs
// negative whenever speed is.
function pmod(x: number, m: number): number {
  return ((x % m) + m) % m;
}

// A still camera inside a moving tube. Each layer lays `steps` copies of one
// primitive down the -Z axis, offset `radius` off the axis and rotated `twist`
// degrees more each step, which draws a helix; `arms` repeats that helix around
// the axis. The copies stream toward +Z and wrap back to the far end, so the
// tunnel is endless without anything being spawned or destroyed.
export class Tunnel implements Component {
  static id = "tunnel";
  static label = "Tunnel";
  static paramPrefix = "tunnel";
  static paramOpts = {
    speed: { min: -20, max: 20, step: 0.05 },
    near: { min: -10, max: 20, step: 0.1 },
    fade: { min: 0, max: 20, step: 0.1 },
    ...layerOpts("l1"),
    ...layerOpts("l2"),
    ...layerOpts("l3"),
  };
  static paramDefaults = {
    speed: 4,
    near: 6,
    wireframe: 0,
    fade: 4,
    // Long radial blades close in — the "off-center box copied and rotated"
    // spiral, one arm so the helix reads as a single ribbon.
    ...layerDefaults("l1", {
      shape: 0, steps: 140, arms: 1, spacing: 0.35, radius: 1.6,
      radiusWave: 0.25, waveFreq: 2, twist: 11,
      spin: 4, speedMul: 1, sizeX: 0.9, sizeY: 0.07, sizeZ: 0.07,
      color: 0xff9d4d, emission: 1.2,
    }),
    // Three long rails further out, counter-twisting and drifting slower for
    // parallax against l1.
    ...layerDefaults("l2", {
      shape: 0, steps: 90, arms: 3, spacing: 0.6, radius: 2.7,
      radiusWave: 0, waveFreq: 1, twist: -7,
      spin: -6, speedMul: 0.6, sizeX: 0.06, sizeY: 0.06, sizeZ: 1.6,
      color: 0x35d6ff, emission: 2.5,
    }),
    // Sparse fast beads at the outside, bright enough to streak through bloom.
    ...layerDefaults("l3", {
      shape: 2, steps: 60, arms: 6, spacing: 0.9, radius: 3.6,
      radiusWave: 0.6, waveFreq: 3, twist: 3,
      spin: 10, speedMul: 1.5, sizeX: 0.14, sizeY: 0.14, sizeZ: 0.14,
      color: 0xff4de0, emission: 4,
    }),
  };
  static paramKinds = {
    wireframe: "discrete" as const,
    ...layerKinds("l1"),
    ...layerKinds("l2"),
    ...layerKinds("l3"),
  };
  static paramDiscreteOptions = {
    wireframe: [0, 1],
    "l1.shape": [0, 1, 2, 3, 4],
    "l2.shape": [0, 1, 2, 3, 4],
    "l3.shape": [0, 1, 2, 3, 4],
  };
  static paramDiscreteLabels = {
    wireframe: ["Off", "On"],
    "l1.shape": SHAPE_LABELS,
    "l2.shape": SHAPE_LABELS,
    "l3.shape": SHAPE_LABELS,
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private layers: Layer[] = [];
  private dummy = new Object3D();
  private lastTime = NaN;
  private lastWireframe = NaN;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    for (const key of LAYER_KEYS) this.layers.push(new Layer(key));
  }

  private p(layer: Layer, name: string): number {
    return this.params[`${layer.key}.${name}`];
  }

  // Size the mesh to exactly `n` instances and swap in the current shape. The
  // material outlives the mesh (its uniforms hold the layer color), so only the
  // geometry and the InstancedMesh itself are recycled here.
  private rebuild(layer: Layer, n: number, shape: number): void {
    this.disposeMesh(layer);
    layer.count = n;
    layer.lastShape = shape;
    if (n === 0) return;

    const mesh = new InstancedMesh(SHAPES[shape](), layer.mat, n);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // Instances rewrite every frame without a bounding-sphere refresh; culling
    // on stale bounds would drop the whole tunnel.
    mesh.frustumCulled = false;
    layer.mesh = mesh;
    this.scene.add(mesh);
  }

  private disposeMesh(layer: Layer): void {
    if (!layer.mesh) return;
    this.scene.remove(layer.mesh);
    layer.mesh.geometry.dispose();
    layer.mesh.dispose();
    layer.mesh = null;
  }

  update(): void {
    // Wall-clock dt, clamped so a backgrounded tab doesn't jump the tunnel a
    // hundred units on the frame it comes back.
    const now = performance.now();
    const dt = Number.isFinite(this.lastTime) ? Math.min((now - this.lastTime) / 1000, 0.1) : 0;
    this.lastTime = now;

    if (this.params.wireframe !== this.lastWireframe) {
      const on = this.params.wireframe >= 0.5;
      for (const layer of this.layers) layer.mat.wireframe = on;
      this.lastWireframe = this.params.wireframe;
    }

    for (const layer of this.layers) this.updateLayer(layer, dt);
  }

  private updateLayer(layer: Layer, dt: number): void {
    const steps = Math.round(this.p(layer, "steps"));
    const arms = Math.max(1, Math.round(this.p(layer, "arms")));
    // Drop whole steps rather than partial rings when the product overflows, so
    // the tunnel gets shorter instead of losing a slice of its symmetry.
    const usableSteps = Math.min(steps, Math.floor(MAX_PER_LAYER / arms));
    const n = usableSteps * arms;
    const shape = Math.round(this.p(layer, "shape"));
    if (n !== layer.count || shape !== layer.lastShape) this.rebuild(layer, n, shape);

    const hex = this.p(layer, "color");
    if (hex !== layer.lastColor) {
      layer.color.setHex(hex);
      layer.lastColor = hex;
    }
    layer.emissiveU.value = this.p(layer, "emission");

    const mesh = layer.mesh;
    if (!mesh || n === 0) return;

    const spacing = Math.max(this.p(layer, "spacing"), 1e-3);
    const depth = usableSteps * spacing;
    layer.travel += dt * this.params.speed * this.p(layer, "speedMul");
    layer.travel = pmod(layer.travel, depth);
    layer.spinPhase += dt * this.p(layer, "spin") * DEG;

    const near = this.params.near;
    const fade = this.params.fade;
    const radius = this.p(layer, "radius");
    const wave = this.p(layer, "radiusWave");
    const waveFreq = this.p(layer, "waveFreq");
    const twist = this.p(layer, "twist") * DEG;
    const sx = this.p(layer, "sizeX");
    const sy = this.p(layer, "sizeY");
    const sz = this.p(layer, "sizeZ");
    const armStep = TWO_PI / arms;

    let slot = 0;
    for (let i = 0; i < usableSteps; i++) {
      // phase 0 is the far end of the tunnel; it grows as the copy approaches.
      const phase = pmod(i * spacing + layer.travel, depth);
      const z = near - depth + phase;
      // Shrink to nothing at both ends so copies don't pop in or out mid-view.
      // Both ends, not just the far one, because the camera can be flown out to
      // the side where the near wrap is visible too.
      const f = fade > 0 ? Math.min(Math.min(phase, depth - phase) / fade, 1) : 1;
      // Wave rides on phase (not the index), so the pinch stays put in world
      // space while the geometry streams through it.
      const r = radius + wave * Math.sin(TWO_PI * waveFreq * (phase / depth));
      const base = i * twist + layer.spinPhase;

      for (let a = 0; a < arms; a++) {
        const theta = base + a * armStep;
        this.dummy.position.set(r * Math.cos(theta), r * Math.sin(theta), z);
        // Rotation about world Z only: local +X ends up radially outward and
        // local +Z stays pointed down the tunnel, so sizeX is radial thickness
        // and sizeZ is length along the tunnel whatever the angle.
        this.dummy.rotation.set(0, 0, theta);
        this.dummy.scale.set(sx * f, sy * f, sz * f);
        this.dummy.updateMatrix();
        mesh.setMatrixAt(slot++, this.dummy.matrix);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    for (const layer of this.layers) {
      this.disposeMesh(layer);
      releaseLitMaterial(layer.mat);
    }
    this.layers = [];
  }
}

// The three layers are identical parameter sets under an `lN.` prefix. Written
// out as helpers rather than by hand so a new control lands in all three at
// once and can't drift between them.
function layerOpts(k: string): Record<string, { min: number; max: number; step: number }> {
  return {
    [`${k}.steps`]: { min: 0, max: 256, step: 1 },
    [`${k}.arms`]: { min: 1, max: 8, step: 1 },
    [`${k}.spacing`]: { min: 0.02, max: 3, step: 0.01 },
    [`${k}.radius`]: { min: 0, max: 8, step: 0.01 },
    [`${k}.radiusWave`]: { min: -4, max: 4, step: 0.01 },
    [`${k}.waveFreq`]: { min: 0, max: 8, step: 0.05 },
    [`${k}.twist`]: { min: -60, max: 60, step: 0.1 },
    [`${k}.spin`]: { min: -180, max: 180, step: 0.5 },
    [`${k}.speedMul`]: { min: -3, max: 3, step: 0.01 },
    [`${k}.sizeX`]: { min: 0.01, max: 4, step: 0.01 },
    [`${k}.sizeY`]: { min: 0.01, max: 4, step: 0.01 },
    [`${k}.sizeZ`]: { min: 0.01, max: 4, step: 0.01 },
    [`${k}.emission`]: { min: 0, max: 8, step: 0.05 },
  };
}

function layerDefaults(k: string, v: Record<string, number>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [name, value] of Object.entries(v)) out[`${k}.${name}`] = value;
  return out;
}

function layerKinds(k: string): Record<string, "discrete" | "color"> {
  return { [`${k}.shape`]: "discrete", [`${k}.color`]: "color" };
}
