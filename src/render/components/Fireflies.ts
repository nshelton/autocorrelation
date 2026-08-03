import {
  InstancedMesh,
  IcosahedronGeometry,
  InstancedBufferAttribute,
  Object3D,
  Color,
} from "three";
import { MeshBasicNodeMaterial } from "three/webgpu";
import { vec4, uniform, instancedBufferAttribute } from "three/tsl";
import { createCurlNoise } from "../curl-noise";
import type { Component, ComponentDeps } from "./Component";

// Fixed pool ceiling. 1024 instances is also the WebGPU uniform-buffer cliff
// documented in ParticleView — the mesh is sized to the live count, never this.
const MAX_FIREFLIES = 1024;
// Per-firefly size jitter, multiplied by the `size` param.
const SIZE_MIN = 0.5;
const SIZE_MAX = 1.5;
// Per-firefly flicker rate spread, multiplied by the `flickerSpeed` param.
const RATE_MIN = 0.4;
const RATE_MAX = 2.5;
// Inward speed per unit of overshoot outside `zone`. A soft pull rather than a
// torus wrap: nothing teleports across the frame, which would read as a pop.
const CONTAIN_PULL = 3;

// Emissive point lights drifting on a curl-noise field, each blinking on its
// own phase. No physics, no lighting — the material is unlit, so its color IS
// the emission, and brightness > 1 gives the bloom threshold something to
// clip. Opaque (not additive), so it writes depth like any solid geometry;
// that's what makes it usable as a DOF test subject.
export class Fireflies implements Component {
  static id = "fireflies";
  static label = "Fireflies";
  static paramPrefix = "fireflies";
  static paramOpts = {
    zone: { min: 0.2, max: 5, step: 0.05 },
    size: { min: 0.002, max: 0.08, step: 0.001 },
    speed: { min: 0, max: 2, step: 0.01 },
    noiseScale: { min: 0.1, max: 4, step: 0.05 },
    flickerSpeed: { min: 0, max: 8, step: 0.05 },
    flickerAmount: { min: 0, max: 1, step: 0.01 },
    brightness: { min: 0, max: 8, step: 0.05 },
  };
  static paramDefaults = {
    count: 256,
    zone: 1.5,
    size: 0.015,
    speed: 0.3,
    noiseScale: 0.8,
    flickerSpeed: 1.5,
    flickerAmount: 0.8,
    brightness: 2.5,
    color: 0xffd27a,
  };
  static paramKinds = {
    count: "discrete" as const,
    color: "color" as const,
  };
  static paramDiscreteOptions = {
    count: [64, 128, 256, 512, 1024],
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private mesh: InstancedMesh | null = null;
  private count = 0;
  // SoA, allocated once at MAX_FIREFLIES. Slots past `count` hold stale state
  // and are simply not drawn; raising the count reuses them from where they
  // left off instead of reseeding everyone.
  private pos = new Float32Array(MAX_FIREFLIES * 3);
  private phase = new Float32Array(MAX_FIREFLIES);
  private rate = new Float32Array(MAX_FIREFLIES);
  private scales = new Float32Array(MAX_FIREFLIES);
  // Per-instance emission multiplier (flicker x brightness), sized to the live
  // count and owned by the mesh's instanced attribute.
  private bright = new Float32Array(0);
  private brightAttr: InstancedBufferAttribute | null = null;
  private curlNoise: (x: number, y: number, z: number, out: Float32Array) => void;
  private curlOut = new Float32Array(3);
  private lastNoiseScale: number;
  // Referenced live by the material's uniform node; mutate in place, never
  // reassign, so a rebuilt mesh keeps pointing at the same instance.
  private color = new Color();
  private lastColor = NaN;
  private lastTime = NaN;
  private dummy = new Object3D();

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    this.curlNoise = createCurlNoise({ scale: params.noiseScale });
    this.lastNoiseScale = params.noiseScale;
    this.rebuild(Math.round(params.count));
  }

  // Uniform point in the ball of radius `zone` (cbrt of the radial random, so
  // they aren't clumped at the center), plus per-firefly size + blink rate.
  private seed(i: number): void {
    const r = this.params.zone * Math.cbrt(Math.random());
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const sinPhi = Math.sin(phi);
    this.pos[i * 3] = r * sinPhi * Math.cos(theta);
    this.pos[i * 3 + 1] = r * sinPhi * Math.sin(theta);
    this.pos[i * 3 + 2] = r * Math.cos(phi);
    this.phase[i] = Math.random() * Math.PI * 2;
    this.rate[i] = RATE_MIN + Math.random() * (RATE_MAX - RATE_MIN);
    this.scales[i] = SIZE_MIN + Math.random() * (SIZE_MAX - SIZE_MIN);
  }

  // Size the InstancedMesh to exactly `n` — see ParticleView for why the
  // WebGPU backend punishes over-allocating the instance buffer.
  private rebuild(n: number): void {
    for (let i = this.count; i < n; i++) this.seed(i);
    this.disposeMesh();

    this.bright = new Float32Array(n);
    this.brightAttr = new InstancedBufferAttribute(this.bright, 1);
    const mat = new MeshBasicNodeMaterial();
    // Unlit: colorNode is the final emitted color. The per-instance float
    // carries flicker x brightness and can exceed 1 — headroom for bloom.
    mat.colorNode = vec4(
      uniform(this.color).mul(instancedBufferAttribute(this.brightAttr, "float", 1, 0)),
      1.0,
    );

    const mesh = new InstancedMesh(new IcosahedronGeometry(1, 1), mat, n);
    // Instances move every frame without a boundingSphere refresh, so let the
    // renderer draw the whole pool rather than cull it on stale bounds.
    mesh.frustumCulled = false;
    this.mesh = mesh;
    this.scene.add(mesh);
    this.count = n;
  }

  private disposeMesh(): void {
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    (this.mesh.material as MeshBasicNodeMaterial).dispose();
    this.mesh.dispose();
    this.mesh = null;
  }

  update(): void {
    const n = Math.round(this.params.count);
    if (n !== this.count) this.rebuild(n);
    const mesh = this.mesh;
    if (!mesh || !this.brightAttr) return;

    // noiseScale is closed over at construction, so a slider move means a new
    // field. Same guard ParticleView uses.
    if (this.params.noiseScale !== this.lastNoiseScale) {
      this.curlNoise = createCurlNoise({ scale: this.params.noiseScale });
      this.lastNoiseScale = this.params.noiseScale;
    }
    if (this.params.color !== this.lastColor) {
      this.color.setHex(this.params.color);
      this.lastColor = this.params.color;
    }

    // Wall-clock dt (this component ignores the physics world), clamped so a
    // backgrounded tab doesn't fling everyone out of the zone on return.
    const now = performance.now();
    const dt = Number.isFinite(this.lastTime) ? Math.min((now - this.lastTime) / 1000, 0.1) : 0;
    this.lastTime = now;

    const { zone, speed, size, brightness, flickerSpeed, flickerAmount } = this.params;

    for (let i = 0; i < n; i++) {
      const o = i * 3;
      let x = this.pos[o];
      let y = this.pos[o + 1];
      let z = this.pos[o + 2];

      // Velocity field, not an acceleration — curl noise is divergence-free,
      // so advecting straight through it swirls without anyone piling up.
      this.curlNoise(x, y, z, this.curlOut);
      let vx = this.curlOut[0] * speed;
      let vy = this.curlOut[1] * speed;
      let vz = this.curlOut[2] * speed;

      const d = Math.sqrt(x * x + y * y + z * z);
      if (d > zone) {
        const k = ((d - zone) * CONTAIN_PULL) / d;
        vx -= x * k;
        vy -= y * k;
        vz -= z * k;
      }

      x += vx * dt;
      y += vy * dt;
      z += vz * dt;
      this.pos[o] = x;
      this.pos[o + 1] = y;
      this.pos[o + 2] = z;

      // Phase accumulates per firefly rather than being derived from a global
      // clock, so moving flickerSpeed changes the rate without jumping anyone
      // mid-blink. Squared sine for a snappier on/off than a plain sine.
      this.phase[i] += dt * flickerSpeed * this.rate[i];
      const s = 0.5 + 0.5 * Math.sin(this.phase[i]);
      this.bright[i] = brightness * (1 - flickerAmount + flickerAmount * s * s);

      const r = size * this.scales[i];
      this.dummy.position.set(x, y, z);
      this.dummy.scale.set(r, r, r);
      this.dummy.updateMatrix();
      mesh.setMatrixAt(i, this.dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
    this.brightAttr.needsUpdate = true;
  }

  dispose(): void {
    this.disposeMesh();
    this.brightAttr = null;
  }
}
