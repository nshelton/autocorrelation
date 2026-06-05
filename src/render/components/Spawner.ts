import {
  InstancedMesh,
  BoxGeometry,
  IcosahedronGeometry,
  CylinderGeometry,
  BufferGeometry,
  Object3D,
} from "three";
import { MeshBasicNodeMaterial } from "three/webgpu";
import { vec3, vec4, float, dot, max, normalWorld } from "three/tsl";
import RAPIER from "@dimforge/rapier3d-compat";
import { createCurlNoise } from "../curl-noise";
import type { Component, ComponentDeps } from "./Component";

// Per-type fixed pool size. 512 instances * 64-byte matrix = 32KB, under the
// WebGPU 1024-instance uniform-buffer cliff documented in ParticleView.
const MAX_PER_TYPE = 512;
// Base half-extent for every shape (cube half-side, sphere/disk radius).
const BASE = 0.08;
const DISK_HALF_HEIGHT = BASE * 0.18;
// Per-object random size jitter, multiplied by objectScale.
const SCALE_MIN = 0.6;
const SCALE_MAX = 1.4;
const LIFETIME_JITTER_SECS = 1.0;
// Lower bound on the collider's shrink factor — keeps it from collapsing to a
// degenerate size as an object fades out (the mesh still shrinks fully to 0).
const COLLIDER_MIN_FADE = 0.15;

// Force field types (forceFieldType param values).
const FIELD_LINEAR = 0;
const FIELD_CURL = 1;
const FIELD_ATTRACT = 2;

// Spawn buttons can't reach the live component instance (paramButtons.onClick
// only gets the ParamStore), so they push pending counts here and the live
// Spawner drains them each frame. Same module-singleton bridge OrbitalCloud
// uses with shTween. Survives HMR; Spawner.reset()s it on construct so presses
// banked while disabled don't burst-spawn on enable.
class SpawnQueue {
  cube = 0;
  sphere = 0;
  disk = 0;
  request(type: "cube" | "sphere" | "disk"): void {
    this[type]++;
  }
  reset(): void {
    this.cube = this.sphere = this.disk = 0;
  }
}
export const spawnQueue = new SpawnQueue();

// Static per-shape geometry/collider/color. resize() keeps the collider in sync
// with the per-object visual scale at spawn time.
interface ShapeDef {
  geometry: () => BufferGeometry;
  collider: () => RAPIER.ColliderDesc;
  resize: (c: RAPIER.Collider, s: number) => void;
  color: [number, number, number];
  // Param key holding this shape's per-object size multiplier.
  scaleParam: string;
  // Param key holding this shape's continuous spawn rate (objects/second).
  rateParam: string;
}

const SHAPES: ShapeDef[] = [
  {
    geometry: () => new BoxGeometry(BASE * 2, BASE * 2, BASE * 2),
    collider: () => RAPIER.ColliderDesc.cuboid(BASE, BASE, BASE),
    resize: (c, s) => c.setHalfExtents({ x: BASE * s, y: BASE * s, z: BASE * s }),
    color: [0.95, 0.45, 0.25],
    scaleParam: "cubeScale",
    rateParam: "cubeRate",
  },
  {
    geometry: () => new IcosahedronGeometry(BASE, 2),
    collider: () => RAPIER.ColliderDesc.ball(BASE),
    resize: (c, s) => c.setRadius(BASE * s),
    color: [0.3, 0.7, 0.95],
    scaleParam: "sphereScale",
    rateParam: "sphereRate",
  },
  {
    geometry: () => new CylinderGeometry(BASE, BASE, DISK_HALF_HEIGHT * 2, 20),
    collider: () => RAPIER.ColliderDesc.cylinder(DISK_HALF_HEIGHT, BASE),
    resize: (c, s) => {
      c.setRadius(BASE * s);
      c.setHalfHeight(DISK_HALF_HEIGHT * s);
    },
    color: [0.6, 0.9, 0.4],
    scaleParam: "diskScale",
    rateParam: "diskRate",
  },
];

// One shape's ring buffer: fixed body/collider pool + parallel SoA slot state.
// Plain data; all logic lives in Spawner.
class Pool {
  bodies: RAPIER.RigidBody[] = [];
  colliders: RAPIER.Collider[] = [];
  active = new Uint8Array(MAX_PER_TYPE);
  lifetime = new Float32Array(MAX_PER_TYPE);
  maxLifetime = new Float32Array(MAX_PER_TYPE);
  scale = new Float32Array(MAX_PER_TYPE);
  next = 0;
  // Fractional spawn-rate carry: rate*dt accumulates here; whole units spawn.
  accum = 0;
  mesh!: InstancedMesh;
  constructor(public def: ShapeDef) {}
}

export class Spawner implements Component {
  static id = "spawner";
  static label = "Spawner";
  static paramPrefix = "spawner";
  static paramOpts = {
    forceFieldType: { min: 0, max: 0, step: 0 }, // discrete, see below
    forceStrength: { min: 0, max: 30, step: 0.1 },
    noiseScale: { min: 0.01, max: 1.0, step: 0.01 },
    spawnImpulse: { min: 0, max: 10, step: 0.1 },
    lifetime: { min: 1, max: 15, step: 0.1 },
    restitution: { min: 0, max: 1, step: 0.01 },
    damping: { min: 0, max: 2, step: 0.01 },
    timescale: { min: 0, max: 3, step: 0.01 },
    cubeScale: { min: 0.2, max: 9, step: 0.05 },
    sphereScale: { min: 0.2, max: 9, step: 0.05 },
    diskScale: { min: 0.2, max: 9, step: 0.05 },
    cubeRate: { min: 0, max: 50, step: 0.5 },
    sphereRate: { min: 0, max: 50, step: 0.5 },
    diskRate: { min: 0, max: 50, step: 0.5 },
  };
  static paramDefaults = {
    forceFieldType: 0,
    forceStrength: 9.8,
    noiseScale: 0.5,
    spawnImpulse: 3,
    lifetime: 4,
    restitution: 0.5,
    damping: 0.1,
    timescale: 1.0,
    cubeScale: 1.0,
    sphereScale: 1.0,
    diskScale: 1.0,
    cubeRate: 0,
    sphereRate: 0,
    diskRate: 0,
    wireframe: 0,
  };
  static paramKinds = {
    forceFieldType: "discrete" as const,
    wireframe: "discrete" as const,
  };
  static paramDiscreteOptions = {
    forceFieldType: [0, 1, 2],
    wireframe: [0, 1],
  };
  static paramDiscreteLabels = {
    forceFieldType: ["Linear", "Curl", "Attract"],
    wireframe: ["Off", "On"],
  };
  static paramButtons = [
    { title: "Spawn Cube", onClick: () => spawnQueue.request("cube") },
    { title: "Spawn Sphere", onClick: () => spawnQueue.request("sphere") },
    { title: "Spawn Disk", onClick: () => spawnQueue.request("disk") },
  ];

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private world: RAPIER.World | null = null;
  private pools: Pool[] = [];
  private dummy = new Object3D();
  private curlOut = new Float32Array(3);
  private curlNoise!: (x: number, y: number, z: number, out: Float32Array) => void;
  private lastNoiseScale = NaN;
  private lastDamping = NaN;
  private lastRestitution = NaN;
  private lastTimescale = NaN;
  private lastGravityY = NaN;
  private lastTime = NaN;
  private lastWireframe = NaN;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    spawnQueue.reset();
    void this.init();
  }

  private async init(): Promise<void> {
    await RAPIER.init();
    if (this.disposed) return;

    this.curlNoise = createCurlNoise({ scale: this.params.noiseScale });
    this.lastNoiseScale = this.params.noiseScale;
    this.world = new RAPIER.World({ x: 0, y: 0, z: 0 });

    for (const def of SHAPES) {
      const pool = new Pool(def);
      // Pre-create the full body/collider pool, all disabled (out of the
      // simulation) until spawned. Bodies sit at the origin; their instance
      // matrices start zero-scale so empty slots are invisible.
      for (let i = 0; i < MAX_PER_TYPE; i++) {
        const body = this.world.createRigidBody(
          RAPIER.RigidBodyDesc.dynamic()
            .setLinearDamping(this.params.damping)
            .setAngularDamping(this.params.damping)
            .setEnabled(false),
        );
        const collider = this.world.createCollider(
          def.collider().setRestitution(this.params.restitution),
          body,
        );
        pool.bodies.push(body);
        pool.colliders.push(collider);
      }
      pool.mesh = this.createMesh(def);
      this.pools.push(pool);
    }
  }

  private createMesh(def: ShapeDef): InstancedMesh {
    // Hand-rolled lambert on an UNLIT MeshBasicNodeMaterial. Real lit node
    // materials (MeshStandardNodeMaterial) can't resolve scene lights in this
    // project — the node renderer throws "Light node not found" (dual three
    // instances; see BoxView.ts / OrbitalCloud.ts).
    const mat = new MeshBasicNodeMaterial();
    const lightDir = vec3(0.408, 0.866, 0.306);
    const ndotl = max(dot(normalWorld, lightDir), float(0.0));
    const lit = ndotl.mul(0.7).add(0.3);
    mat.colorNode = vec4(vec3(...def.color).mul(lit), 1.0);

    const mesh = new InstancedMesh(def.geometry(), mat, MAX_PER_TYPE);
    // Objects roam the whole frame; skip per-instance frustum culling on the
    // (origin-centered, zero-sized) bounding sphere which would cull them all.
    mesh.frustumCulled = false;
    // Start every slot zero-scale (invisible) until activated.
    this.dummy.scale.setScalar(0);
    this.dummy.updateMatrix();
    for (let i = 0; i < MAX_PER_TYPE; i++) mesh.setMatrixAt(i, this.dummy.matrix);
    mesh.instanceMatrix.needsUpdate = true;
    this.scene.add(mesh);
    return mesh;
  }

  // Activate the next ring slot of `pool` at the origin with a random-direction
  // impulse. Recycles the oldest live object when the pool is full.
  private spawn(pool: Pool): void {
    const i = pool.next;
    pool.next = (pool.next + 1) % MAX_PER_TYPE;

    const s = (SCALE_MIN + Math.random() * (SCALE_MAX - SCALE_MIN)) * this.params[pool.def.scaleParam];
    pool.scale[i] = s;
    pool.lifetime[i] = this.params.lifetime + Math.random() * LIFETIME_JITTER_SECS;
    pool.maxLifetime[i] = pool.lifetime[i];
    pool.active[i] = 1;
    pool.def.resize(pool.colliders[i], s);

    // Random direction on the unit sphere * impulse magnitude.
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const sinPhi = Math.sin(phi);
    const k = this.params.spawnImpulse;
    const vx = sinPhi * Math.cos(theta) * k;
    const vy = sinPhi * Math.sin(theta) * k;
    const vz = Math.cos(phi) * k;

    const body = pool.bodies[i];
    body.setEnabled(true);
    body.setTranslation({ x: 0, y: 0, z: 0 }, true);
    body.setLinvel({ x: vx, y: vy, z: vz }, true);
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  update(): void {
    if (!this.world) return;
    const type = Math.round(this.params.forceFieldType);

    // noiseScale is hot — createCurlNoise closes over scale at construction.
    if (this.params.noiseScale !== this.lastNoiseScale) {
      this.curlNoise = createCurlNoise({ scale: this.params.noiseScale });
      this.lastNoiseScale = this.params.noiseScale;
    }
    // Only the Linear field uses world gravity (-Y); Curl and Attract zero it
    // and apply a per-body impulse below. Set only on change.
    const gy = type === FIELD_LINEAR ? -this.params.forceStrength : 0;
    if (gy !== this.lastGravityY) {
      this.world.gravity = { x: 0, y: gy, z: 0 };
      this.lastGravityY = gy;
    }
    // Hot param sweeps over the full pools (active or not — cheap, and keeps
    // recycled bodies correct). Guarded so we don't churn setters every frame.
    if (this.params.damping !== this.lastDamping) {
      for (const pool of this.pools)
        for (const b of pool.bodies) {
          b.setLinearDamping(this.params.damping);
          b.setAngularDamping(this.params.damping);
        }
      this.lastDamping = this.params.damping;
    }
    if (this.params.restitution !== this.lastRestitution) {
      for (const pool of this.pools)
        for (const c of pool.colliders) c.setRestitution(this.params.restitution);
      this.lastRestitution = this.params.restitution;
    }
    if (this.params.timescale !== this.lastTimescale) {
      this.world.timestep = (1 / 60) * this.params.timescale;
      this.lastTimescale = this.params.timescale;
    }
    if (this.params.wireframe !== this.lastWireframe) {
      const on = this.params.wireframe >= 0.5;
      for (const pool of this.pools)
        (pool.mesh.material as MeshBasicNodeMaterial).wireframe = on;
      this.lastWireframe = this.params.wireframe;
    }

    // Drain spawn requests (button + audio trigger) into the matching pool.
    for (let n = 0; n < spawnQueue.cube; n++) this.spawn(this.pools[0]);
    for (let n = 0; n < spawnQueue.sphere; n++) this.spawn(this.pools[1]);
    for (let n = 0; n < spawnQueue.disk; n++) this.spawn(this.pools[2]);
    spawnQueue.reset();

    // Continuous per-type spawn rate (objects/sec). Wall-clock dt (not physics
    // dt) so the rate is independent of timescale; clamped so a backgrounded
    // tab doesn't dump a huge burst on the next frame.
    const now = performance.now();
    const rdt = Number.isFinite(this.lastTime)
      ? Math.min((now - this.lastTime) / 1000, 0.1)
      : 0;
    this.lastTime = now;
    for (const pool of this.pools) {
      const rate = this.params[pool.def.rateParam];
      if (rate <= 0) {
        pool.accum = 0;
        continue;
      }
      pool.accum += rate * rdt;
      while (pool.accum >= 1) {
        this.spawn(pool);
        pool.accum -= 1;
      }
    }

    this.world.step();
    const dt = this.world.timestep;
    const fieldK = this.params.forceStrength * dt;

    for (const pool of this.pools) {
      for (let i = 0; i < MAX_PER_TYPE; i++) {
        if (!pool.active[i]) continue;
        const body = pool.bodies[i];

        pool.lifetime[i] -= dt;
        if (pool.lifetime[i] <= 0) {
          pool.active[i] = 0;
          body.setEnabled(false);
          this.dummy.scale.setScalar(0);
          this.dummy.position.set(0, 0, 0);
          this.dummy.quaternion.set(0, 0, 0, 1);
          this.dummy.updateMatrix();
          pool.mesh.setMatrixAt(i, this.dummy.matrix);
          continue;
        }

        const t = body.translation();
        if (type === FIELD_CURL) {
          this.curlNoise(t.x, t.y, t.z, this.curlOut);
          const v = body.linvel();
          body.setLinvel(
            {
              x: v.x + this.curlOut[0] * fieldK,
              y: v.y + this.curlOut[1] * fieldK,
              z: v.z + this.curlOut[2] * fieldK,
            },
            true,
          );
        } else if (type === FIELD_ATTRACT) {
          // Constant-magnitude pull toward the origin (unit direction * strength).
          const inv = fieldK / Math.max(Math.hypot(t.x, t.y, t.z), 1e-4);
          const v = body.linvel();
          body.setLinvel(
            { x: v.x - t.x * inv, y: v.y - t.y * inv, z: v.z - t.z * inv },
            true,
          );
        }

        const r = body.rotation();
        // Shrink linearly to 0 over the object's lifetime so it fades out of
        // existence right as it expires. The collider tracks the mesh but is
        // floored at COLLIDER_MIN_FADE so it never collapses to a degenerate
        // (tunneling-prone) near-zero size in the last instants before expiry.
        const fade = pool.lifetime[i] / pool.maxLifetime[i];
        const s = pool.scale[i] * fade;
        pool.def.resize(pool.colliders[i], pool.scale[i] * Math.max(fade, COLLIDER_MIN_FADE));
        this.dummy.position.set(t.x, t.y, t.z);
        this.dummy.quaternion.set(r.x, r.y, r.z, r.w);
        this.dummy.scale.set(s, s, s);
        this.dummy.updateMatrix();
        pool.mesh.setMatrixAt(i, this.dummy.matrix);
      }
      pool.mesh.instanceMatrix.needsUpdate = true;
    }
  }

  dispose(): void {
    this.disposed = true;
    for (const pool of this.pools) {
      this.scene.remove(pool.mesh);
      pool.mesh.geometry.dispose();
      (pool.mesh.material as MeshBasicNodeMaterial).dispose();
      pool.mesh.dispose();
    }
    this.pools = [];
    if (this.world) {
      this.world.free();
      this.world = null;
    }
  }
}
