import {
  InstancedMesh,
  BoxGeometry,
  IcosahedronGeometry,
  CylinderGeometry,
  BufferGeometry,
  Object3D,
  Color,
} from "three";
import type { MeshStandardNodeMaterial } from "three/webgpu";
import { makeLitMaterial, releaseLitMaterial } from "./litMaterial";
import { vec4, uniform } from "three/tsl";
import RAPIER from "@dimforge/rapier3d-compat";
import { createCurlNoise } from "../curl-noise";
import { getPhysicsWorld } from "./physics";
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
// Floor on the spawn radius, so no two objects are ever born at EXACTLY the
// same point. Perfectly coincident shapes leave parry deriving the contact
// normal from a zero-length separation vector; that degenerate case panics
// rapier inside world.step(), which aborts the wasm module outright
// ("RuntimeError: unreachable") rather than throwing something catchable.
// Three orders of magnitude under BASE, so it's visually a no-op.
const SPAWN_JITTER = 1e-4;

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
  // Param key holding this shape's 0xRRGGBB color.
  colorParam: string;
  // Param key holding this shape's per-object size multiplier.
  scaleParam: string;
  // Exactly one of the two below. `rateParam` = objects/second, spawned at the
  // origin (what the buttons and audio triggers push into). `ambient` = hold a
  // live population of `amountParam` instead: physics and force fields are
  // identical, only the spawn policy differs — they appear spread through a
  // ball of `radiusParam` at `driftParam` speed and are topped back up as they
  // expire, so the scene always has that many of them to hit.
  rateParam?: string;
  ambient?: { amountParam: string; radiusParam: string; driftParam: string };
}

const SHAPES: ShapeDef[] = [
  {
    geometry: () => new BoxGeometry(BASE * 2, BASE * 2, BASE * 2),
    collider: () => RAPIER.ColliderDesc.cuboid(BASE, BASE, BASE),
    resize: (c, s) => c.setHalfExtents({ x: BASE * s, y: BASE * s, z: BASE * s }),
    colorParam: "cubeColor",
    scaleParam: "cubeScale",
    rateParam: "cubeRate",
  },
  {
    geometry: () => new IcosahedronGeometry(BASE, 2),
    collider: () => RAPIER.ColliderDesc.ball(BASE),
    resize: (c, s) => c.setRadius(BASE * s),
    colorParam: "sphereColor",
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
    colorParam: "diskColor",
    scaleParam: "diskScale",
    rateParam: "diskRate",
  },
  {
    geometry: () => new IcosahedronGeometry(BASE, 2),
    collider: () => RAPIER.ColliderDesc.ball(BASE),
    resize: (c, s) => c.setRadius(BASE * s),
    colorParam: "sphere2Color",
    scaleParam: "sphere2Scale",
    ambient: {
      amountParam: "sphere2Amount",
      radiusParam: "sphere2Radius",
      driftParam: "sphere2Drift",
    },
  },
];

function randUnit(out: Float32Array): void {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const sinPhi = Math.sin(phi);
  out[0] = sinPhi * Math.cos(theta);
  out[1] = sinPhi * Math.sin(theta);
  out[2] = Math.cos(phi);
}

// One shape's ring buffer: fixed body/collider pool + parallel SoA slot state.
// Plain data; all logic lives in Spawner.
class Pool {
  bodies: RAPIER.RigidBody[] = [];
  colliders: RAPIER.Collider[] = [];
  active = new Uint8Array(MAX_PER_TYPE);
  lifetime = new Float32Array(MAX_PER_TYPE);
  maxLifetime = new Float32Array(MAX_PER_TYPE);
  scale = new Float32Array(MAX_PER_TYPE);
  // Fade factor last pushed into the collider — resizes are throttled to >3%
  // deltas because every resize dirties the broad-phase.
  fadeApplied = new Float32Array(MAX_PER_TYPE);
  next = 0;
  // Shape color, referenced live by the material's uniform node — mutate it in
  // place (setHex) and the next frame picks it up; never reassign.
  color = new Color();
  lastColor = NaN;
  // Live slot count, maintained by spawn/expire. Ambient pools top up against
  // it every frame; counting the `active` array instead would be a 512-slot
  // scan per frame for the same number.
  live = 0;
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
    cubeScale: { min: 0.2, max: 9, step: 0.05 },
    sphereScale: { min: 0.2, max: 9, step: 0.05 },
    diskScale: { min: 0.2, max: 9, step: 0.05 },
    sphere2Scale: { min: 0.2, max: 9, step: 0.05 },
    cubeRate: { min: 0, max: 50, step: 0.5 },
    sphereRate: { min: 0, max: 50, step: 0.5 },
    diskRate: { min: 0, max: 50, step: 0.5 },
    sphere2Amount: { min: 0, max: MAX_PER_TYPE, step: 1 },
    sphere2Radius: { min: 0, max: 6, step: 0.05 },
    sphere2Drift: { min: 0, max: 3, step: 0.01 },
  };
  static paramDefaults = {
    forceFieldType: 0,
    forceStrength: 9.8,
    noiseScale: 0.5,
    spawnImpulse: 3,
    lifetime: 4,
    restitution: 0.5,
    damping: 0.1,
    cubeScale: 1.0,
    sphereScale: 1.0,
    diskScale: 1.0,
    sphere2Scale: 1.0,
    cubeRate: 0,
    sphereRate: 0,
    diskRate: 0,
    // Non-zero so there's something in the scene to hit the moment the module
    // is on — the impulse shapes stay at 0 and wait for a button or trigger.
    sphere2Amount: 60,
    sphere2Radius: 2.0,
    sphere2Drift: 0.25,
    // sRGB packed defaults — these are the previously hardcoded linear colors
    // encoded to sRGB, so the picker's swatch and the render agree.
    cubeColor: 0xf9b389,
    sphereColor: 0x95daf9,
    diskColor: 0xcbf3aa,
    sphere2Color: 0xddd4f9,
    wireframe: 0,
  };
  static paramKinds = {
    forceFieldType: "discrete" as const,
    wireframe: "discrete" as const,
    cubeColor: "color" as const,
    sphereColor: "color" as const,
    diskColor: "color" as const,
    sphere2Color: "color" as const,
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
  private dirScratch = new Float32Array(3);
  private curlNoise!: (x: number, y: number, z: number, out: Float32Array) => void;
  private lastNoiseScale = NaN;
  private lastDamping = NaN;
  private lastRestitution = NaN;
  private lastTime = NaN;
  private lastWireframe = NaN;
  private lastFieldType = NaN;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    spawnQueue.reset();
    void this.init();
  }

  private async init(): Promise<void> {
    const world = await getPhysicsWorld();
    if (this.disposed) return;
    this.world = world;

    this.curlNoise = createCurlNoise({ scale: this.params.noiseScale });
    this.lastNoiseScale = this.params.noiseScale;
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
      pool.mesh = this.createMesh(pool);
      this.pools.push(pool);
    }
  }

  private createMesh(pool: Pool): InstancedMesh {
    // Real lit material. The old hand-rolled lambert predated the vite alias
    // that collapsed the dual three instances ("Light node not found") —
    // scene lights resolve now, and a lit material is what receives shadows.
    const mat = makeLitMaterial();
    // The uniform holds the pool's Color by reference; update() mutates it when
    // the param moves, no material rebuild.
    pool.color.setHex(this.params[pool.def.colorParam]);
    pool.lastColor = this.params[pool.def.colorParam];
    mat.colorNode = vec4(uniform(pool.color), 1.0);

    const mesh = new InstancedMesh(pool.def.geometry(), mat, MAX_PER_TYPE);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
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

  // Activate the next ring slot of `pool`. Regular shapes launch from the
  // origin with a random-direction impulse; ambient pools appear anywhere
  // inside their radius and drift. Recycles the oldest live object when the
  // pool is full.
  //
  // `staggered` starts the object part-way through its life instead of at the
  // beginning — see the ambient top-up in update() for why.
  private spawn(pool: Pool, staggered = false): void {
    const i = pool.next;
    pool.next = (pool.next + 1) % MAX_PER_TYPE;

    const s = (SCALE_MIN + Math.random() * (SCALE_MAX - SCALE_MIN)) * this.params[pool.def.scaleParam];
    pool.scale[i] = s;
    const full = this.params.lifetime + Math.random() * LIFETIME_JITTER_SECS;
    pool.maxLifetime[i] = full;
    pool.lifetime[i] = staggered ? Math.random() * full : full;
    // Recycling a still-live slot replaces it rather than adding to the count.
    if (!pool.active[i]) pool.live++;
    pool.active[i] = 1;
    pool.def.resize(pool.colliders[i], s);
    pool.fadeApplied[i] = 1;

    const amb = pool.def.ambient;
    const d = this.dirScratch;
    // Position: the origin for impulse shapes; a uniform point in the ball of
    // radius sphere2Radius for ambient ones (cbrt spreads them evenly through
    // the volume instead of clumping at the center). SPAWN_JITTER floors both
    // — impulse shapes because they'd otherwise all stack on the origin, and
    // ambient ones because sphere2Radius goes to 0.
    randUnit(d);
    const spread = Math.max(amb ? this.params[amb.radiusParam] : 0, SPAWN_JITTER);
    const r = spread * Math.cbrt(Math.random());
    const px = d[0] * r, py = d[1] * r, pz = d[2] * r;
    // Fresh direction for the drift so ambient spheres don't all stream
    // radially outward from wherever they appeared.
    if (amb) randUnit(d);
    const k = amb ? this.params[amb.driftParam] : this.params.spawnImpulse;

    const body = pool.bodies[i];
    body.setEnabled(true);
    body.setTranslation({ x: px, y: py, z: pz }, true);
    body.setLinvel({ x: d[0] * k, y: d[1] * k, z: d[2] * k }, true);
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  update(): void {
    if (!this.world) return;
    const type = Math.round(this.params.forceFieldType);
    // Fields skip sleeping bodies, so a settled pile would ignore the
    // dropdown; one wake-all on type change keeps it responsive.
    if (type !== this.lastFieldType) {
      if (Number.isFinite(this.lastFieldType)) {
        for (const pool of this.pools)
          for (let i = 0; i < MAX_PER_TYPE; i++)
            if (pool.active[i]) pool.bodies[i].wakeUp();
      }
      this.lastFieldType = type;
    }

    // noiseScale is hot — createCurlNoise closes over scale at construction.
    if (this.params.noiseScale !== this.lastNoiseScale) {
      this.curlNoise = createCurlNoise({ scale: this.params.noiseScale });
      this.lastNoiseScale = this.params.noiseScale;
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
    // Colors are hot: mutate the Color the material's uniform already points at.
    for (const pool of this.pools) {
      const hex = this.params[pool.def.colorParam];
      if (hex === pool.lastColor) continue;
      pool.color.setHex(hex);
      pool.lastColor = hex;
    }
    if (this.params.wireframe !== this.lastWireframe) {
      const on = this.params.wireframe >= 0.5;
      for (const pool of this.pools)
        (pool.mesh.material as MeshStandardNodeMaterial).wireframe = on;
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
      const amb = pool.def.ambient;
      if (amb) {
        // Population target, not a rate: refill whatever expired since the last
        // frame. Clamped to the pool so a maxed-out slider can't spin forever.
        // Lowering the slider doesn't cull anyone — the surplus just ages out.
        const target = Math.min(Math.round(this.params[amb.amountParam]), MAX_PER_TYPE);
        // Filling more than one slot in a frame is either the first fill or an
        // amount increase. Born together they also DIE together (only the ~1s
        // lifetime jitter separates them), which empties the pool, triggers
        // another batch, and locks the population into a visible oscillation.
        // Staggering their starting ages across a full lifetime spreads the
        // deaths out permanently: from then on slots free up a few at a time
        // and each replacement gets a normal full life.
        const staggered = target - pool.live > 1;
        while (pool.live < target) {
          // Walk the ring to a free slot first: recycling a live object would
          // reset a perfectly good sphere's lifetime without growing the
          // population, and the loop could spin. live < target <= MAX_PER_TYPE
          // guarantees a free slot exists, so this always terminates.
          while (pool.active[pool.next]) pool.next = (pool.next + 1) % MAX_PER_TYPE;
          this.spawn(pool, staggered);
        }
        continue;
      }
      const rate = pool.def.rateParam ? this.params[pool.def.rateParam] : 0;
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

    // App stepped the shared world already; read its dt to scale the fields.
    const dt = this.world.timestep;
    const fieldK = this.params.forceStrength * dt;

    for (const pool of this.pools) {
      for (let i = 0; i < MAX_PER_TYPE; i++) {
        if (!pool.active[i]) continue;
        const body = pool.bodies[i];

        pool.lifetime[i] -= dt;
        if (pool.lifetime[i] <= 0) {
          pool.active[i] = 0;
          pool.live--;
          body.setEnabled(false);
          this.dummy.scale.setScalar(0);
          this.dummy.position.set(0, 0, 0);
          this.dummy.quaternion.set(0, 0, 0, 1);
          this.dummy.updateMatrix();
          pool.mesh.setMatrixAt(i, this.dummy.matrix);
          continue;
        }

        const t = body.translation();
        // Sleep etiquette: sleeping bodies are skipped AND the field writes
        // pass wake=false — the wake flag resets the sleep timer, so with
        // wake=true a body at rest could never fall asleep in the first
        // place. The type-change wake above keeps the dropdown live.
        if (body.isSleeping()) {
          // Skip field writes but keep fading/rendering below.
        } else if (type === FIELD_LINEAR) {
          // Per-body rather than world gravity: the world is shared now, so a
          // component can't own world.gravity. dv = -strength*dt is exactly
          // what a gravity of -forceStrength would have applied.
          const v = body.linvel();
          body.setLinvel({ x: v.x, y: v.y - fieldK, z: v.z }, false);
        } else if (type === FIELD_CURL) {
          this.curlNoise(t.x, t.y, t.z, this.curlOut);
          const v = body.linvel();
          body.setLinvel(
            {
              x: v.x + this.curlOut[0] * fieldK,
              y: v.y + this.curlOut[1] * fieldK,
              z: v.z + this.curlOut[2] * fieldK,
            },
            false,
          );
        } else if (type === FIELD_ATTRACT) {
          // Constant-magnitude pull toward the origin (unit direction * strength).
          const inv = fieldK / Math.max(Math.hypot(t.x, t.y, t.z), 1e-4);
          const v = body.linvel();
          body.setLinvel(
            { x: v.x - t.x * inv, y: v.y - t.y * inv, z: v.z - t.z * inv },
            false,
          );
        }

        const r = body.rotation();
        // Shrink linearly to 0 over the object's lifetime so it fades out of
        // existence right as it expires. The collider tracks the mesh but is
        // floored at COLLIDER_MIN_FADE so it never collapses to a degenerate
        // (tunneling-prone) near-zero size in the last instants before expiry.
        const fade = pool.lifetime[i] / pool.maxLifetime[i];
        const s = pool.scale[i] * fade;
        // Throttled: a resize dirties the broad-phase even at identical size,
        // so per-frame resizes made every active object re-pair every frame.
        // ~3% steps land ~10 resizes over a lifetime instead.
        const cf = Math.max(fade, COLLIDER_MIN_FADE);
        if (Math.abs(cf - pool.fadeApplied[i]) > 0.03) {
          pool.def.resize(pool.colliders[i], pool.scale[i] * cf);
          pool.fadeApplied[i] = cf;
        }
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
    // Shared world: remove our own bodies (colliders go with them), never free
    // it — that would invalidate every other component's handles. Has to run
    // before this.pools is cleared.
    if (this.world) {
      for (const pool of this.pools)
        for (const b of pool.bodies) this.world.removeRigidBody(b);
      this.world = null;
    }
    for (const pool of this.pools) {
      this.scene.remove(pool.mesh);
      pool.mesh.geometry.dispose();
      releaseLitMaterial(pool.mesh.material as MeshStandardNodeMaterial);
      pool.mesh.dispose();
    }
    this.pools = [];
  }
}
