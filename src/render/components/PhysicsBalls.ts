import { InstancedMesh, IcosahedronGeometry, Object3D, Color } from "three";
import { MeshBasicNodeMaterial } from "three/webgpu";
import { vec4, uniform } from "three/tsl";
import RAPIER from "@dimforge/rapier3d-simd-compat";
import { getPhysicsWorld } from "./physics";
import type { Component, ComponentDeps } from "./Component";

// Fixed pool ceiling, well under the 512-instance mark Spawner already relies
// on staying under the WebGPU uniform-buffer cliff (see ParticleView) — real
// rigid-body collision solving is also far pricier per-instance than
// Fireflies' pure drift, so the cap stays modest.
const MAX_BALLS = 256;
// Per-ball radius = size * clamp(1 + sizeSeed*sizeRandomness, MIN, MAX). Seed
// is a fixed per-slot random in [-1, 1], so sizeRandomness redistributes
// existing balls instead of re-rolling them.
const RADIUS_FACTOR_MIN = 0.2;
const RADIUS_FACTOR_MAX = 2.2;

// A spherical ball pit sharing the rapier world: real dynamic bodies that
// collide with each other AND with every other physics component (Spawner,
// Serpent), fall under physics.gravity, and are pulled toward the origin by a
// constant spring. Unlit + emissive like Fireflies: color x brightness with
// brightness able to exceed 1 gives the bloom threshold something to clip,
// and it's opaque (not additive) so it still writes depth for DOF.
export class PhysicsBalls implements Component {
  static id = "physicsBalls";
  static label = "Physics Balls";
  static paramPrefix = "physicsBalls";
  static paramOpts = {
    zone: { min: 0.3, max: 5, step: 0.05 },
    size: { min: 0.01, max: 0.3, step: 0.005 },
    sizeRandomness: { min: 0, max: 1, step: 0.01 },
    restitution: { min: 0, max: 1, step: 0.01 },
    damping: { min: 0, max: 2, step: 0.01 },
    containStrength: { min: 0, max: 30, step: 0.1 },
    brightness: { min: 0, max: 8, step: 0.05 },
  };
  static paramDefaults = {
    count: 48,
    zone: 1.2,
    size: 0.06,
    sizeRandomness: 0.4,
    restitution: 0.5,
    damping: 0.3,
    containStrength: 10,
    brightness: 3,
    color: 0x8ec9ff,
  };
  static paramKinds = {
    count: "discrete" as const,
    color: "color" as const,
  };
  static paramDiscreteOptions = {
    count: [8, 16, 32, 64, 128, 256],
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private world: RAPIER.World | null = null;
  private mesh: InstancedMesh | null = null;
  private bodies: RAPIER.RigidBody[] = [];
  private colliders: RAPIER.Collider[] = [];
  private active = new Uint8Array(MAX_BALLS);
  // Fixed per-slot radius jitter, rolled once at pool creation.
  private sizeSeed = new Float32Array(MAX_BALLS);
  private live = 0;
  // Referenced live by the material's uniform node; mutate in place, never
  // reassign, so the material keeps pointing at the same instance. Holds
  // color x brightness pre-multiplied — Color stores plain floats, so a
  // brightness > 1 rides along unclamped straight into the bloom pass.
  private color = new Color();
  private lastColor = NaN;
  private lastBrightness = NaN;
  private lastSize = NaN;
  private lastSizeRandomness = NaN;
  private lastRestitution = NaN;
  private lastDamping = NaN;
  private dummy = new Object3D();
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    void this.init();
  }

  private async init(): Promise<void> {
    const world = await getPhysicsWorld();
    if (this.disposed) return;
    this.world = world;

    for (let i = 0; i < MAX_BALLS; i++) this.sizeSeed[i] = Math.random() * 2 - 1;

    for (let i = 0; i < MAX_BALLS; i++) {
      const body = world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic()
          .setLinearDamping(this.params.damping)
          .setAngularDamping(this.params.damping)
          .setEnabled(false),
      );
      const collider = world.createCollider(
        RAPIER.ColliderDesc.ball(this.radiusOf(i)).setRestitution(this.params.restitution),
        body,
      );
      this.bodies.push(body);
      this.colliders.push(collider);
    }
    this.lastDamping = this.params.damping;
    this.lastRestitution = this.params.restitution;
    this.lastSize = this.params.size;
    this.lastSizeRandomness = this.params.sizeRandomness;

    this.mesh = this.createMesh();
    this.setCount(Math.round(this.params.count));
  }

  private createMesh(): InstancedMesh {
    const mat = new MeshBasicNodeMaterial();
    this.color.setHex(this.params.color).multiplyScalar(this.params.brightness);
    this.lastColor = this.params.color;
    this.lastBrightness = this.params.brightness;
    // Unlit: colorNode is the final emitted color, so it IS the emission.
    mat.colorNode = vec4(uniform(this.color), 1.0);

    const mesh = new InstancedMesh(new IcosahedronGeometry(1, 1), mat, MAX_BALLS);
    // Balls roam the whole zone; skip per-instance frustum culling on the
    // (origin-centered, zero-sized) bounding sphere which would cull them all.
    mesh.frustumCulled = false;
    this.dummy.scale.setScalar(0);
    this.dummy.updateMatrix();
    for (let i = 0; i < MAX_BALLS; i++) mesh.setMatrixAt(i, this.dummy.matrix);
    mesh.instanceMatrix.needsUpdate = true;
    this.scene.add(mesh);
    return mesh;
  }

  private radiusOf(i: number): number {
    const f = 1 + this.sizeSeed[i] * this.params.sizeRandomness;
    return this.params.size * Math.min(RADIUS_FACTOR_MAX, Math.max(RADIUS_FACTOR_MIN, f));
  }

  // Uniform point in the ball of radius `zone` (cbrt of the radial random, so
  // balls aren't clumped at the center).
  private seed(i: number): void {
    const r = this.params.zone * Math.cbrt(Math.random());
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const sinPhi = Math.sin(phi);
    const body = this.bodies[i];
    body.setEnabled(true);
    body.setTranslation(
      { x: r * sinPhi * Math.cos(theta), y: r * sinPhi * Math.sin(theta), z: r * Math.cos(phi) },
      true,
    );
    body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  // Grow activates the next range of slots (fresh position in the zone);
  // shrink disables the top of the range and hides its instance.
  private setCount(n: number): void {
    for (let i = n; i < this.live; i++) {
      this.active[i] = 0;
      this.bodies[i].setEnabled(false);
      this.dummy.scale.setScalar(0);
      this.dummy.position.set(0, 0, 0);
      this.dummy.updateMatrix();
      this.mesh!.setMatrixAt(i, this.dummy.matrix);
    }
    for (let i = this.live; i < n; i++) {
      this.active[i] = 1;
      this.seed(i);
    }
    this.live = n;
    if (this.mesh) this.mesh.instanceMatrix.needsUpdate = true;
  }

  update(): void {
    if (!this.world || !this.mesh) return;

    const n = Math.round(this.params.count);
    if (n !== this.live) this.setCount(n);

    if (this.params.color !== this.lastColor || this.params.brightness !== this.lastBrightness) {
      this.color.setHex(this.params.color).multiplyScalar(this.params.brightness);
      this.lastColor = this.params.color;
      this.lastBrightness = this.params.brightness;
    }
    if (this.params.damping !== this.lastDamping) {
      for (const b of this.bodies) {
        b.setLinearDamping(this.params.damping);
        b.setAngularDamping(this.params.damping);
      }
      this.lastDamping = this.params.damping;
    }
    if (this.params.restitution !== this.lastRestitution) {
      for (const c of this.colliders) c.setRestitution(this.params.restitution);
      this.lastRestitution = this.params.restitution;
    }
    if (this.params.size !== this.lastSize || this.params.sizeRandomness !== this.lastSizeRandomness) {
      for (let i = 0; i < MAX_BALLS; i++) this.colliders[i].setRadius(this.radiusOf(i));
      this.lastSize = this.params.size;
      this.lastSizeRandomness = this.params.sizeRandomness;
    }

    // App stepped the shared world already; read its dt to scale the field.
    const dt = this.world.timestep;
    const fieldK = this.params.containStrength * dt;

    for (let i = 0; i < MAX_BALLS; i++) {
      if (!this.active[i]) continue;
      const body = this.bodies[i];
      const t = body.translation();

      // Sleep etiquette (see Spawner): skip field writes on sleeping bodies —
      // wake=true would reset the sleep timer and a settled ball could never
      // sleep in the first place.
      if (!body.isSleeping()) {
        // Constant Hooke's-law spring pulling every ball toward the origin —
        // active everywhere, not just past `zone`, so balls always have some
        // force on them instead of sitting dead still when gravity is 0.
        // `zone` still sets the spawn radius, giving the spring room to work.
        const v = body.linvel();
        body.setLinvel(
          { x: v.x - t.x * fieldK, y: v.y - t.y * fieldK, z: v.z - t.z * fieldK },
          false,
        );
      }

      const r = body.rotation();
      this.dummy.position.set(t.x, t.y, t.z);
      this.dummy.quaternion.set(r.x, r.y, r.z, r.w);
      this.dummy.scale.setScalar(this.radiusOf(i));
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this.disposed = true;
    // Shared world: remove only our own bodies (colliders ride along) and
    // clear the arrays in the same breath — a setter on a removed body panics.
    if (this.world) {
      for (const b of this.bodies) this.world.removeRigidBody(b);
      this.world = null;
    }
    this.bodies = [];
    this.colliders = [];
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as MeshBasicNodeMaterial).dispose();
      this.mesh.dispose();
      this.mesh = null;
    }
  }
}
