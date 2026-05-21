import {
  InstancedMesh,
  IcosahedronGeometry,
  Object3D,
  Color,
  InstancedBufferAttribute,
} from "three";
import { MeshBasicNodeMaterial } from "three/webgpu";
import {
  vec3,
  vec4,
  float,
  dot,
  max,
  normalWorld,
  instancedBufferAttribute,
} from "three/tsl";
import RAPIER from "@dimforge/rapier3d-compat";
import { createCurlNoise } from "../curl-noise";
import type { Component, ComponentDeps } from "./Component";

const MAX_PARTICLES = 10000;
const BASE_RADIUS = 0.04;
const COLLISION_RATIO = 0.5;
const SPAWN_POINT = { x: 0, y: 0, z: 0 };
const ATTRACTOR_POSITION = { x: 0.5, y: 0, z: 0 };
// Scale factor random range. Visual radius = BASE_RADIUS * scale.
const SCALE_MIN = 0.5;
const SCALE_MAX = 1.5;
// Per-particle lifetime jitter on top of the slider value.
const LIFETIME_JITTER_SECS = 1.0;

export class ParticleView implements Component {
  static id = "particleView";
  static label = "Particle View";
  static paramPrefix = "particleView";
  static paramOpts = {
    numParticles: { min: 0, max: 0, step: 0 }, // ignored — discrete kind below
    lifetime: { min: 1, max: 10, step: 0.1 },
    noiseScale: { min: 0.1, max: 5.0, step: 0.05 },
    noiseStrength: { min: 0, max: 20, step: 0.1 },
    containerSize: { min: 0.5, max: 4, step: 0.05 },
    restitution: { min: 0, max: 1, step: 0.01 },
    damping: { min: 0, max: 2, step: 0.01 },
    attractorStrength: { min: 0, max: 50, step: 0.1 },
    attractorMinRadius: { min: 0.05, max: 0.5, step: 0.01 },
  };
  static paramDefaults = {
    numParticles: 2000,
    lifetime: 3,
    noiseScale: 1.5,
    noiseStrength: 5,
    containerSize: 1.5,
    restitution: 0.6,
    damping: 0.2,
    attractorStrength: 5,
    attractorMinRadius: 0.2,
  };
  static paramKinds = {
    numParticles: "discrete" as const,
  };
  static paramDiscreteOptions = {
    numParticles: [500, 1000, 2000, 5000, 10000],
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private numParticles: number;
  private mesh: InstancedMesh | null = null;
  private world: RAPIER.World | null = null;
  private bodies: RAPIER.RigidBody[] = [];
  private colliders: RAPIER.Collider[] = [];
  private wallColliders: RAPIER.Collider[] = [];
  private lifetimes!: Float32Array;
  private maxLifetimes!: Float32Array;
  private scales!: Float32Array;
  private dummy = new Object3D();
  private curlOut = new Float32Array(3);
  private curlNoise!: (x: number, y: number, z: number, out: Float32Array) => void;
  private lastNoiseScale = NaN;
  private lastDamping = NaN;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.params = params;
    this.numParticles = Math.round(params.numParticles);
    void this.init();
  }

  private async init(): Promise<void> {
    await RAPIER.init();
    if (this.disposed) return;

    // SoA storage. Allocated to MAX_PARTICLES once so we never re-allocate.
    this.lifetimes = new Float32Array(MAX_PARTICLES);
    this.maxLifetimes = new Float32Array(MAX_PARTICLES);
    this.scales = new Float32Array(MAX_PARTICLES);

    this.curlNoise = createCurlNoise({ scale: this.params.noiseScale });

    this.world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    this.addWalls(this.params.containerSize);
    this.spawnBodies(this.numParticles);

    // InstancedMesh allocated to MAX_PARTICLES; mesh.count controls how many
    // we render. Per-instance color via our own InstancedBufferAttribute.
    const colorArr = new Float32Array(MAX_PARTICLES * 3);
    const tmpColor = new Color(1, 1, 1);
    for (let i = 0; i < MAX_PARTICLES; i++) tmpColor.toArray(colorArr, i * 3);
    const colorAttr = new InstancedBufferAttribute(colorArr, 3);

    const mat = new MeshBasicNodeMaterial();
    const instColor = vec3(instancedBufferAttribute(colorAttr, "vec3", 3, 0));
    const lightDir = vec3(0.408, 0.866, 0.306);
    const ndotl = max(dot(normalWorld, lightDir), float(0.0));
    const lit = ndotl.mul(0.7).add(0.3);
    mat.colorNode = vec4(instColor.mul(lit), 1.0);

    const geom = new IcosahedronGeometry(BASE_RADIUS, 1);
    const mesh = new InstancedMesh(geom, mat, MAX_PARTICLES);
    mesh.count = this.numParticles;
    this.mesh = mesh;
    this.scene.add(mesh);
  }

  private addWalls(half: number): void {
    if (!this.world) return;
    // Six thin static box colliders forming a closed cube of half-extent
    // `half`. Thin so they don't visibly occupy the scene; restitution from
    // the body side dominates the bounce.
    const t = 0.05; // wall thickness
    const make = (
      hx: number,
      hy: number,
      hz: number,
      x: number,
      y: number,
      z: number,
    ) => {
      const desc = RAPIER.ColliderDesc.cuboid(hx, hy, hz)
        .setTranslation(x, y, z)
        .setRestitution(this.params.restitution);
      this.wallColliders.push(this.world!.createCollider(desc));
    };
    make(t, half + t, half + t, half + t, 0, 0); // +x
    make(t, half + t, half + t, -(half + t), 0, 0); // -x
    make(half + t, t, half + t, 0, half + t, 0); // +y
    make(half + t, t, half + t, 0, -(half + t), 0); // -y
    make(half + t, half + t, t, 0, 0, half + t); // +z
    make(half + t, half + t, t, 0, 0, -(half + t)); // -z
  }

  private spawnBodies(n: number): void {
    if (!this.world) return;
    const c = this.params.containerSize;
    for (let i = 0; i < n; i++) {
      const x = (Math.random() - 0.5) * 2 * c * 0.7;
      const y = (Math.random() - 0.5) * 2 * c * 0.7;
      const z = (Math.random() - 0.5) * 2 * c * 0.7;
      const scale = SCALE_MIN + Math.random() * (SCALE_MAX - SCALE_MIN);
      this.scales[i] = scale;
      this.maxLifetimes[i] = this.params.lifetime + Math.random() * LIFETIME_JITTER_SECS;
      this.lifetimes[i] = Math.random() * this.maxLifetimes[i]; // stagger initial expirations
      const body = this.world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic()
          .setTranslation(x, y, z)
          .setLinvel(
            (Math.random() - 0.5),
            (Math.random() - 0.5),
            (Math.random() - 0.5),
          )
          .setLinearDamping(this.params.damping)
          .setAngularDamping(this.params.damping),
      );
      const collider = this.world.createCollider(
        RAPIER.ColliderDesc.ball(BASE_RADIUS * scale * COLLISION_RATIO)
          .setRestitution(this.params.restitution),
        body,
      );
      this.bodies.push(body);
      this.colliders.push(collider);
    }
  }

  private respawn(i: number): void {
    const body = this.bodies[i];
    const newScale = SCALE_MIN + Math.random() * (SCALE_MAX - SCALE_MIN);
    this.scales[i] = newScale;
    this.maxLifetimes[i] = this.params.lifetime + Math.random() * LIFETIME_JITTER_SECS;
    this.lifetimes[i] = this.maxLifetimes[i];
    body.setTranslation(SPAWN_POINT, true);
    body.setLinvel(
      { x: (Math.random() - 0.5), y: (Math.random() - 0.5), z: (Math.random() - 0.5) },
      true,
    );
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    this.colliders[i].setRadius(BASE_RADIUS * newScale * COLLISION_RATIO);
  }

  update(): void {
    if (!this.world || !this.mesh) return;
    const noiseStrength = this.params.noiseStrength;
    const attractorStrength = this.params.attractorStrength;
    const attractorMinRadius = this.params.attractorMinRadius;
    // noiseScale is a hot param — re-create the noise function only when
    // the slider value changes. createCurlNoise's `scale` is closed over
    // at construction, so there's no per-call way to vary it.
    if (this.params.noiseScale !== this.lastNoiseScale) {
      this.curlNoise = createCurlNoise({ scale: this.params.noiseScale });
      this.lastNoiseScale = this.params.noiseScale;
    }
    // damping is hot — sweep all bodies only when the slider moved.
    // Without this guard we'd burn ~1.2M setter calls/s at 10k particles.
    if (this.params.damping !== this.lastDamping) {
      for (const b of this.bodies) {
        b.setLinearDamping(this.params.damping);
        b.setAngularDamping(this.params.damping);
      }
      this.lastDamping = this.params.damping;
    }
    this.world.step();
    const dt = this.world.timestep;

    for (let i = 0; i < this.numParticles; i++) {
      const body = this.bodies[i];

      this.lifetimes[i] -= dt;
      if (this.lifetimes[i] <= 0) {
        this.respawn(i);
      }

      const t = body.translation();
      // Curl noise as a velocity impulse — same pattern BoxView uses for
      // its spring force. Additive on linvel; cheap and stable.
      this.curlNoise(t.x, t.y, t.z, this.curlOut);
      const v = body.linvel();
      body.setLinvel(
        {
          x: v.x + this.curlOut[0] * noiseStrength * dt,
          y: v.y + this.curlOut[1] * noiseStrength * dt,
          z: v.z + this.curlOut[2] * noiseStrength * dt,
        },
        true,
      );

      // Newtonian attractor: F = strength * (A - p) / |A - p|^3.
      // Clamp |A - p| at attractorMinRadius to avoid singularity at r=0.
      if (attractorStrength > 0) {
        const dx = ATTRACTOR_POSITION.x - t.x;
        const dy = ATTRACTOR_POSITION.y - t.y;
        const dz = ATTRACTOR_POSITION.z - t.z;
        const r2 = dx * dx + dy * dy + dz * dz;
        const r = Math.sqrt(r2);
        if (r >= attractorMinRadius) {
          const invR3 = 1 / (r * r2);
          const k = attractorStrength * invR3;
          body.addForce({ x: dx * k, y: dy * k, z: dz * k }, true);
        }
      }

      const r = body.rotation();
      const s = this.scales[i];
      this.dummy.position.set(t.x, t.y, t.z);
      this.dummy.quaternion.set(r.x, r.y, r.z, r.w);
      this.dummy.scale.set(s, s, s);
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this.disposed = true;
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as MeshBasicNodeMaterial).dispose();
      this.mesh.dispose();
      this.mesh = null;
    }
    if (this.world) {
      // Frees all bodies + colliders too.
      this.world.free();
      this.world = null;
    }
    this.bodies = [];
    this.colliders = [];
    this.wallColliders = [];
  }
}
