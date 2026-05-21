import {
  InstancedMesh,
  IcosahedronGeometry,
  Mesh,
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
    timescale: { min: 0, max: 3, step: 0.01 },
    lifetime: { min: 1, max: 10, step: 0.1 },
    noiseScale: { min: 0.1, max: 5.0, step: 0.05 },
    noiseStrength: { min: 0, max: 20, step: 0.1 },
    restitution: { min: 0, max: 1, step: 0.01 },
    damping: { min: 0, max: 2, step: 0.01 },
    attractorStrength: { min: 0, max: 50, step: 0.1 },
    attractorRadius: { min: 0.05, max: 1.0, step: 0.01 },
    spawnRadius: { min: 0.1, max: 3.0, step: 0.05 },
    swirlStrength: { min: 0, max: 20, step: 0.1 },
  };
  static paramDefaults = {
    numParticles: 2000,
    timescale: 1.0,
    lifetime: 3,
    noiseScale: 1.5,
    noiseStrength: 2,
    restitution: 0.6,
    damping: 0.2,
    attractorStrength: 5,
    attractorRadius: 0.25,
    spawnRadius: 1.0,
    swirlStrength: 3,
  };
  static paramKinds = {
    numParticles: "discrete" as const,
  };
  static paramDiscreteOptions = {
    numParticles: [500, 1000, 2000, 5000, 10000],
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private paramStore: ComponentDeps["paramStore"];
  private numParticles: number;
  private mesh: InstancedMesh | null = null;
  private world: RAPIER.World | null = null;
  private bodies: RAPIER.RigidBody[] = [];
  private colliders: RAPIER.Collider[] = [];
  private attractorCollider: RAPIER.Collider | null = null;
  private attractorMesh: Mesh | null = null;
  private lifetimes!: Float32Array;
  private maxLifetimes!: Float32Array;
  private scales!: Float32Array;
  private dummy = new Object3D();
  private curlOut = new Float32Array(3);
  // Scratch buffer for orbital initial conditions: [px, py, pz, vx, vy, vz].
  // Reused across every spawn/respawn call to avoid per-particle allocation.
  private spawnState = new Float32Array(6);
  private curlNoise!: (x: number, y: number, z: number, out: Float32Array) => void;
  private lastNoiseScale = NaN;
  private lastDamping = NaN;
  private lastRestitution = NaN;
  private lastAttractorRadius = NaN;
  private lastTimescale = NaN;
  private storeUnsub: (() => void) | null = null;
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.paramStore = deps.paramStore;
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
    this.addAttractor(this.params.attractorRadius);
    this.spawnBodies(this.numParticles);

    // Visible attractor sphere — non-instanced single Mesh at fixed position.
    // Geometry is unit radius; mesh.scale carries the actual radius so the
    // attractorRadius hot-sweep is a single scale update, no geometry rebuild.
    const aMat = new MeshBasicNodeMaterial();
    const aLightDir = vec3(0.408, 0.866, 0.306);
    const aNdotl = max(dot(normalWorld, aLightDir), float(0.0));
    const aLit = aNdotl.mul(0.7).add(0.3);
    aMat.colorNode = vec4(vec3(1.0, 0.6, 0.2).mul(aLit), 1.0);
    const aGeom = new IcosahedronGeometry(1, 2);
    const aMesh = new Mesh(aGeom, aMat);
    aMesh.position.set(ATTRACTOR_POSITION.x, ATTRACTOR_POSITION.y, ATTRACTOR_POSITION.z);
    aMesh.scale.setScalar(this.params.attractorRadius);
    this.attractorMesh = aMesh;
    this.scene.add(aMesh);

    this.createInstancedMesh(this.numParticles);

    // Seed the lastFoo guards now that init() created the initial
    // curlNoise (and damping/restitution will sweep on first update if
    // their lastFoo stays NaN — leave those NaN to force the sweep).
    this.lastNoiseScale = this.params.noiseScale;

    // Listen for reconfig param changes. Hot params (lifetime, noiseScale,
    // noiseStrength, restitution, attractorStrength, attractorRadius,
    // spawnRadius, swirlStrength, timescale, damping) are read from
    // this.params each frame via the bag — no subscription needed.
    this.storeUnsub = this.paramStore.subscribe((key, value) => {
      if (this.disposed) return;
      if (key === "particleView.numParticles" && typeof value === "number") {
        const n = Math.round(value);
        if (n !== this.numParticles) {
          this.rebuildBodies(n);
        }
      }
    });
  }

  // Allocate the InstancedMesh to *exactly* `n` instances, not MAX_PARTICLES.
  // Three.js's WebGPU backend binds the per-instance matrix buffer as a
  // uniform buffer at small instance counts, capped at 64KB. MAX_PARTICLES=10000
  // produces a 640KB buffer that exceeds the uniform limit and causes silent
  // render failures below ~1024 instances (where the backend doesn't switch
  // to a storage buffer). Sizing to numParticles avoids that path entirely.
  private createInstancedMesh(n: number): void {
    const colorArr = new Float32Array(n * 3);
    const tmpColor = new Color(1, 1, 1);
    for (let i = 0; i < n; i++) tmpColor.toArray(colorArr, i * 3);
    const colorAttr = new InstancedBufferAttribute(colorArr, 3);

    const mat = new MeshBasicNodeMaterial();
    const instColor = vec3(instancedBufferAttribute(colorAttr, "vec3", 3, 0));
    const lightDir = vec3(0.408, 0.866, 0.306);
    const ndotl = max(dot(normalWorld, lightDir), float(0.0));
    const lit = ndotl.mul(0.7).add(0.3);
    mat.colorNode = vec4(instColor.mul(lit), 1.0);

    const geom = new IcosahedronGeometry(BASE_RADIUS, 1);
    const mesh = new InstancedMesh(geom, mat, n);
    this.mesh = mesh;
    this.scene.add(mesh);
  }

  private disposeInstancedMesh(): void {
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    (this.mesh.material as MeshBasicNodeMaterial).dispose();
    this.mesh.dispose();
    this.mesh = null;
  }

  private addAttractor(radius: number): void {
    if (!this.world) return;
    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(
        ATTRACTOR_POSITION.x,
        ATTRACTOR_POSITION.y,
        ATTRACTOR_POSITION.z,
      ),
    );
    this.attractorCollider = this.world.createCollider(
      RAPIER.ColliderDesc.ball(radius).setRestitution(this.params.restitution),
      body,
    );
  }

  // Write orbital initial conditions for a single particle into spawnState:
  // position = random point on a sphere of radius spawnRadius around the
  // attractor; velocity = tangential, magnitude = sqrt(attractorStrength).
  // For our 1/r force law, v = sqrt(strength) is the circular-orbit speed
  // (independent of radius), so freshly spawned particles immediately settle
  // into roughly circular orbits. 0.8..1.2 jitter so they aren't synced.
  private orbitalInit(): void {
    const r = this.params.spawnRadius;
    // Uniformly random unit vector on the unit sphere.
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const sinPhi = Math.sin(phi);
    const dx = sinPhi * Math.cos(theta);
    const dy = sinPhi * Math.sin(theta);
    const dz = Math.cos(phi);
    // Tangential direction = normalize(cross(d, up)). Use Y as the up axis;
    // fall back to X when d is near-parallel to Y so the cross isn't degenerate.
    let ax = 0, ay = 1, az = 0;
    if (Math.abs(dy) > 0.95) { ax = 1; ay = 0; az = 0; }
    let tx = dy * az - dz * ay;
    let ty = dz * ax - dx * az;
    let tz = dx * ay - dy * ax;
    const tmag = Math.sqrt(tx * tx + ty * ty + tz * tz);
    tx /= tmag; ty /= tmag; tz /= tmag;
    const speed = Math.sqrt(this.params.attractorStrength) * (0.8 + Math.random() * 0.4);
    this.spawnState[0] = ATTRACTOR_POSITION.x + dx * r;
    this.spawnState[1] = ATTRACTOR_POSITION.y + dy * r;
    this.spawnState[2] = ATTRACTOR_POSITION.z + dz * r;
    this.spawnState[3] = tx * speed;
    this.spawnState[4] = ty * speed;
    this.spawnState[5] = tz * speed;
  }

  private spawnBodies(n: number): void {
    if (!this.world) return;
    const s = this.spawnState;
    for (let i = 0; i < n; i++) {
      this.orbitalInit();
      const scale = SCALE_MIN + Math.random() * (SCALE_MAX - SCALE_MIN);
      this.scales[i] = scale;
      this.maxLifetimes[i] = this.params.lifetime + Math.random() * LIFETIME_JITTER_SECS;
      this.lifetimes[i] = Math.random() * this.maxLifetimes[i]; // stagger initial expirations
      const body = this.world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic()
          .setTranslation(s[0], s[1], s[2])
          .setLinvel(s[3], s[4], s[5])
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
    this.orbitalInit();
    const s = this.spawnState;
    body.setTranslation({ x: s[0], y: s[1], z: s[2] }, true);
    body.setLinvel({ x: s[3], y: s[4], z: s[5] }, true);
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    this.colliders[i].setRadius(BASE_RADIUS * newScale * COLLISION_RATIO);
  }

  private rebuildBodies(n: number): void {
    if (!this.world) return;
    // Free the entire world (drops all bodies + colliders), recreate it,
    // re-add walls, spawn the new body pool. Also dispose + recreate the
    // InstancedMesh at the new size — see createInstancedMesh for why we
    // size to numParticles rather than reusing one mesh at MAX_PARTICLES.
    this.world.free();
    this.bodies = [];
    this.colliders = [];
    this.world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    this.addAttractor(this.params.attractorRadius);
    this.spawnBodies(n);
    this.disposeInstancedMesh();
    this.createInstancedMesh(n);
    this.numParticles = n;
    // Force the hot-param trackers to NaN so the next frame re-applies
    // their values to the freshly-created colliders + bodies.
    this.lastDamping = NaN;
    this.lastRestitution = NaN;
    this.lastAttractorRadius = NaN;
    this.lastTimescale = NaN;
  }

  update(): void {
    if (!this.world || !this.mesh) return;
    const noiseStrength = this.params.noiseStrength;
    const attractorStrength = this.params.attractorStrength;
    const attractorRadius = this.params.attractorRadius;
    const swirlStrength = this.params.swirlStrength;
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
    // restitution is hot — sweep all colliders (particles + walls +
    // attractor) only when the slider moved. Same rationale as damping.
    if (this.params.restitution !== this.lastRestitution) {
      const r = this.params.restitution;
      for (const c of this.colliders) c.setRestitution(r);
      if (this.attractorCollider) this.attractorCollider.setRestitution(r);
      this.lastRestitution = r;
    }
    // attractorRadius is hot — sweep the physical sphere collider AND the
    // visible mesh scale on change. Force-clamp distance uses the same
    // value so particles never get sucked inside the sphere.
    if (attractorRadius !== this.lastAttractorRadius) {
      if (this.attractorCollider) this.attractorCollider.setRadius(attractorRadius);
      if (this.attractorMesh) this.attractorMesh.scale.setScalar(attractorRadius);
      this.lastAttractorRadius = attractorRadius;
    }
    // timescale is hot — multiplier on rapier's per-step dt. Range capped
    // at 3x because the solver assumes small timesteps; bigger and contacts
    // start to tunnel. 0 = paused (rendering continues, physics stops).
    if (this.params.timescale !== this.lastTimescale) {
      this.world.timestep = (1 / 60) * this.params.timescale;
      this.lastTimescale = this.params.timescale;
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

      // Attractor as a velocity impulse (NOT addForce). Two reasons:
      // (1) addForce divides by rapier's auto-computed mass — a 4cm
      // sphere is ~3e-5 kg, so any force produces enormous acceleration
      // and the system explodes at default slider values.
      // (2) An inverse-cube law (the textbook Newtonian F/m = GM/r²) is
      // numerically harsh: it goes from gentle at r=1 to brutal at
      // r=0.1. We use a 1/r falloff instead — acceleration drops with
      // distance but never by the same orders of magnitude. Clamped at
      // attractorRadius (the physical sphere surface) so particles never
      // see infinite pull from the singularity at r=0.
      let ax = 0, ay = 0, az = 0;
      if (attractorStrength > 0) {
        const dx = ATTRACTOR_POSITION.x - t.x;
        const dy = ATTRACTOR_POSITION.y - t.y;
        const dz = ATTRACTOR_POSITION.z - t.z;
        const distSq = dx * dx + dy * dy + dz * dz;
        const dist = Math.sqrt(distSq);
        const clamped = Math.max(dist, attractorRadius);
        // accelMag ∝ 1/r (linear falloff in 1/distance, not 1/distance²).
        // Direction = unit vector toward attractor = (dx, dy, dz) / dist.
        // So per-axis: dx/dist * (strength / clamped) = dx * strength / (dist * clamped).
        const k = attractorStrength / (dist * clamped);
        ax = dx * k;
        ay = dy * k;
        az = dz * k;
      }

      // Swirl: rigid-rotation field around the Y axis through the attractor.
      // cross((0, 1, 0), (p - attractor)) = (pz - Az, 0, -(px - Ax)).
      // Magnitude scales with distance from the axis, so every particle has
      // the same angular velocity — clean orbital look.
      const swDx = t.x - ATTRACTOR_POSITION.x;
      const swDz = t.z - ATTRACTOR_POSITION.z;
      const sx = swDz * swirlStrength;
      const sz = -swDx * swirlStrength;

      body.setLinvel(
        {
          x: v.x + this.curlOut[0] * noiseStrength * dt + ax * dt + sx * dt,
          y: v.y + this.curlOut[1] * noiseStrength * dt + ay * dt,
          z: v.z + this.curlOut[2] * noiseStrength * dt + az * dt + sz * dt,
        },
        true,
      );

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
    this.storeUnsub?.();
    this.storeUnsub = null;
    this.disposeInstancedMesh();
    if (this.attractorMesh) {
      this.scene.remove(this.attractorMesh);
      this.attractorMesh.geometry.dispose();
      (this.attractorMesh.material as MeshBasicNodeMaterial).dispose();
      this.attractorMesh = null;
    }
    if (this.world) {
      // Frees all bodies + colliders too (including the attractor).
      this.world.free();
      this.world = null;
    }
    this.bodies = [];
    this.colliders = [];
    this.attractorCollider = null;
  }
}
