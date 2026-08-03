import {
  InstancedMesh,
  CapsuleGeometry,
  Object3D,
  Color,
  Vector3,
  Quaternion,
} from "three";
import type { MeshStandardNodeMaterial } from "three/webgpu";
import { makeLitMaterial, releaseLitMaterial } from "./litMaterial";
import { vec4, uniform } from "three/tsl";
import RAPIER from "@dimforge/rapier3d-compat";
import { getPhysicsWorld } from "./physics";
import type { Component, ComponentDeps } from "./Component";

// Serpent: a chain of N capsule segments over N+1 joints. The head steers
// toward a random target inside a sphere of radius `zone` and leaves a trail
// (ring buffer of past positions); each joint is pinned at a fixed arc length
// back along that trail. Every segment therefore travels the exact curve the
// head took — pull-toward-predecessor chains instead let the tail pivot in
// place when the head doubles back. Segment 0 is a kinematic rapier body (the
// scripted, immovable head); the rest are dynamic, chained by spherical
// joints and velocity-servoed toward their spine slot with strength fading
// head→tail (`flop`), so debris impacts swing the tail while the head stays
// on rails.
const UP = new Vector3(0, 1, 0);
// Re-pick the wander target on arrival, or after this long regardless — a
// slow head with a high turn rate can orbit a target forever otherwise.
const RETARGET_SECS = 5;
// Cap on the servo's corrective speed (world units/s): a segment knocked far
// off the spine swims back instead of snapping.
const CORR_MAX_SPEED = 8;
// Membership group 1, filter excludes group 1: segments never collide with
// each other (overlapping neighbors would jitter) but still hit everything.
const SERPENT_GROUPS = 0x0002fffd;

export class Serpent implements Component {
  static id = "serpent";
  static label = "Serpent";
  static paramPrefix = "serpent";
  static paramOpts = {
    segLen: { min: 0.02, max: 0.3, step: 0.005 },
    radius: { min: 0.01, max: 0.15, step: 0.005 },
    taper: { min: 0, max: 1, step: 0.01 },
    scale: { min: 1, max: 5, step: 0.05 },
    flop: { min: 0, max: 1, step: 0.01 },
    speed: { min: 0, max: 3, step: 0.01 },
    turn: { min: 0.5, max: 12, step: 0.1 },
    zone: { min: 0.2, max: 3, step: 0.05 },
  };
  static paramDefaults = {
    segments: 24,
    segLen: 0.08,
    radius: 0.04,
    taper: 0.6,
    scale: 1,
    flop: 0.5,
    speed: 0.8,
    turn: 4,
    zone: 1.0,
    color: 0x9ce65a,
  };
  static paramKinds = {
    segments: "discrete" as const,
    color: "color" as const,
  };
  static paramDiscreteOptions = {
    segments: [8, 16, 24, 32, 48, 64],
  };

  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private world: RAPIER.World | null = null;
  private mesh: InstancedMesh | null = null;
  private bodies: RAPIER.RigidBody[] = [];
  private colliders: RAPIER.Collider[] = [];
  private jointRefs: RAPIER.ImpulseJoint[] = [];
  private joints: Vector3[] = [];
  // Head path history, ring buffer, newest at trailHead. Sample spacing is
  // segLen/10 (or one frame's travel if larger), capacity covers 1.2x the
  // body length — always kept full, so joint placement never runs off the end.
  private trail: Vector3[] = [];
  private trailHead = 0;
  private vel = new Vector3();
  private target = new Vector3();
  private retarget = 0;
  private segments = 0;
  private lastSegLen = NaN;
  private lastRadius = NaN;
  private lastTaper = NaN;
  // Referenced live by the material's uniform node; mutate in place, never
  // reassign, so rebuilt materials keep pointing at the same instance.
  private color = new Color();
  private lastColor = NaN;
  private dummy = new Object3D();
  private dir = new Vector3();
  private mid = new Vector3();
  private quat = new Quaternion();
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
    this.rebuild();
  }

  // Drop bodies + mesh and build `segments` fresh ones. Only called when the
  // segment count changes (and once at startup). The head position survives a
  // rebuild; the body re-forms behind it within a few frames.
  private rebuild(): void {
    const world = this.world;
    if (!world) return;
    for (const b of this.bodies) world.removeRigidBody(b);
    this.bodies = [];
    this.colliders = [];
    this.disposeMesh();

    const n = Math.round(this.params.segments);
    this.segments = n;
    // `scale` multiplies the effective segment size everywhere (world-space
    // geometry, colliders, trail spacing), so the whole body grows as one.
    const segLen = this.params.segLen * this.params.scale;
    const radius = this.params.radius * this.params.scale;
    const taper = this.params.taper;
    this.lastSegLen = segLen;
    this.lastRadius = radius;
    this.lastTaper = taper;

    const head = this.joints[0] ?? new Vector3();
    this.joints = [head];
    let back = 0;
    for (let i = 1; i <= n; i++) {
      back += this.segmentLength(i - 1);
      this.joints.push(new Vector3(head.x - back, head.y, head.z));
    }
    // Seed the full ring as a straight line behind the head, using the same
    // index walk the placement loop uses, so the trail is valid immediately.
    const cap = n * 12 + 8;
    const minStep = segLen * 0.1;
    this.trail = [];
    this.trailHead = 0;
    for (let k = 0; k < cap; k++) this.trail.push(new Vector3());
    for (let k = 0; k < cap; k++) {
      this.trail[(this.trailHead - k + cap) % cap].set(head.x - k * minStep, head.y, head.z);
    }
    this.pickTarget();

    // Init line points head→tail along -X; local +Y maps to that direction.
    this.dir.set(-1, 0, 0);
    this.quat.setFromUnitVectors(UP, this.dir);
    for (let i = 0; i < n; i++) {
      this.mid.addVectors(this.joints[i], this.joints[i + 1]).multiplyScalar(0.5);
      // Segment 0 is the immovable scripted head. The rest are dynamic so the
      // solver can knock them around; gravity scale 0 means only collisions
      // move them, not physics.gravity. The spine servo in update() supplies
      // their motion otherwise.
      const desc = (
        i === 0
          ? RAPIER.RigidBodyDesc.kinematicPositionBased()
          : RAPIER.RigidBodyDesc.dynamic()
              .setGravityScale(0)
              .setLinearDamping(0.5)
              .setAngularDamping(4)
      )
        .setTranslation(this.mid.x, this.mid.y, this.mid.z)
        .setRotation(this.quat);
      const body = world.createRigidBody(desc);
      // Collider cylinder length = segLen, so the cap hemispheres overlap the
      // neighbor segments and the chain stays solid through bends.
      this.bodies.push(body);
      this.colliders.push(
        world.createCollider(
          RAPIER.ColliderDesc.capsule(this.segmentLength(i) / 2, this.segmentRadius(i))
            .setCollisionGroups(SERPENT_GROUPS),
          body,
        ),
      );
    }
    // Spherical joints pin each segment's tail end (+Y) to the next one's
    // head end (-Y): the chain bends freely under impact but never separates.
    // Anchors are per-segment — tapered segments are shorter toward the tail.
    // No explicit cleanup needed — removeRigidBody drops attached joints.
    this.jointRefs = [];
    for (let i = 1; i < n; i++) {
      this.jointRefs.push(
        world.createImpulseJoint(
          RAPIER.JointData.spherical(
            { x: 0, y: this.segmentLength(i - 1) / 2, z: 0 },
            { x: 0, y: -this.segmentLength(i) / 2, z: 0 },
          ),
          this.bodies[i - 1],
          this.bodies[i],
          true,
        ),
      );
    }

    // Real lit material (see BoxView.ts — the hand-rolled lambert era ended
    // with the vite three-alias); lit is also what receives shadows.
    const mat = makeLitMaterial();
    mat.colorNode = vec4(uniform(this.color), 1.0);

    const mesh = new InstancedMesh(new CapsuleGeometry(radius, segLen, 4, 10), mat, n);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // The snake roams; the origin-centered zero-size bounding sphere would
    // cull every instance.
    mesh.frustumCulled = false;
    this.mesh = mesh;
    this.scene.add(mesh);
  }

  // Head-to-tail falloff, applied to radius AND length together so tapered
  // segments stay true capsules — radius-only taper leaves long thin sticks at
  // the tail with pathological inertia. Floored so segments never collapse to
  // a degenerate size at taper 1.
  private segmentFactor(i: number): number {
    return Math.max(1 - this.params.taper * (i / (this.segments - 1)), 0.1);
  }

  private segmentRadius(i: number): number {
    return Math.max(this.params.radius * this.params.scale * this.segmentFactor(i), 0.004);
  }

  private segmentLength(i: number): number {
    return this.params.segLen * this.params.scale * this.segmentFactor(i);
  }

  private pickTarget(): void {
    const zone = this.params.zone;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    // cbrt → uniform in the ball, not clumped at the center.
    const r = zone * Math.cbrt(Math.random());
    const sinPhi = Math.sin(phi);
    this.target.set(
      r * sinPhi * Math.cos(theta),
      r * sinPhi * Math.sin(theta),
      r * Math.cos(phi),
    );
    this.retarget = RETARGET_SECS;
  }

  update(): void {
    if (!this.world || !this.mesh) return;
    if (Math.round(this.params.segments) !== this.segments) this.rebuild();
    if (!this.mesh) return;

    if (this.params.color !== this.lastColor) {
      this.color.setHex(this.params.color);
      this.lastColor = this.params.color;
    }

    const segLen = this.params.segLen * this.params.scale;
    const radius = this.params.radius * this.params.scale;
    const taper = this.params.taper;
    if (segLen !== this.lastSegLen || radius !== this.lastRadius || taper !== this.lastTaper) {
      if (segLen !== this.lastSegLen || radius !== this.lastRadius) {
        this.mesh.geometry.dispose();
        this.mesh.geometry = new CapsuleGeometry(radius, segLen, 4, 10);
      }
      for (let i = 0; i < this.colliders.length; i++) {
        this.colliders[i].setRadius(this.segmentRadius(i));
        this.colliders[i].setHalfHeight(this.segmentLength(i) / 2);
      }
      for (let k = 0; k < this.jointRefs.length; k++) {
        this.jointRefs[k].setAnchor1({ x: 0, y: this.segmentLength(k) / 2, z: 0 });
        this.jointRefs[k].setAnchor2({ x: 0, y: -this.segmentLength(k + 1) / 2, z: 0 });
      }
      this.lastSegLen = segLen;
      this.lastRadius = radius;
      this.lastTaper = taper;
    }

    // Physics dt, so the snake freezes with timescale 0 and slows with it,
    // like everything else in the shared world.
    const dt = this.world.timestep;
    const head = this.joints[0];

    this.retarget -= dt;
    if (this.retarget <= 0 || head.distanceTo(this.target) < this.params.zone * 0.15) {
      this.pickTarget();
    }
    this.dir.subVectors(this.target, head);
    const d = this.dir.length();
    if (d > 1e-6) this.dir.multiplyScalar(this.params.speed / d);
    // Exponential steering toward the desired velocity — `turn` is the 1/s
    // rate, so responsiveness is framerate-independent.
    this.vel.lerp(this.dir, 1 - Math.exp(-this.params.turn * dt));
    head.addScaledVector(this.vel, dt);

    // Extend the trail once the head is a sample-step away from the newest
    // point. Fast frames stride further than minStep — that's fine, the head
    // moves in a straight line within a frame so lerp reconstructs it exactly.
    const cap = this.trail.length;
    const minStep = segLen * 0.1;
    if (head.distanceToSquared(this.trail[this.trailHead]) >= minStep * minStep) {
      this.trailHead = (this.trailHead + 1) % cap;
      this.trail[this.trailHead].copy(head);
    }

    // Pin each joint at its cumulative segment-length arc distance back along
    // the trail (lerped between samples). One walk emits all joints in order;
    // tapered segments are shorter, so joints bunch toward the tail.
    let ji = 1;
    let targetD = this.segmentLength(0);
    let acc = 0;
    let prev: Vector3 = head;
    for (let k = 0; k < cap && ji < this.joints.length; k++) {
      const p = this.trail[(this.trailHead - k + cap) % cap];
      const d = prev.distanceTo(p);
      if (d > 1e-9) {
        while (ji < this.joints.length && targetD <= acc + d) {
          this.joints[ji].lerpVectors(prev, p, (targetD - acc) / d);
          targetD += this.segmentLength(ji);
          ji++;
        }
        acc += d;
      }
      prev = p;
    }
    // Trail shorter than the body (segLen just grew): park the leftovers at
    // the oldest point; they unfold as fresh trail accumulates.
    for (; ji < this.joints.length; ji++) this.joints[ji].copy(prev);

    const flop = this.params.flop;
    const fracInv = 1 / (this.segments - 1);
    for (let i = 0; i < this.segments; i++) {
      const a = this.joints[i];
      const b = this.joints[i + 1];
      this.mid.addVectors(a, b).multiplyScalar(0.5);
      const body = this.bodies[i];

      if (i === 0) {
        this.dir.subVectors(b, a).normalize();
        this.quat.setFromUnitVectors(UP, this.dir);
        // App steps the world before components update, so this lands on the
        // next step — the engine derives the body's velocity from the move,
        // which is what gives dynamic bodies a real shove, not a teleport.
        body.setNextKinematicTranslation({ x: this.mid.x, y: this.mid.y, z: this.mid.z });
        body.setNextKinematicRotation({
          x: this.quat.x,
          y: this.quat.y,
          z: this.quat.z,
          w: this.quat.w,
        });
      } else if (dt > 0) {
        // Velocity servo toward the spine slot, weakening head→tail: tight=1
        // fully overwrites velocity (rigid scripted follow), lower keeps more
        // of whatever the last impact did before swimming back. Orientation is
        // left to the joints + angular damping; taper also makes tail
        // segments lighter, so the same hit swings them further for free.
        const tight = 1 - flop * Math.pow(i * fracInv, 1.5);
        const t = body.translation();
        let cx = (this.mid.x - t.x) / dt;
        let cy = (this.mid.y - t.y) / dt;
        let cz = (this.mid.z - t.z) / dt;
        const cm = Math.hypot(cx, cy, cz);
        if (cm > CORR_MAX_SPEED) {
          const k = CORR_MAX_SPEED / cm;
          cx *= k;
          cy *= k;
          cz *= k;
        }
        const v = body.linvel();
        body.setLinvel(
          {
            x: v.x + (cx - v.x) * tight,
            y: v.y + (cy - v.y) * tight,
            z: v.z + (cz - v.z) * tight,
          },
          true,
        );
      }

      // Render from the bodies, not the spine — that's where the slap shows.
      const bt = body.translation();
      const br = body.rotation();
      this.dummy.position.set(bt.x, bt.y, bt.z);
      this.dummy.quaternion.set(br.x, br.y, br.z, br.w);
      // Taper is a uniform per-instance scale on the shared geometry — length
      // and radius shrink together, so the capsule shape stays true and
      // matches the collider exactly.
      this.dummy.scale.setScalar(this.segmentFactor(i));
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  private disposeMesh(): void {
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    releaseLitMaterial(this.mesh.material as MeshStandardNodeMaterial);
    this.mesh.dispose();
    this.mesh = null;
  }

  dispose(): void {
    this.disposed = true;
    this.disposeMesh();
    // Shared world: drop only our own bodies (colliders ride along) and clear
    // the arrays in the same breath — a setter on a removed body panics rapier.
    if (this.world) {
      for (const b of this.bodies) this.world.removeRigidBody(b);
      this.world = null;
    }
    this.bodies = [];
    this.colliders = [];
    this.jointRefs = [];
  }
}
