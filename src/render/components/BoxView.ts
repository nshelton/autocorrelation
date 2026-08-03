import {
  InstancedMesh,
  BoxGeometry,
  IcosahedronGeometry,
  BufferGeometry,
  Object3D,
  Color,
} from "three";
import { MeshLambertNodeMaterial } from "three/webgpu";
import { vec4, uniform } from "three/tsl";
import RAPIER from "@dimforge/rapier3d-simd-compat";
import { getPhysicsWorld } from "./physics";
import type { Component, ComponentDeps } from "./Component";

const CONTAINER_HALF = 1.5;
// Geometry is built once at this size; the per-instance scale is
// (wanted world size / BASE_SIZE), so minSize/maxSize read as world units.
const BASE_SIZE = 0.12;

// primitive param values.
const PRIM_BOX = 0;
const PRIM_SPHERE = 1;

export class BoxView implements Component {
  static id = "boxView";
  static label = "Box View";
  static paramPrefix = "boxView";
  static paramOpts = {
    pull: { min: 0, max: 1, step: 0.01 },
    width: { min: 0, max: 2, step: 0.01 },
    minSize: { min: 0.005, max: 0.5, step: 0.005 },
    maxSize: { min: 0.005, max: 1.0, step: 0.005 },
  };
  static paramDefaults = {
    pull: 0.3,
    width: 0.5,
    primitive: PRIM_BOX,
    count: 1024,
    // Spectrum 0..1 maps into [minSize, maxSize] in world units.
    minSize: 0.02,
    maxSize: 0.36,
    color: 0xffffff,
  };
  static paramKinds = {
    primitive: "discrete" as const,
    count: "discrete" as const,
    color: "color" as const,
  };
  static paramDiscreteOptions = {
    primitive: [PRIM_BOX, PRIM_SPHERE],
    count: [128, 256, 512, 1024, 2048, 4096],
  };
  static paramDiscreteLabels = {
    primitive: ["Box", "Sphere"],
  };

  // Reference to App-owned stable bag — read each frame, mutated by tweakpane.
  // Never reassigned; tweakpane bindings depend on the object identity.
  private params: Record<string, number>;
  private scene: ComponentDeps["scene"];
  private store: ComponentDeps["store"];
  private mesh: InstancedMesh | null = null;
  private world: RAPIER.World | null = null;
  private bodies: RAPIER.RigidBody[] = [];
  private colliders: RAPIER.Collider[] = [];
  private dummy = new Object3D();
  // Live values behind the current bodies/mesh. count and primitive can't be
  // applied in place — a change rebuilds the pool.
  private count = 0;
  private primitive = PRIM_BOX;
  // Referenced live by the material's uniform node; mutate in place, never
  // reassign, so rebuilt materials keep pointing at the same instance.
  private color = new Color();
  private lastColor = NaN;
  // Size last pushed into each collider — resizes are throttled to >3% deltas
  // because every resize dirties the broad-phase (see Spawner).
  private sizeApplied = new Float32Array(0);
  private disposed = false;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.scene = deps.scene;
    this.store = deps.store;
    this.params = params;
    void this.init();
  }

  private async init(): Promise<void> {
    const world = await getPhysicsWorld();
    if (this.disposed) return;
    this.world = world;
    this.rebuild();
  }

  // Drop the current bodies + mesh and build `count` fresh ones of the current
  // primitive. Only called when those two params change (and once at startup),
  // never per frame.
  private rebuild(): void {
    const world = this.world;
    if (!world) return;
    // Shared world: remove only our own bodies (colliders ride along).
    for (const b of this.bodies) world.removeRigidBody(b);
    this.bodies = [];
    this.colliders = [];
    this.disposeMesh();

    const n = Math.round(this.params.count);
    const prim = Math.round(this.params.primitive);
    this.count = n;
    this.primitive = prim;

    const c = CONTAINER_HALF;
    const half = BASE_SIZE / 2;
    for (let i = 0; i < n; i++) {
      const x = (Math.random() - 0.5) * 2 * c * 0.7;
      const y = (Math.random() - 0.5) * 2 * c * 0.7;
      const z = (Math.random() - 0.5) * 2 * c * 0.7;
      const body = world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic()
          .setTranslation(x, y, z)
          .setLinvel(
            (Math.random() - 0.5) * 1.5,
            (Math.random() - 0.5) * 1.5,
            (Math.random() - 0.5) * 1.5,
          )
          .setAngvel({
            x: (Math.random() - 0.5) * 2,
            y: (Math.random() - 0.5) * 2,
            z: (Math.random() - 0.5) * 2,
          })
          .setLinearDamping(0.1)
          .setAngularDamping(0.1),
      );
      const shape =
        prim === PRIM_SPHERE
          ? RAPIER.ColliderDesc.ball(half)
          : RAPIER.ColliderDesc.cuboid(half, half, half);
      this.bodies.push(body);
      this.colliders.push(world.createCollider(shape.setRestitution(0.9), body));
    }
    this.sizeApplied = new Float32Array(n).fill(BASE_SIZE);

    // Sphere radius = box half-extent, so the two primitives occupy the same
    // size envelope and minSize/maxSize mean the same thing for both.
    const geom: BufferGeometry =
      prim === PRIM_SPHERE
        ? new IcosahedronGeometry(half, 2)
        : new BoxGeometry(BASE_SIZE, BASE_SIZE, BASE_SIZE);

    // Real lit material. The old hand-rolled lambert predated the vite alias
    // that collapsed the dual three instances — scene lights resolve now, and
    // a lit material is what receives shadows.
    const mat = new MeshLambertNodeMaterial();
    mat.colorNode = vec4(uniform(this.color), 1.0);

    const mesh = new InstancedMesh(geom, mat, n);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.mesh = mesh;
    this.scene.add(mesh);
  }

  update(): void {
    if (!this.world || !this.mesh) return;
    if (
      Math.round(this.params.count) !== this.count ||
      Math.round(this.params.primitive) !== this.primitive
    ) {
      this.rebuild();
    }
    if (!this.mesh) return;

    if (this.params.color !== this.lastColor) {
      this.color.setHex(this.params.color);
      this.lastColor = this.params.color;
    }

    const PULL = this.params.pull;
    const spec = this.store.get("spectrum");
    const specLen = spec.length;

    const n = this.count;
    const minSize = this.params.minSize;
    // Guard an inverted range (maxSize dragged below minSize) so sizes never
    // go negative and collapse the colliders.
    const maxSize = Math.max(this.params.maxSize, minSize);
    const isSphere = this.primitive === PRIM_SPHERE;
    const halfCount = (n - 1) / 2;

    for (let i = 0; i < n; i++) {
      const b = this.bodies[i];
      const t = b.translation();
      const r = b.rotation();

      const restX = ((i - halfCount) * this.params.width) / n;
      // Skip settled bodies and write with wake=false — the wake flag resets
      // the sleep timer, so wake=true kept the whole pool in the solver
      // forever even when every box sat at rest.
      if (!b.isSleeping()) {
        const vel = b.linvel();
        b.setLinvel(
          {
            x: vel.x + (restX - t.x) * PULL,
            y: vel.y - t.y * PULL,
            z: vel.z - t.z * PULL,
          },
          false,
        );
      }

      // Silence (no spectrum yet) sits every instance at minSize.
      let size = minSize;
      if (specLen > 0) {
        const bin = Math.min(specLen - 1, Math.floor((i / n) * specLen * 0.25));
        size = minSize + (maxSize - minSize) * spec[bin];
      }

      // Collider follows the audio-driven size in >3% steps only; the mesh
      // below stays perfectly smooth, physics just quantizes imperceptibly.
      const applied = this.sizeApplied[i];
      if (Math.abs(size - applied) > applied * 0.03) {
        const h = size / 2;
        if (isSphere) this.colliders[i].setRadius(h);
        else this.colliders[i].setHalfExtents({ x: h, y: h, z: h });
        this.sizeApplied[i] = size;
        // A collider change doesn't wake a sleeping body on its own — audio
        // kicking back in must re-animate settled boxes.
        if (b.isSleeping()) b.wakeUp();
      }

      const s = size / BASE_SIZE;
      this.dummy.position.set(t.x, t.y, t.z);
      this.dummy.quaternion.set(r.x, r.y, r.z, r.w);
      this.dummy.scale.set(s, s, s);
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  private disposeMesh(): void {
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    (this.mesh.material as MeshLambertNodeMaterial).dispose();
    this.mesh.dispose();
    this.mesh = null;
  }

  dispose(): void {
    this.disposed = true;
    this.disposeMesh();
    // Shared world: drop only our own bodies (their colliders go with them).
    // Clearing the arrays in the same breath matters — a setter on a removed
    // body panics rapier with "unreachable".
    if (this.world) {
      for (const b of this.bodies) this.world.removeRigidBody(b);
      this.world = null;
    }
    this.bodies = [];
    this.colliders = [];
  }
}
