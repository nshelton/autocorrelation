import {
  Scene,
  InstancedMesh,
  BoxGeometry,
  MeshBasicMaterial,
  Object3D,
  Color,
} from "three";
import RAPIER from "@dimforge/rapier3d-compat";
import type { FeatureStore } from "../store/FeatureStore";
import type { ParamStore } from "../params/ParamStore";

export interface BoxViewDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
}

const BOX_COUNT = 100;
const CONTAINER_HALF = 1.5;
const BASE_SIZE = 0.12;
// Rest line: boxes spaced along X, centered at origin. Spacing chosen so 100 boxes fit inside the container.
const REST_SPACING = 0.025;
// Per-frame velocity nudge toward rest position (units of 1/frame). Tune for snappier vs. softer pull.
const PULL = 0.01;

export class BoxView {
  private scene: Scene;
  private store: FeatureStore;
  private mesh: InstancedMesh | null = null;
  private world: RAPIER.World | null = null;
  private bodies: RAPIER.RigidBody[] = [];
  private colliders: RAPIER.Collider[] = [];
  private dummy = new Object3D();
  private disposed = false;

  constructor(deps: BoxViewDeps) {
    this.scene = deps.scene;
    this.store = deps.store;
    void this.init();
  }

  private async init(): Promise<void> {
    await RAPIER.init();
    if (this.disposed) return;

    // Zero gravity: boxes float and only collide with each other / walls.
    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });

    const c = CONTAINER_HALF;

    const geom = new BoxGeometry(BASE_SIZE, BASE_SIZE, BASE_SIZE);
    const mat = new MeshBasicMaterial({ color: 0xffffff });
    const mesh = new InstancedMesh(geom, mat, BOX_COUNT);
    const color = new Color();

    const half = BASE_SIZE / 2;
    for (let i = 0; i < BOX_COUNT; i++) {
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
          .setLinearDamping(0.01)
          .setAngularDamping(0.01),
      );
      const collider = world.createCollider(
        RAPIER.ColliderDesc.cuboid(half, half, half).setRestitution(0.9),
        body,
      );
      this.bodies.push(body);
      this.colliders.push(collider);

      color.setHSL(i / BOX_COUNT, 0.7, 0.6);
      mesh.setColorAt(i, color);
    }

    this.world = world;
    this.mesh = mesh;
    this.scene.add(mesh);
  }

  update(): void {
    if (!this.world || !this.mesh) return;
    this.world.step();

    const spec = this.store.get("spectrum");
    const specLen = spec.length;

    const baseHalf = BASE_SIZE / 2;
    const halfCount = (BOX_COUNT - 1) / 2;
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i];
      const t = b.translation();
      const r = b.rotation();

      // Soft pull toward rest position (i, 0, 0) along X line. Adds to velocity each frame;
      // existing linearDamping eventually settles it. Collisions still nudge boxes off-line.
      const restX = (i - halfCount) * REST_SPACING;
      const vel = b.linvel();
      b.setLinvel(
        {
          x: vel.x + (restX - t.x) * PULL,
          y: vel.y - t.y * PULL,
          z: vel.z - t.z * PULL,
        },
        true,
      );

      let s = 1.0;
      if (specLen > 0) {
        const bin = Math.min(
          specLen - 1,
          Math.floor((i / BOX_COUNT) * specLen * 0.25),
        );
        s = 0.4 + spec[bin] * 6.0;
      }

      // Sync collider half-extents to the visual scale so physics matches what's drawn.
      const h = baseHalf * s;
      this.colliders[i].setHalfExtents({ x: h, y: h, z: h });

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
      (this.mesh.material as MeshBasicMaterial).dispose();
      this.mesh.dispose();
      this.mesh = null;
    }
    if (this.world) {
      this.world.free();
      this.world = null;
    }
    this.bodies = [];
    this.colliders = [];
  }
}
