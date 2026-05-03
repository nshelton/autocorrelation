import {
  Scene,
  InstancedMesh,
  InstancedBufferAttribute,
  BoxGeometry,
  Object3D,
  Color,
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
import type { FeatureStore } from "../store/FeatureStore";
import type { ParamStore } from "../params/ParamStore";

export interface BoxViewDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
}

const BOX_COUNT = 1024;
const CONTAINER_HALF = 1.5;
const BASE_SIZE = 0.12;
// Rest line: boxes spaced along X, centered at origin.
const REST_SPACING = 0.0025;
// Per-frame velocity nudge toward rest position.
const PULL = 0.5;

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

    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    world.timestep = 1 / 60;

    const c = CONTAINER_HALF;

    // Per-instance HSL colors. Owned by us (not via setColorAt) so the auto-instance-color
    // path in NodeMaterial doesn't kick in and clobber our custom colorNode.
    const colorArr = new Float32Array(BOX_COUNT * 3);
    const tmpColor = new Color();
    for (let i = 0; i < BOX_COUNT; i++) {
      tmpColor.setHSL(i / BOX_COUNT, 0.7, 0.6);
      tmpColor.toArray(colorArr, i * 3);
    }
    const colorAttr = new InstancedBufferAttribute(colorArr, 3);

    const mat = new MeshBasicNodeMaterial();
    const instColor = vec3(instancedBufferAttribute(colorAttr, "vec3", 3, 0));
    // Hardcoded normalized world-space light direction (≈ from upper-right-front).
    const lightDir = vec3(0.408, 0.866, 0.306);
    const ndotl = max(dot(normalWorld, lightDir), float(0.0));
    // 0.3 ambient + 0.7 lambert. Multiplied by per-instance albedo.
    const lit = ndotl.mul(0.7).add(0.3);
    mat.colorNode = vec4(instColor.mul(lit), 1.0);

    const geom = new BoxGeometry(BASE_SIZE, BASE_SIZE, BASE_SIZE);
    const mesh = new InstancedMesh(geom, mat, BOX_COUNT);

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
          .setLinearDamping(0.5)
          .setAngularDamping(0.5),
      );
      const collider = world.createCollider(
        RAPIER.ColliderDesc.cuboid(half, half, half).setRestitution(0.9),
        body,
      );
      this.bodies.push(body);
      this.colliders.push(collider);
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
        s = 0.1 + spec[bin] * 3.0;
      }

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
      (this.mesh.material as MeshBasicNodeMaterial).dispose();
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
