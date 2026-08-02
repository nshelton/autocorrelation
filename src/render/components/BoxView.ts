import {
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
import { getPhysicsWorld } from "./physics";
import type { Component, ComponentDeps } from "./Component";

const BOX_COUNT = 1024;
const CONTAINER_HALF = 1.5;
const BASE_SIZE = 0.12;

export class BoxView implements Component {
  static id = "boxView";
  static label = "Box View";
  static paramPrefix = "boxView";
  static paramOpts = {
    pull: { min: 0, max: 1, step: 0.01 },
    width: { min: 0, max: 2, step: 0.01 },
  };
  static paramDefaults = {
    pull: 0.3,
    width: 0.5,
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

    const c = CONTAINER_HALF;

    // Per-instance HSL colors via our own InstancedBufferAttribute. setColorAt would route
    // through NodeMaterial's vInstanceColor varying, which is broken for our setup in r170.
    const colorArr = new Float32Array(BOX_COUNT * 3);
    const tmpColor = new Color();
    for (let i = 0; i < BOX_COUNT; i++) {
      tmpColor.setHSL(1, 1, 1);
      tmpColor.toArray(colorArr, i * 3);
    }
    const colorAttr = new InstancedBufferAttribute(colorArr, 3);

    // MeshStandardNodeMaterial with custom colorNode silently drops lights in r170 + WebGPU +
    // InstancedMesh. Hand-rolled lambert on MeshBasicNodeMaterial is what works.
    const mat = new MeshBasicNodeMaterial();
    const instColor = vec3(instancedBufferAttribute(colorAttr, "vec3", 3, 0));
    const lightDir = vec3(0.408, 0.866, 0.306);
    const ndotl = max(dot(normalWorld, lightDir), float(0.0));
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
          .setLinearDamping(0.1)
          .setAngularDamping(0.1),
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
    const PULL = this.params.pull;

    const spec = this.store.get("spectrum");
    const specLen = spec.length;

    const baseHalf = BASE_SIZE / 2;
    const halfCount = (BOX_COUNT - 1) / 2;
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i];
      const t = b.translation();
      const r = b.rotation();

      const restX = ((i - halfCount) * this.params.width) / BOX_COUNT;
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
