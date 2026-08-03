import { describe, expect, it } from "vitest";
import RAPIER from "@dimforge/rapier3d-simd-compat";
import { getPhysicsWorld, stepPhysics } from "../../src/render/components/physics";

describe("shared physics world", () => {
  it("hands every caller the same world, even concurrently", async () => {
    const [a, b] = await Promise.all([getPhysicsWorld(), getPhysicsWorld()]);
    expect(a).toBe(b);
    expect(await getPhysicsWorld()).toBe(a);
  });

  it("stepPhysics owns timestep and gravity", async () => {
    const world = await getPhysicsWorld();
    stepPhysics(0.5, -9.8);
    expect(world.timestep).toBeCloseTo(1 / 120);
    expect(world.gravity.y).toBeCloseTo(-9.8);
    stepPhysics(0, 0);
    expect(world.timestep).toBe(0);
    expect(world.gravity.y).toBe(0);
  });

  it("one component's teardown leaves another's bodies usable", async () => {
    // The dispose contract: remove your own bodies, never free the world.
    // world.free() here is what used to panic every other component's handles.
    const world = await getPhysicsWorld();
    const mine = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic());
    // A collider gives the body mass; without one rapier's effective inverse
    // mass is 0 and gravity moves it nowhere.
    world.createCollider(RAPIER.ColliderDesc.ball(0.1), mine);
    const theirs = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic());
    world.createCollider(RAPIER.ColliderDesc.ball(0.1), theirs);

    world.removeRigidBody(theirs);
    stepPhysics(1, -9.8);

    expect(() => mine.setLinearDamping(0.2)).not.toThrow();
    expect(mine.translation().y).toBeLessThan(0); // gravity reached it
    world.removeRigidBody(mine);
  });

  it("bodies from different components collide", async () => {
    const world = await getPhysicsWorld();
    // A fixed floor from "component A" and a falling ball from "component B".
    const floor = world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, 0, 0));
    world.createCollider(RAPIER.ColliderDesc.cuboid(5, 0.1, 5), floor);
    const ball = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setTranslation(0, 2, 0),
    );
    world.createCollider(RAPIER.ColliderDesc.ball(0.2), ball);

    for (let i = 0; i < 240; i++) stepPhysics(1, -9.8);

    // Came to rest on top of the floor rather than falling through it.
    expect(ball.translation().y).toBeGreaterThan(0.2);
    expect(ball.translation().y).toBeLessThan(0.5);
    world.removeRigidBody(ball);
    world.removeRigidBody(floor);
  });
});
