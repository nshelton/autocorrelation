import RAPIER from "@dimforge/rapier3d-compat";
import { initRapier } from "./rapier";

// ONE rapier world for every physics component. Objects from different scenes
// share it, so they collide and stack instead of ghosting through each other.
//
// The rules that keep that safe:
//  - Nobody frees the world. `world.free()` invalidates every other
//    component's handles — that's the use-after-free that panics rapier
//    ("null pointer passed to rust" / "unreachable"). Components remove their
//    OWN bodies on dispose (removeRigidBody drops the attached colliders too)
//    and drop the JS references in the same breath, because a setter on a
//    removed body panics.
//  - Nobody steps it. App calls stepPhysics() once per frame before
//    components update; three components each stepping would run the sim at
//    3x speed and the rate would change as scenes toggle.
//  - Nobody sets timestep or gravity. They are world-global, so they live on
//    the physics.* params instead of being fought over per component.
//
// Module singleton, so it survives App teardown/HMR like spawnQueue and
// shTween. A component that skips its removeRigidBody on dispose leaks bodies
// into every later session.
let world: RAPIER.World | null = null;

// Fixed base step. physics.timescale multiplies it; 0 pauses the simulation.
const BASE_TIMESTEP = 1 / 60;

export async function getPhysicsWorld(): Promise<RAPIER.World> {
  await initRapier();
  world ??= new RAPIER.World({ x: 0, y: 0, z: 0 });
  return world;
}

// No-op until some component has asked for the world.
export function stepPhysics(timescale: number, gravityY: number): void {
  if (!world) return;
  world.timestep = BASE_TIMESTEP * timescale;
  if (world.gravity.y !== gravityY) world.gravity = { x: 0, y: gravityY, z: 0 };
  world.step();
}
