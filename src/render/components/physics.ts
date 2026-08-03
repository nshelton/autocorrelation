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
  if (!world) {
    world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    // Half of rapier's default 4. Dense clusters spend most of the step in
    // the contact solver; piles get slightly softer in exchange for ~40% off
    // that cost — the right trade for a visualizer.
    world.numSolverIterations = 2;
  }
  return world;
}

// No-op until some component has asked for the world.
export function stepPhysics(timescale: number, gravityY: number): void {
  if (!world) return;
  world.timestep = BASE_TIMESTEP * timescale;
  if (world.gravity.y !== gravityY) world.gravity = { x: 0, y: gravityY, z: 0 };
  world.step();
}

// `active` = enabled AND awake — what the solver actually chews on. bodies.len()
// counts the pre-created disabled pool slots too, which is why total counts sit
// flat while the real workload ramps. The sweep is a per-body wasm call over
// every slot, so it's refreshed every 15 frames, not every frame.
let activeCache = 0;
let activeTtl = 0;

export function physicsStats(): { bodies: number; colliders: number; active: number } {
  if (!world) return { bodies: 0, colliders: 0, active: 0 };
  if (--activeTtl <= 0) {
    activeTtl = 15;
    let n = 0;
    world.bodies.forEach((b) => {
      if (b.isEnabled() && !b.isSleeping()) n++;
    });
    activeCache = n;
  }
  return { bodies: world.bodies.len(), colliders: world.colliders.len(), active: activeCache };
}
