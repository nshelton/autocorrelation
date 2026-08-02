import type { ParamSchema } from "./ParamStore";

// Global physics controls. Every physics component shares one rapier world
// (see render/components/physics.ts), so timestep and gravity can only be set
// in one place — here — rather than per scene.
export const physicsSchemas: ParamSchema[] = [
  // Multiplies the fixed 1/60 step. 0 pauses the sim; rendering continues.
  { key: "physics.timescale", label: "timescale", kind: "continuous", min: 0, max: 3, step: 0.01, default: 1, reconfig: false },
  // World gravity, m/s² down. Default 0 — scenes that want a field apply their
  // own per-body (Spawner's force field, ParticleView's attractor). Turn this
  // up when you want objects to settle and stack.
  { key: "physics.gravity", label: "gravity", kind: "continuous", min: 0, max: 30, step: 0.1, default: 0, reconfig: false },
];
