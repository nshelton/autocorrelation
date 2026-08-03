import type { ParamSchema } from "./ParamStore";

// Integer-coded; index → preset name. Must stay in sync with
// CAMERA_PRESET_NAMES in App.ts (which feeds these to rig.goTo).
export const cameraSchemas: ParamSchema[] = [
  { key: "camera.rotate",    label: "Rotate",    kind: "continuous", min: 0, max: 3, step: 0.1, default: 0, reconfig: false },
  { key: "camera.fov",    label: "FOV",    kind: "continuous", min: 20, max: 120, step: 1, default: 60, reconfig: false },
  { key: "camera.distance", label: "Distance", kind: "continuous", min: 0.5, max: 10, step: 0.05, default: 4, reconfig: false },
  { key: "camera.preset", label: "Preset", kind: "discrete",
    options: [0, 1, 2, 3, 4, 5],
    optionLabels: ["front", "side", "spectrum", "rms", "buffer-acf", "rms-acf"],
    default: 0, reconfig: false },
  // Additive procedural "swim" — Brownian wander layered on top of the base
  // pose. Roughness = how fast the noise changes; amplitude = how far it moves.
  { key: "camera.swim.enabled",      label: "Swim",          kind: "boolean",    default: false, reconfig: false },
  { key: "camera.swim.posRoughness", label: "Pos roughness", kind: "continuous", min: 0, max: 2,  step: 0.01, default: 0.3, reconfig: false },
  { key: "camera.swim.posAmplitude", label: "Pos amplitude", kind: "continuous", min: 0, max: 2,  step: 0.01, default: 0.2, reconfig: false },
  { key: "camera.swim.rotRoughness", label: "Rot roughness", kind: "continuous", min: 0, max: 2,  step: 0.01, default: 0.3, reconfig: false },
  { key: "camera.swim.rotAmplitude", label: "Rot amplitude", kind: "continuous", min: 0, max: 10, step: 0.1,  default: 1.0, reconfig: false },
];
