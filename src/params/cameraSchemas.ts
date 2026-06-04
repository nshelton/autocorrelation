import type { ParamSchema } from "./ParamStore";

// Integer-coded; index → preset name. Must stay in sync with
// CAMERA_PRESET_NAMES in App.ts (which feeds these to rig.goTo).
export const cameraSchemas: ParamSchema[] = [
  { key: "camera.rotate",    label: "Rotate",    kind: "continuous", min: 0, max: 10, step: 1, default: 0, reconfig: false },
  { key: "camera.fov",    label: "FOV",    kind: "continuous", min: 20, max: 120, step: 1, default: 60, reconfig: false },
  { key: "camera.preset", label: "Preset", kind: "discrete",
    options: [0, 1, 2, 3, 4, 5],
    optionLabels: ["front", "side", "spectrum", "rms", "buffer-acf", "rms-acf"],
    default: 0, reconfig: false },
];
