import type { ParamSchema } from "./ParamStore";

// System-level controls. Registered in main.ts before any UI is built so
// PresetStore can read the tween length on the very first apply().
export const systemSchemas: ParamSchema[] = [
  // Seconds to glide into a preset (module or system). 0 = snap immediately.
  { key: "system.presetTweenSecs", label: "preset tween", kind: "continuous", min: 0, max: 5, step: 0.05, default: 0.5, reconfig: false },
];
