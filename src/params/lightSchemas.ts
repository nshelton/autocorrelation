import type { ParamSchema } from "./ParamStore";

export const lightSchemas: ParamSchema[] = [
  // 0.7 matches the retired hand-rolled lambert's `ndotl*0.7` term; the top
  // end is deliberately way past that for blown-out looks through the bloom.
  { key: "light.directional.intensity", label: "brightness", kind: "continuous", default: 0.7, min: 0, max: 5, step: 0.01, reconfig: false },
];
