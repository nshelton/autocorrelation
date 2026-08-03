import type { ParamSchema } from "./ParamStore";

export const lightSchemas: ParamSchema[] = [
  // 0.7 matches the retired hand-rolled lambert's `ndotl*0.7` term; the top
  // end is deliberately way past that for blown-out looks through the bloom.
  { key: "light.directional.intensity", label: "brightness", kind: "continuous", default: 0.7, min: 0, max: 5, step: 0.01, reconfig: false },
  { key: "light.directional.color",     label: "light color", kind: "color", default: 0xffffff, reconfig: false },
  // Direction as azimuth/elevation (degrees) on a fixed radius-10 orbit, so
  // the shadow camera's [1, 25] depth bracket holds from any angle. Defaults
  // match the retired hardcoded position (4.08, 8.66, 3.06).
  { key: "light.directional.azimuth",   label: "azimuth",   kind: "continuous", default: 37, min: -180, max: 180, step: 1, reconfig: false },
  { key: "light.directional.elevation", label: "elevation", kind: "continuous", default: 60, min: -90, max: 90, step: 1, reconfig: false },
  { key: "light.ambient.intensity",     label: "ambient",   kind: "continuous", default: 0.3, min: 0, max: 2, step: 0.01, reconfig: false },
  { key: "light.ambient.color",         label: "ambient color", kind: "color", default: 0xffffff, reconfig: false },
];
