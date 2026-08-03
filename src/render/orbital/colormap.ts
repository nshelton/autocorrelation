import { vec3 } from "three/tsl";

// Default light direction, normalized. Matches the lambert axis used by
// OrbitalCloud cubes/splats. Pre-normalized JS-side so TSL doesn't fold a
// normalize() at shader build time.
export const LIGHT_DIR = /*@__PURE__*/ (() => {
  const m = Math.hypot(0.408, 0.866, 0.306);
  return vec3(0.408 / m, 0.866 / m, 0.306 / m);
})();
