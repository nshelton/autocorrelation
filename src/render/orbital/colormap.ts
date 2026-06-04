import { Fn, vec3 } from "three/tsl";

// Matplotlib coolwarm endpoints. Diverging colormap: COOL at large negative
// values, MID at zero, WARM at large positive. Shared by every orbital
// renderer so the same orbital reads as the same colors across views.
export const COOL = /*@__PURE__*/ vec3(0.230, 0.299, 0.754);
export const MID  = /*@__PURE__*/ vec3(0.865, 0.865, 0.865);
export const WARM = /*@__PURE__*/ vec3(0.706, 0.016, 0.150);

// Default light direction, normalized. Matches the lambert axis used by
// OrbitalCloud cubes/splats. Pre-normalized JS-side so TSL doesn't fold a
// normalize() at shader build time.
export const LIGHT_DIR = /*@__PURE__*/ (() => {
  const m = Math.hypot(0.408, 0.866, 0.306);
  return vec3(0.408 / m, 0.866 / m, 0.306 / m);
})();

// Hot-cold colormap with algebraic-sigmoid normalization.
//   t = (psi · colorScale) / (|psi · colorScale| + 1)  in (-1, 1)
//   warm side gets (t > 0); cool side gets (t < 0); MID at t == 0.
// Linear-combo form (no select/if) — TSL ConditionalNode had a known
// foot-gun with per-instance attributes; the linear split is equivalent.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const evalColormap = /*@__PURE__*/ Fn(([psi, colorScale]: [any, any]) => {
  const x = psi.mul(colorScale);
  const tNorm = x.div(x.abs().add(1));
  const tPos = tNorm.max(0);
  const tNeg = tNorm.min(0).abs();
  return MID
    .add(WARM.sub(MID).mul(tPos))
    .add(COOL.sub(MID).mul(tNeg));
});
