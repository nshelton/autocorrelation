import { Fn } from "three/tsl";
import { evalShTsl } from "./sh-basis";
import { evalRadialTsl } from "./radial";

// ψ(r) = R_n(|r|/radialScale) · Y_l_m(r̂, shCoefs).
// Direction-of-r̂ extraction guards against |r|=0 with a min 1e-6 (same
// epsilon used by the OrbitalCloud kernel so both renderers see identical
// values at the origin).
export const evalPsi = Fn(
  ([pos, shCoefs, n, radialScale]: [any, any, any, any]) => {
    const rLen = pos.length().max(1e-6);
    const rScaled = rLen.div(radialScale);
    const xh = pos.x.div(rLen);
    const yh = pos.y.div(rLen);
    const zh = pos.z.div(rLen);
    const R = evalRadialTsl(rScaled, n);
    const Y = evalShTsl(shCoefs, xh, yh, zh);
    return R.mul(Y);
  },
);
