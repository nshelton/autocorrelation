// Hydrogen radial functions for principal quantum number n ∈ {1..4}, without
// the physics normalization constant (which doesn't affect visualization).
//
//   R_n(r) = L_{n-1}(2r/n) · exp(-r/n)
//
// where L_k is the simple Laguerre polynomial of degree k:
//   L_0(x) = 1
//   L_1(x) = 1 - x
//   L_2(x) = 1 - 2x + x²/2
//   L_3(x) = 1 - 3x + 3x²/2 - x³/6
//
// Higher n → more radial nodes (zero-density shells). Sign flips at each node.
// Visualization picks up the radial sign via the SH-side sign(ψ) read-out.
//
// Unknown n falls through to R_1; the shader does the same to keep behaviour
// in lockstep.

export function evalRadialJs(r: number, n: number): number {
  switch (n) {
    case 1: {
      // L_0(x) = 1; R_1 = exp(-r)
      return Math.exp(-r);
    }
    case 2: {
      const x = r;            // 2r/n = 2r/2 = r
      const lag = 1 - x;
      return lag * Math.exp(-r / 2);
    }
    case 3: {
      const x = (2 * r) / 3;
      const lag = 1 - 2 * x + (x * x) / 2;
      return lag * Math.exp(-r / 3);
    }
    case 4: {
      const x = r / 2;        // 2r/n = r/2
      const lag = 1 - 3 * x + (3 * x * x) / 2 - (x * x * x) / 6;
      return lag * Math.exp(-r / 4);
    }
    default: {
      // Unknown n → R_1 shape.
      return Math.exp(-r);
    }
  }
}

import { Fn, If, float } from "three/tsl";

// TSL mirror. n is a uniform float (we cast via toInt() inside the Fn).
// r is a TSL float node.
//
// We branch on the integer value of n using If chains — TSL doesn't have a
// native switch statement, but the chain compiles to a flat select tree in
// WGSL. The fall-through case is R_1 to match evalRadialJs.
export const evalRadialTsl = /*@__PURE__*/ Fn(([r, n]: [any, any]) => {
  const out = float(0).toVar();
  const ni = n.toInt();

  If(ni.equal(1), () => {
    out.assign(r.negate().exp());
  })
    .ElseIf(ni.equal(2), () => {
      const x = r;
      const lag = float(1).sub(x);
      out.assign(lag.mul(r.div(2).negate().exp()));
    })
    .ElseIf(ni.equal(3), () => {
      const x = r.mul(2 / 3);
      const lag = float(1).sub(x.mul(2)).add(x.mul(x).div(2));
      out.assign(lag.mul(r.div(3).negate().exp()));
    })
    .ElseIf(ni.equal(4), () => {
      const x = r.div(2);
      const lag = float(1)
        .sub(x.mul(3))
        .add(x.mul(x).mul(3).div(2))
        .sub(x.mul(x).mul(x).div(6));
      out.assign(lag.mul(r.div(4).negate().exp()));
    })
    .Else(() => {
      out.assign(r.negate().exp());
    });

  return out;
});
