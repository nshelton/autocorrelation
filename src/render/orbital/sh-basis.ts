// Real spherical harmonics for l = 0..3 (16 terms total).
//
// Coefficient layout (16 entries, both for JS and TSL paths):
//   0:  (l=0, m= 0)  — s
//   1:  (l=1, m=-1)  — p_y
//   2:  (l=1, m= 0)  — p_z
//   3:  (l=1, m= 1)  — p_x
//   4:  (l=2, m=-2)  — d_xy
//   5:  (l=2, m=-1)  — d_yz
//   6:  (l=2, m= 0)  — d_z²
//   7:  (l=2, m= 1)  — d_zx
//   8:  (l=2, m= 2)  — d_x²−y²
//   9:  (l=3, m=-3)
//  10:  (l=3, m=-2)
//  11:  (l=3, m=-1)
//  12:  (l=3, m= 0)
//  13:  (l=3, m= 1)
//  14:  (l=3, m= 2)
//  15:  (l=3, m= 3)
//
// Closed-form coefficients precomputed as constants. Polynomials are evaluated
// on the input (x, y, z) directly — caller normalizes if they want unit-vector
// inputs (we don't, to save a sqrt when caller already has it).

export const SH_COUNT = 16;

const N00 = 0.5 * Math.sqrt(1 / Math.PI);
const N1  = Math.sqrt(3 / (4 * Math.PI));
const N2A = 0.5 * Math.sqrt(15 / Math.PI);   // for m = ±2, ±1
const N20 = 0.25 * Math.sqrt(5 / Math.PI);    // for m = 0
const N22 = 0.25 * Math.sqrt(15 / Math.PI);
const N33 = 0.25 * Math.sqrt(35 / (2 * Math.PI));
const N32 = 0.25 * Math.sqrt(105 / Math.PI);  // for d at l=3 m=±2
const N31 = 0.25 * Math.sqrt(21 / (2 * Math.PI));
const N30 = 0.25 * Math.sqrt(7 / Math.PI);
const N3_M2 = 0.5 * Math.sqrt(105 / Math.PI); // for l=3 m=-2 (different by factor of 2)

export function evalShJs(coefs: Float32Array, x: number, y: number, z: number): number {
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;

  // l = 0
  const s =
    coefs[0]  * N00 +

  // l = 1
    coefs[1]  * N1  * y +
    coefs[2]  * N1  * z +
    coefs[3]  * N1  * x +

  // l = 2
    coefs[4]  * N2A * x * y +
    coefs[5]  * N2A * y * z +
    coefs[6]  * N20 * (3 * zz - 1) +
    coefs[7]  * N2A * x * z +
    coefs[8]  * N22 * (xx - yy) +

  // l = 3
    coefs[9]  * N33 * y * (3 * xx - yy) +
    coefs[10] * N3_M2 * x * y * z +
    coefs[11] * N31 * y * (5 * zz - 1) +
    coefs[12] * N30 * z * (5 * zz - 3) +
    coefs[13] * N31 * x * (5 * zz - 1) +
    coefs[14] * N32 * z * (xx - yy) +
    coefs[15] * N33 * x * (xx - 3 * yy);

  return s;
}

import { Fn, float } from "three/tsl";

// TSL mirror of evalShJs. Takes a storage-backed coefficient array (indexable
// by `coefs.element(i)`) and three vec3 components, returns a TSL float node.
//
// Called from the compute kernel; see OrbitalCloud.ts. The closed-form
// constants are inlined as floats — TSL doesn't have a constant-folder we
// can rely on but the WGSL compiler does.
//
// IMPORTANT: coefs must be a uniform array (NOT instancedArray), since every
// particle reads the same 16 coefficients. Pass via `uniformArray(16, "float")`
// and update on each frame from the params bag.
export const evalShTsl = /*@__PURE__*/ Fn(
  ([coefs, x, y, z]: [any, any, any, any]) => {
    const xx = x.mul(x);
    const yy = y.mul(y);
    const zz = z.mul(z);

    // Constants — re-declared inside the Fn so they fold into the shader.
    const n00 = float(N00);
    const n1  = float(N1);
    const n2a = float(N2A);
    const n20 = float(N20);
    const n22 = float(N22);
    const n33 = float(N33);
    const n32 = float(N32);
    const n31 = float(N31);
    const n30 = float(N30);
    const n3m2 = float(N3_M2);

    return coefs.element(0).mul(n00)
      .add(coefs.element(1).mul(n1).mul(y))
      .add(coefs.element(2).mul(n1).mul(z))
      .add(coefs.element(3).mul(n1).mul(x))
      .add(coefs.element(4).mul(n2a).mul(x).mul(y))
      .add(coefs.element(5).mul(n2a).mul(y).mul(z))
      .add(coefs.element(6).mul(n20).mul(zz.mul(3).sub(1)))
      .add(coefs.element(7).mul(n2a).mul(x).mul(z))
      .add(coefs.element(8).mul(n22).mul(xx.sub(yy)))
      .add(coefs.element(9).mul(n33).mul(y).mul(xx.mul(3).sub(yy)))
      .add(coefs.element(10).mul(n3m2).mul(x).mul(y).mul(z))
      .add(coefs.element(11).mul(n31).mul(y).mul(zz.mul(5).sub(1)))
      .add(coefs.element(12).mul(n30).mul(z).mul(zz.mul(5).sub(3)))
      .add(coefs.element(13).mul(n31).mul(x).mul(zz.mul(5).sub(1)))
      .add(coefs.element(14).mul(n32).mul(z).mul(xx.sub(yy)))
      .add(coefs.element(15).mul(n33).mul(x).mul(xx.sub(yy.mul(3))));
  },
);
