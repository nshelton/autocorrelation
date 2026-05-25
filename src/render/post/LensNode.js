// LensNode: combined barrel distortion + radial chromatic aberration + vignette
// in a single resample pass. Modeled on three's RGBShiftNode addon — uses
// convertToTexture() to wrap the upstream node graph into a sampleable texture
// so we can sample it at warped UVs without managing our own RenderTarget.
//
// distortion > 0  → barrel  (corners pushed outward)
// distortion < 0  → pincushion
// chromatic = uv-fraction offset between R/G/B samples (radial; stronger at edges)
// vignetteStrength = corner darkening 0..1
// vignetteRadius = where darkening begins (radial distance, 0..1; 1 = none)

import { TempNode, nodeObject, Fn, uv, uniform, vec2, vec3, vec4, float, smoothstep, convertToTexture } from 'three/tsl';

class LensNode extends TempNode {
  static get type() { return 'LensNode'; }

  constructor(textureNode, distortion = 0, chromatic = 0, vignetteStrength = 0, vignetteRadius = 0.4) {
    super('vec4');
    this.textureNode = textureNode;
    this.distortion = uniform(distortion);
    this.chromatic = uniform(chromatic);
    this.vignetteStrength = uniform(vignetteStrength);
    this.vignetteRadius = uniform(vignetteRadius);
  }

  setup() {
    const { textureNode } = this;
    const sample = (u) => textureNode.uv(u);

    return Fn(() => {
      const center = vec2(0.5, 0.5);
      const offset = uv().sub(center);
      const r2 = offset.dot(offset);

      // Barrel/pincushion: scale offset by (1 + k*r^2).
      const distort = float(1.0).add(r2.mul(this.distortion));
      const warped = center.add(offset.mul(distort));

      // Radial chromatic aberration: R sampled outward, B inward, G centered.
      // Magnitude scales with offset (so center has zero CA, corners have most).
      const caOff = offset.mul(this.chromatic);
      const cr = sample(warped.add(caOff));
      const cg = sample(warped);
      const cb = sample(warped.sub(caOff));

      // Vignette: r² peaks at 0.5 in corners, so 2*r² ∈ [0,1] across screen.
      // smoothstep ramp from vignetteRadius to 1 gives the falloff curve.
      const radial = r2.mul(2.0);
      const v = float(1.0).sub(
        smoothstep(this.vignetteRadius, float(1.0), radial).mul(this.vignetteStrength),
      );

      return vec4(vec3(cr.r, cg.g, cb.b).mul(v), cg.a);
    })();
  }
}

export default LensNode;

export const lens = (node, distortion, chromatic, vignetteStrength, vignetteRadius) =>
  nodeObject(new LensNode(convertToTexture(node), distortion, chromatic, vignetteStrength, vignetteRadius));
