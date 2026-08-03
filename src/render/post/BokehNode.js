// BokehNode: gather bokeh blur with a polygonal aperture. Replaces three's
// DepthOfFieldNode, whose linear-distance defocus made any nonzero aperture
// slam the far field (viewZ −1000) straight to full blur. Defocus here is
// inverse-distance, so the far field converges instead of saturating and
// `aperture` reads as a plain physical control:
//
//   blur = aperture · clamp(1 − focus/dist, −1, 1)
//
// 0 = everything sharp; larger = narrower in-focus zone around `focus`, with
// the background blur radius approaching exactly `aperture`. The clamp caps
// the near field (dist < focus/2) at the same radius, kernel point-reflected —
// real lenses invert the aperture image in front of focus.
//
// Taps sit on three concentric bands warped into an n-gon by the regular-
// polygon edge-distance function, and every tap is jittered per pixel (one
// IGN hash + per-tap R2 offsets) within its angular cell and radial band —
// the sparse kernel dithers into stable noise instead of visible tap rings.
// Luminance-weighted accumulation lets bright HDR taps dominate so
// out-of-focus highlights resolve into crisp aperture shapes (this pass runs
// pre-tonemap).
//
// Taps are also depth-tested against their own blur radius (see `tap` below),
// which is what keeps a sharp object from being scattered across the aperture
// disc of the blurred background behind it.
//
// The 49-tap gather runs in its own half-resolution pass — a quarter of the
// fragments. `bokeh()` then mixes that upsampled result back over the sharp
// source by blur radius, so in-focus pixels never pay the upsample: they read
// the full-res source, exactly as they did when the gather ran inline.
//
// focus    = view distance of the focal plane (world units)
// aperture = far-field blur radius in UV units of screen WIDTH; doubles as
//            "how shallow" — 0 disables
// blades   = polygon side count (pass a large n, e.g. 48, for a circle)
// rotation = blade orientation, radians
// boost    = highlight weighting strength (0 = plain average)

import { HalfFloatType, RenderTarget, Vector2 } from 'three';
import { PostProcessingUtils } from 'three/webgpu';
import { TempNode, nodeObject, Fn, NodeUpdateType, NodeMaterial, QuadMesh, uv, uniform, vec2, vec3, vec4, float, clamp, cos, sin, mod, min, mix, smoothstep, luminance, PI, fract, dot, screenCoordinate, textureSize, passTexture, convertToTexture, perspectiveDepthToViewZ, cameraNear, cameraFar } from 'three/tsl';

const _quadMesh = /*@__PURE__*/ new QuadMesh();
const _size = /*@__PURE__*/ new Vector2();

let _rendererState;

// Signed blur radius, UV units of screen width. Negative in front of the focal
// plane. Shared by the gather (kernel scale) and the composite (mix weight) so
// the two can't drift apart.
const defocus = (viewZ, focus, aperture) =>
  clamp(float(1.0).sub(focus.div(viewZ.negate())), -1.0, 1.0).mul(aperture);

// Tap layout: [nominal angle, angular cell width, band lo, band hi] per tap.
// Counts scale with band circumference so density stays even; half-step
// stagger breaks up radial lines. Jitter fills each cell × band per pixel.
const TAPS = [];
for (const [lo, hi, count] of [[0, 1 / 3, 8], [1 / 3, 2 / 3, 16], [2 / 3, 1, 24]]) {
  for (let i = 0; i < count; i++) {
    TAPS.push([((i + 0.5) * 2 * Math.PI) / count, (2 * Math.PI) / count, lo, hi]);
  }
}

class BokehNode extends TempNode {
  static get type() { return 'BokehNode'; }

  constructor(textureNode, viewZNode, depthNode, focusNode, apertureNode, bladesNode, rotationNode, boostNode) {
    super('vec4');
    this.textureNode = textureNode;
    this.viewZNode = viewZNode;
    // Raw depth texture as well: the per-tap defocus test below needs viewZ at
    // arbitrary tap UVs, which a prebaked viewZ node can't give us.
    this.depthNode = depthNode;
    this.focusNode = focusNode;
    this.apertureNode = apertureNode;
    this.bladesNode = bladesNode;
    this.rotationNode = rotationNode;
    this.boostNode = boostNode;

    this._aspect = uniform(0);
    this.resolutionScale = 0.5;

    // HalfFloat: this pass runs pre-tonemap, so the gather output is HDR and
    // must survive until bloom.
    this._rt = new RenderTarget(1, 1, { type: HalfFloatType, depthBuffer: false });
    this._rt.texture.name = 'BokehNode.blur';
    this._material = null;
    this._textureNode = passTexture(this, this._rt.texture);

    this.updateBeforeType = NodeUpdateType.FRAME;
  }

  getTextureNode() {
    return this._textureNode;
  }

  updateBefore({ renderer }) {
    const map = this.textureNode.value;
    this._aspect.value = map.image.width / map.image.height;

    _rendererState = PostProcessingUtils.resetRendererState(renderer, _rendererState);

    const size = renderer.getDrawingBufferSize(_size);
    this._rt.setSize(
      Math.max(1, Math.round(size.width * this.resolutionScale)),
      Math.max(1, Math.round(size.height * this.resolutionScale)),
    );

    _quadMesh.material = this._material;
    renderer.setRenderTarget(this._rt);
    _quadMesh.render(renderer);

    PostProcessingUtils.restoreRendererState(renderer, _rendererState);
  }

  setup(builder) {
    const textureNode = this.textureNode;
    const uvNode = textureNode.uvNode || uv();
    const sample = (u) => textureNode.uv(u);
    const viewZAt = (u) => perspectiveDepthToViewZ(this.depthNode.uv(u).x, cameraNear, cameraFar);
    const cocAt = (u) => defocus(viewZAt(u), this.focusNode, this.apertureNode).abs();

    const gather = Fn(() => {
      // Offsets are defined relative to screen width; scaling v by aspect
      // makes the kernel square in pixels.
      const aspectcorrect = vec2(1.0, this._aspect);

      // Inverse-distance defocus: 0 at the focal plane, → +1 far field,
      // −1 for the near field (sign point-reflects the kernel).
      const blur = defocus(this.viewZNode, this.focusNode, this.apertureNode).toVar();

      // Regular n-gon with circumradius 1: boundary distance in direction φ is
      // cos(π/n) / cos(φ mod 2π/n − π/n). 1 at vertices, inradius at edges.
      const halfSeg = PI.div(this.bladesNode);
      const inradius = cos(halfSeg);

      // One IGN hash per pixel (same formula as GrainEffect); per-tap R2
      // additive-recurrence offsets decorrelate the taps off that single hash.
      const n = fract(float(52.9829189).mul(fract(dot(screenCoordinate.xy, vec2(0.06711056, 0.00583715)))));

      const acc = vec3(0.0).toVar();
      const wsum = float(0.0).toVar();

      // Scatter-as-gather: a tap only reaches this pixel if its OWN blur radius
      // is at least as wide as its distance from us (`reach`, same UV-of-width
      // units as the CoC). The kernel radius comes from the center pixel alone,
      // so without this test a background pixel gathers at full far-field radius
      // and drags in whatever sharp geometry sits inside that disc — and the
      // luminance weighting then lets it win the sum outright. That is what made
      // a sharp emissive sphere against distant background smear out to the
      // background's blur radius instead of its own.
      //
      // Feather is one half-res pixel and sits entirely BELOW the threshold, so
      // a uniform-depth gather (rim taps have cocTap == reach) keeps full weight
      // and the disc doesn't lose its edge.
      const feather = float(2.0).div(float(textureSize(textureNode, 0).x));

      // Luminance clamped so a single extreme-HDR tap can't dominate the sum
      // and shimmer as geometry moves through the kernel.
      const tap = (offset, reach) => {
        const u = uvNode.add(offset);
        const c = sample(u);
        const lum = min(luminance(c.rgb), 4.0);
        const w = this.boostNode.mul(lum).mul(lum).add(1.0)
          .mul(smoothstep(reach.sub(feather), reach, cocAt(u)));
        acc.addAssign(c.rgb.mul(w));
        wsum.addAssign(w);
      };

      // Center tap: reach 0, so it always survives — wsum can never hit zero.
      tap(vec2(0.0), float(0.0));

      TAPS.forEach(([theta, cell, lo, hi], i) => {
        // Boundary distance uses the jittered pre-rotation angle — jittered
        // taps still land inside the aperture polygon, so the aggregate shape
        // stays crisp while the tap pattern dithers. The tap direction itself
        // carries the rotation, so the polygon spins with `rotation`.
        const thetaJ = fract(n.add(i * 0.7548776662)).sub(0.5).mul(cell).add(theta);
        const radJ = fract(n.add(i * 0.5698402910)).mul(hi - lo).add(lo);
        const polyR = inradius.div(cos(mod(thetaJ, halfSeg.mul(2.0)).sub(halfSeg)));
        const phi = this.rotationNode.add(thetaJ);
        const kernelR = polyR.mul(radJ);
        const offset = vec2(cos(phi), sin(phi)).mul(kernelR).mul(aspectcorrect).mul(blur);
        // |blur| because the near field carries a negative blur that only
        // point-reflects the offset; the distance travelled is the magnitude.
        tap(offset, kernelR.mul(blur.abs()));
      });

      return vec4(acc.div(wsum), 1.0);
    });

    const material = this._material || (this._material = new NodeMaterial());
    material.fragmentNode = gather().context(builder.getSharedContext());
    material.name = 'Bokeh';
    material.needsUpdate = true;

    return this._textureNode;
  }

  dispose() {
    this._rt.dispose();
  }
}

export default BokehNode;

export const bokeh = (node, viewZ, depth, focus, aperture, blades, rotation, boost) => {
  // One RTT for the source: the composite below reads it sharp and the gather
  // pass reads it blurred, so letting each convert its own input would render
  // the upstream chain twice.
  const src = convertToTexture(node);
  const viewZNode = nodeObject(viewZ);
  const focusNode = nodeObject(focus);
  const apertureNode = nodeObject(aperture);

  const blurred = nodeObject(new BokehNode(
    src, viewZNode, nodeObject(depth), focusNode, apertureNode,
    nodeObject(blades), nodeObject(rotation), nodeObject(boost),
  ));

  // Blur radius in full-res pixels decides who wins. Under half a pixel the
  // gather is a no-op that only cost us the half-res round trip, so take the
  // sharp source; past two pixels the blur has already erased the detail the
  // upsample would have lost. Keeps the focal plane pixel-exact.
  const srcWidth = float(textureSize(src, 0).x);
  const radiusPx = defocus(viewZNode, focusNode, apertureNode).abs().mul(srcWidth);
  return mix(src, blurred.getTextureNode(), smoothstep(0.5, 2.0, radiusPx));
};
