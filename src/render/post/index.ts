import type { WebGPURenderer } from "three/webgpu";
import type { PostEffect } from "./PostEffect";
import { AoEffect } from "./effects/AoEffect";
import { BloomEffect } from "./effects/BloomEffect";
import { TonemapEffect } from "./effects/TonemapEffect";
import { LensEffect } from "./effects/LensEffect";

// Canonical order. Reorder only by editing this list. Lens sits after tonemap
// so optical artifacts (distortion, CA, vignette) apply to the final LDR image
// like a film-camera path.
export function buildPostEffects(renderer: WebGPURenderer): PostEffect[] {
  return [
    new AoEffect(),
    new BloomEffect(),
    new TonemapEffect(renderer),
    new LensEffect(),
  ];
}
