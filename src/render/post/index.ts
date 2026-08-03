import type { PostEffect } from "./PostEffect";
import { AoEffect } from "./effects/AoEffect";
import { DofEffect } from "./effects/DofEffect";
import { BloomEffect } from "./effects/BloomEffect";
import { TonemapEffect } from "./effects/TonemapEffect";
import { LensEffect } from "./effects/LensEffect";
import { GrainEffect } from "./effects/GrainEffect";

// Canonical order. Reorder only by editing this list. DOF and bloom run in
// linear HDR (the scene pass renders to an RGBA16F target): DOF sits before
// bloom so defocused highlights keep their energy and still bloom, like a
// real lens. Tonemap maps HDR to display range in-chain, so lens artifacts
// (distortion, CA, vignette) apply to the tonemapped LDR image like a
// film-camera path. Grain is absolutely last — it's the film stock on top of
// everything, including any lens artifacts.
export function buildPostEffects(): PostEffect[] {
  return [
    new AoEffect(),
    new DofEffect(),
    new BloomEffect(),
    new TonemapEffect(),
    new LensEffect(),
    new GrainEffect(),
  ];
}
