import type { WebGPURenderer } from "three/webgpu";
import type { PostEffect } from "./PostEffect";
import { AoEffect } from "./effects/AoEffect";
import { TonemapEffect } from "./effects/TonemapEffect";

// Canonical order. Reorder only by editing this list.
export function buildPostEffects(renderer: WebGPURenderer): PostEffect[] {
  return [
    new AoEffect(),
    new TonemapEffect(renderer),
  ];
}
