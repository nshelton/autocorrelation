import type { PostEffect } from "./PostEffect";
import { AoEffect } from "./effects/AoEffect";

// Canonical order. Effects are reordered only by editing this list.
export const POST_EFFECTS: PostEffect[] = [
  new AoEffect(),
];
