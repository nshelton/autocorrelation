import { Fn, uniform, vec2, vec4, float, hash, texture, frameId, screenCoordinate } from "three/tsl";
import type { ShaderNodeObject } from "three/tsl";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";
import type { Modulator } from "../../../params/Modulator";
import type { ParamProxyRegistry } from "../../../params/bindParam";
import { getBlueNoiseTexture, BLUE_NOISE_SIZE } from "../blueNoise";

// Film grain sampled from a procedural blue-noise mask (void-and-cluster,
// see blueNoise.ts) — no clumping, evenly distributed, no baked asset. The
// mask tiles (RepeatWrapping) and is shifted by a new pseudo-random offset
// every frame for the "film moving" temporal effect; a static offset would
// just look like a fixed dither pattern.
export class GrainEffect implements PostEffect {
  readonly id = "grain";
  readonly label = "Grain";
  readonly needs = {} as const;
  enabled = false;

  private strengthU = uniform(0.08);
  private scaleU = uniform(1.0);
  private unsub: (() => void) | null = null;

  registerParams(store: ParamStore): void {
    this.strengthU.value = store.get("post.grain.strength") as number;
    this.scaleU.value = store.get("post.grain.scale") as number;
    this.unsub = store.subscribe((key, value) => {
      if (typeof value !== "number") return;
      if (key === "post.grain.strength") this.strengthU.value = value;
      else if (key === "post.grain.scale") this.scaleU.value = value;
    });
  }

  build(input: ShaderNodeObject<Node>, _ctx: PassCtx): ShaderNodeObject<Node> {
    const mask = getBlueNoiseTexture();
    return Fn(() => {
      // New pseudo-random shift per frame; the mask wraps (RepeatWrapping)
      // so any shift lands on a valid texel.
      const shift = vec2(hash(float(frameId)), hash(float(frameId).add(1.0)))
        .mul(float(BLUE_NOISE_SIZE));
      const uv = screenCoordinate.xy.div(this.scaleU).add(shift).div(float(BLUE_NOISE_SIZE));
      // Multiplicative, not additive: Grain runs after tonemap, in clipped
      // [0,1] display space. Additive noise clips asymmetrically near black/
      // white (e.g. a near-black pixel's negative excursions clip to 0 but
      // its positive ones survive), biasing the image lighter overall on a
      // mostly-dark scene. Scaling by the pixel's own value keeps
      // E[input * (1 + g)] == input exactly, and black stays black.
      const gain = texture(mask, uv).r.sub(0.5).mul(this.strengthU).add(1.0);
      return (input as ShaderNodeObject<Node>).mul(vec4(gain, gain, gain, 1.0));
    })() as ShaderNodeObject<Node>;
  }

  bindUI(
    folder: FolderApi,
    store: ParamStore,
    _modulator: Modulator,
    proxies: ParamProxyRegistry,
  ): void {
    const b = {
      enabled: store.get("post.grain.enabled") as boolean,
      strength: store.get("post.grain.strength") as number,
      scale: store.get("post.grain.scale") as number,
    };
    folder
      .addBinding(b, "enabled", { label: "Enabled" })
      .on("change", (e: { value: boolean }) => store.set("post.grain.enabled", e.value));
    folder
      .addBinding(b, "strength", { label: "Strength", min: 0, max: 0.5, step: 0.005 })
      .on("change", (e: { value: number }) => store.set("post.grain.strength", e.value));
    folder
      .addBinding(b, "scale", { label: "Scale", min: 0.1, max: 4.0, step: 0.05 })
      .on("change", (e: { value: number }) => store.set("post.grain.scale", e.value));
    // The widgets bind `b`, which only ever moves on user input — re-pull it so
    // programmatic writes (preset load, reset) show up in the panel.
    proxies.set("post.grain.enabled", () => { b.enabled = store.get("post.grain.enabled") as boolean; });
    proxies.set("post.grain.strength", () => { b.strength = store.get("post.grain.strength") as number; });
    proxies.set("post.grain.scale", () => { b.scale = store.get("post.grain.scale") as number; });
  }

  dispose(): void {
    this.unsub?.();
    this.unsub = null;
  }
}
