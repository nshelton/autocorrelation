import { Fn, uniform, vec2, vec4, float, fract, dot, frameId, screenCoordinate } from "three/tsl";
import type { ShaderNodeObject } from "three/tsl";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";
import type { Modulator } from "../../../params/Modulator";
import type { ParamProxyRegistry } from "../../../params/bindParam";

// Film grain via Interleaved Gradient Noise (Jorge Jimenez, COD: Advanced
// Warfare). IGN is a single-hash formula that produces a noise pattern
// visually indistinguishable from real blue noise for grain purposes — no
// low-frequency clumping, evenly distributed, very cheap. The per-frame
// `frameId` shift gives the "film moving" temporal effect; without it the
// noise pattern would be static.
//
// True blue noise would require sampling a precomputed texture; IGN is the
// industry-standard zero-asset alternative.
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
    return Fn(() => {
      // Per-pixel coord with per-frame shift. The constants 47 and 17 are
      // arbitrary primes chosen for maximum decorrelation between frames.
      const coord = screenCoordinate.xy
        .mul(this.scaleU)
        .add(float(frameId).mul(vec2(47.0, 17.0)));
      // IGN: fract(52.98... * fract(dot(coord, magic_vec2))).
      const noise = fract(
        float(52.9829189).mul(fract(dot(coord, vec2(0.06711056, 0.00583715)))),
      );
      // Center around 0; monochrome grain applied to all RGB channels equally.
      const g = noise.sub(0.5).mul(this.strengthU);
      return (input as ShaderNodeObject<Node>).add(vec4(g, g, g, 0.0));
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
