import { uniform } from "three/tsl";
import type { ShaderNodeObject } from "three/tsl";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
// @ts-expect-error - local copy of three's GTAONode example, no .d.ts
import { ao } from "../GTAONode.js";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";

export class AoEffect implements PostEffect {
  readonly id = "ao";
  readonly label = "AO";
  readonly needs = { normal: true } as const;
  enabled = true;

  // Live aoNode handle (rebuilt on each PostStack.build). The radius
  // subscription writes into this — guarded because the subscription
  // can fire before the first build().
  private aoNode: { radius: { value: number } } | null = null;
  private intensityU = uniform(1.0);
  private store: ParamStore | null = null;

  registerParams(store: ParamStore): void {
    this.store = store;
    store.subscribe((key, value) => {
      if (key === "post.ao.radius" && typeof value === "number") {
        if (this.aoNode) this.aoNode.radius.value = value;
      } else if (key === "post.ao.intensity" && typeof value === "number") {
        this.intensityU.value = value;
      }
    });
    this.intensityU.value = store.get("post.ao.intensity") as number;
  }

  build(input: ShaderNodeObject<Node>, ctx: PassCtx): ShaderNodeObject<Node> {
    if (!ctx.sceneNormal) throw new Error("AoEffect requires sceneNormal");
    // ao() returns a TempNode with `radius`, `thickness`, `distanceExponent`,
    // `scale` uniform fields. The .js source has no .d.ts so we capture the
    // runtime shape: a ShaderNodeObject with a `radius` uniform field.
    const aoNode = ao(ctx.sceneDepth, ctx.sceneNormal, ctx.camera) as ShaderNodeObject<Node> & { radius: { value: number } };
    this.aoNode = aoNode;
    if (this.store) aoNode.radius.value = this.store.get("post.ao.radius") as number;
    // mix(1, aoColor, intensity) = aoColor * intensity + (1 - intensity).
    // intensity = 0 → no AO contribution; intensity = 1 → full GTAO.
    const factor = aoNode.mul(this.intensityU).add(this.intensityU.oneMinus());
    return input.mul(factor);
  }

  bindUI(folder: FolderApi, store: ParamStore): void {
    const b = {
      enabled: store.get("post.ao.enabled") as boolean,
      radius: store.get("post.ao.radius") as number,
      intensity: store.get("post.ao.intensity") as number,
    };
    folder
      .addBinding(b, "enabled", { label: "Enabled" })
      .on("change", (e: { value: boolean }) => store.set("post.ao.enabled", e.value));
    folder
      .addBinding(b, "radius", { label: "Radius", min: 0.05, max: 2.0, step: 0.05 })
      .on("change", (e: { value: number }) => store.set("post.ao.radius", e.value));
    folder
      .addBinding(b, "intensity", { label: "Intensity", min: 0, max: 2, step: 0.05 })
      .on("change", (e: { value: number }) => store.set("post.ao.intensity", e.value));
  }

  dispose(): void {
    this.aoNode = null;
  }
}
