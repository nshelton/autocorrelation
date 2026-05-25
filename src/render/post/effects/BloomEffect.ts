// @ts-expect-error - local copy of three's BloomNode addon (PostProcessingUtils
// import path patched to three/webgpu), no .d.ts
import { bloom } from "../BloomNode.js";
import type { ShaderNodeObject } from "three/tsl";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";

// Wraps three's TSL bloom(node, strength, radius, threshold). The three
// args become uniforms on the returned BloomNode; hot updates write
// `bloomNode.<field>.value`. Output is `input + bloom(input)`.
export class BloomEffect implements PostEffect {
  readonly id = "bloom";
  readonly label = "Bloom";
  readonly needs = {} as const;
  enabled = false;

  private bloomNode: { strength: { value: number }; radius: { value: number }; threshold: { value: number } } | null = null;
  private store: ParamStore | null = null;
  private unsub: (() => void) | null = null;

  registerParams(store: ParamStore): void {
    this.store = store;
    this.unsub = store.subscribe((key, value) => {
      if (typeof value !== "number" || !this.bloomNode) return;
      if (key === "post.bloom.strength")       this.bloomNode.strength.value = value;
      else if (key === "post.bloom.radius")    this.bloomNode.radius.value = value;
      else if (key === "post.bloom.threshold") this.bloomNode.threshold.value = value;
    });
  }

  build(input: ShaderNodeObject<Node>, _ctx: PassCtx): ShaderNodeObject<Node> {
    const s = this.store!;
    const strength  = s.get("post.bloom.strength")  as number;
    const radius    = s.get("post.bloom.radius")    as number;
    const threshold = s.get("post.bloom.threshold") as number;
    const node = bloom(input, strength, radius, threshold) as ShaderNodeObject<Node> & {
      strength:  { value: number };
      radius:    { value: number };
      threshold: { value: number };
    };
    this.bloomNode = node;
    return input.add(node);
  }

  bindUI(folder: FolderApi, store: ParamStore): void {
    const b = {
      enabled:   store.get("post.bloom.enabled")   as boolean,
      strength:  store.get("post.bloom.strength")  as number,
      radius:    store.get("post.bloom.radius")    as number,
      threshold: store.get("post.bloom.threshold") as number,
    };
    folder
      .addBinding(b, "enabled", { label: "Enabled" })
      .on("change", (e: { value: boolean }) => store.set("post.bloom.enabled", e.value));
    folder
      .addBinding(b, "strength", { label: "Strength", min: 0, max: 3, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.bloom.strength", e.value));
    folder
      .addBinding(b, "radius", { label: "Radius", min: 0, max: 1, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.bloom.radius", e.value));
    folder
      .addBinding(b, "threshold", { label: "Threshold", min: 0, max: 2, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.bloom.threshold", e.value));
  }

  dispose(): void {
    this.unsub?.();
    this.unsub = null;
    this.bloomNode = null;
  }
}
