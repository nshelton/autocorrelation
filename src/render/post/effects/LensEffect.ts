// @ts-expect-error - local TSL node, no .d.ts
import { lens } from "../LensNode.js";
import type { ShaderNodeObject } from "three/tsl";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";
import type { Modulator } from "../../../params/Modulator";
import type { ParamProxyRegistry } from "../../../params/bindParam";

// Wraps LensNode (barrel + CA + vignette in one resample). Hot updates write
// `lensNode.<field>.value`; rebuild via PostStack only triggers on enable toggle.
export class LensEffect implements PostEffect {
  readonly id = "lens";
  readonly label = "Lens";
  readonly needs = {} as const;
  enabled = false;

  private lensNode: {
    distortion: { value: number };
    chromatic: { value: number };
    vignetteStrength: { value: number };
    vignetteRadius: { value: number };
  } | null = null;
  private store: ParamStore | null = null;
  private unsub: (() => void) | null = null;

  registerParams(store: ParamStore): void {
    this.store = store;
    this.unsub = store.subscribe((key, value) => {
      if (typeof value !== "number" || !this.lensNode) return;
      if (key === "post.lens.distortion")          this.lensNode.distortion.value = value;
      else if (key === "post.lens.chromatic")      this.lensNode.chromatic.value = value;
      else if (key === "post.lens.vignette")       this.lensNode.vignetteStrength.value = value;
      else if (key === "post.lens.vignetteRadius") this.lensNode.vignetteRadius.value = value;
    });
  }

  build(input: ShaderNodeObject<Node>, _ctx: PassCtx): ShaderNodeObject<Node> {
    const s = this.store!;
    const distortion     = s.get("post.lens.distortion")     as number;
    const chromatic      = s.get("post.lens.chromatic")      as number;
    const vignette       = s.get("post.lens.vignette")       as number;
    const vignetteRadius = s.get("post.lens.vignetteRadius") as number;
    const node = lens(input, distortion, chromatic, vignette, vignetteRadius) as ShaderNodeObject<Node> & {
      distortion: { value: number };
      chromatic: { value: number };
      vignetteStrength: { value: number };
      vignetteRadius: { value: number };
    };
    this.lensNode = node;
    return node;
  }

  bindUI(
    folder: FolderApi,
    store: ParamStore,
    _modulator: Modulator,
    proxies: ParamProxyRegistry,
  ): void {
    const b = {
      enabled:        store.get("post.lens.enabled")        as boolean,
      distortion:     store.get("post.lens.distortion")     as number,
      chromatic:      store.get("post.lens.chromatic")      as number,
      vignette:       store.get("post.lens.vignette")       as number,
      vignetteRadius: store.get("post.lens.vignetteRadius") as number,
    };
    folder
      .addBinding(b, "enabled", { label: "Enabled" })
      .on("change", (e: { value: boolean }) => store.set("post.lens.enabled", e.value));
    folder
      .addBinding(b, "distortion", { label: "Distortion", min: -0.5, max: 0.5, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.lens.distortion", e.value));
    folder
      .addBinding(b, "chromatic", { label: "Chromatic", min: 0, max: 0.05, step: 0.001 })
      .on("change", (e: { value: number }) => store.set("post.lens.chromatic", e.value));
    folder
      .addBinding(b, "vignette", { label: "Vignette", min: 0, max: 1, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.lens.vignette", e.value));
    folder
      .addBinding(b, "vignetteRadius", { label: "Vig. radius", min: 0, max: 1, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.lens.vignetteRadius", e.value));
    // The widgets bind `b`, which only ever moves on user input — re-pull it so
    // programmatic writes (preset load, reset) show up in the panel.
    proxies.set("post.lens.enabled", () => { b.enabled = store.get("post.lens.enabled") as boolean; });
    proxies.set("post.lens.distortion", () => { b.distortion = store.get("post.lens.distortion") as number; });
    proxies.set("post.lens.chromatic", () => { b.chromatic = store.get("post.lens.chromatic") as number; });
    proxies.set("post.lens.vignette", () => { b.vignette = store.get("post.lens.vignette") as number; });
    proxies.set("post.lens.vignetteRadius", () => { b.vignetteRadius = store.get("post.lens.vignetteRadius") as number; });
  }

  dispose(): void {
    this.unsub?.();
    this.unsub = null;
    this.lensNode = null;
  }
}
