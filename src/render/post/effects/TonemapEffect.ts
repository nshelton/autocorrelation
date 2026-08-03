import {
  NoToneMapping,
  AgXToneMapping,
  ACESFilmicToneMapping,
  NeutralToneMapping,
  type ToneMapping,
} from "three";
import { uniform, vec3, vec4, mix, luminance, toneMapping } from "three/tsl";
import type { ShaderNodeObject } from "three/tsl";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";
import type { Modulator } from "../../../params/Modulator";
import { bindParam, type ParamProxyRegistry } from "../../../params/bindParam";

// Index -> three.js ToneMapping constant. Must match optionLabels in
// postSchemas.ts (`post.tonemap.mode`): ["None","AgX","ACES","Neutral"].
const TONEMAP_TABLE: ToneMapping[] = [
  NoToneMapping,
  AgXToneMapping,
  ACESFilmicToneMapping,
  NeutralToneMapping,
];

// Linear-space mid-grey; the contrast power curve pivots here so pushing
// contrast doesn't shift overall exposure.
const MID_GREY = 0.18;

// In-chain HDR grade + tonemap: exposure -> contrast (power curve about
// mid-grey) -> saturation, all in linear HDR, then the selected three.js
// curve maps to display range right here in the chain. renderer.toneMapping
// is never touched (stays at its NoToneMapping default), so PostProcessing's
// final renderOutput only does the sRGB conversion — this is what actually
// puts Lens/Grain after the tonemap, matching the canonical order in index.ts.
//
// Mode changes rebuild (the curve function is baked into the node graph);
// exposure/contrast/saturation are uniforms and hot-update.
export class TonemapEffect implements PostEffect {
  readonly id = "tonemap";
  readonly label = "Tonemap";
  readonly needs = {} as const;
  enabled = true;

  private exposureU = uniform(1.0);
  private contrastU = uniform(1.0);
  private saturationU = uniform(1.0);
  private store: ParamStore | null = null;
  private unsub: (() => void) | null = null;

  registerParams(store: ParamStore, requestRebuild: () => void): void {
    this.store = store;
    this.exposureU.value = store.get("post.tonemap.exposure") as number;
    this.contrastU.value = store.get("post.tonemap.contrast") as number;
    this.saturationU.value = store.get("post.tonemap.saturation") as number;

    this.unsub = store.subscribe((key, value) => {
      if (key === "post.tonemap.mode") {
        requestRebuild();
        return;
      }
      if (typeof value !== "number") return;
      if (key === "post.tonemap.exposure")        this.exposureU.value = value;
      else if (key === "post.tonemap.contrast")   this.contrastU.value = value;
      else if (key === "post.tonemap.saturation") this.saturationU.value = value;
    });
  }

  build(input: ShaderNodeObject<Node>, _ctx: PassCtx): ShaderNodeObject<Node> {
    const exposed = input.rgb.mul(this.exposureU);
    const contrasted = exposed.div(MID_GREY).pow(this.contrastU).mul(MID_GREY);
    const graded = vec4(
      mix(vec3(luminance(contrasted)), contrasted, this.saturationU),
      input.a,
    );
    const mode = TONEMAP_TABLE[this.store!.get("post.tonemap.mode") as number] ?? NoToneMapping;
    // Exposure already applied above, so the curve gets 1.0. Cast because
    // ToneMappingNode's own `toneMapping: number` field shadows the chaining
    // method in @types/three's ShaderNodeObject surface.
    return mode === NoToneMapping
      ? graded
      : (toneMapping(mode, 1.0, graded) as unknown as ShaderNodeObject<Node>);
  }

  bindUI(
    folder: FolderApi,
    store: ParamStore,
    modulator: Modulator,
    proxies: ParamProxyRegistry,
  ): void {
    for (const key of [
      "post.tonemap.enabled",
      "post.tonemap.mode",
      "post.tonemap.exposure",
      "post.tonemap.contrast",
      "post.tonemap.saturation",
    ]) {
      const schema = store.schemaFor(key);
      if (!schema) throw new Error(`TonemapEffect.bindUI: schema ${key} missing`);
      bindParam(folder, store, modulator, schema, proxies);
    }
  }

  dispose(): void {
    this.unsub?.();
    this.unsub = null;
  }
}
