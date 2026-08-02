import {
  NoToneMapping,
  AgXToneMapping,
  ACESFilmicToneMapping,
  NeutralToneMapping,
  type ToneMapping,
} from "three";
import { uniform } from "three/tsl";
import type { ShaderNodeObject } from "three/tsl";
import type { Node, WebGPURenderer } from "three/webgpu";
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

// Drives the renderer's tone-mapping constant (which PostProcessing applies
// via renderOutput in update()) and inserts an exposure uniform multiply
// before that. When disabled, sets renderer.toneMapping = NoToneMapping
// and skips the multiply.
//
// Mode and enabled changes require a rebuild because renderOutput is baked
// into the QuadMesh material at update() time — flipping needsUpdate is what
// picks up a new renderer.toneMapping value.
export class TonemapEffect implements PostEffect {
  readonly id = "tonemap";
  readonly label = "Tonemap";
  readonly needs = {} as const;
  enabled = true;

  private renderer: WebGPURenderer;
  private exposureU = uniform(1.0);
  private unsub: (() => void) | null = null;

  constructor(renderer: WebGPURenderer) {
    this.renderer = renderer;
  }

  registerParams(store: ParamStore, requestRebuild: () => void): void {
    // Seed renderer tone-mapping and exposure from store.
    this.applyMode(store);
    this.exposureU.value = store.get("post.tonemap.exposure") as number;

    this.unsub = store.subscribe((key, value) => {
      if (key === "post.tonemap.exposure" && typeof value === "number") {
        this.exposureU.value = value;
      } else if (key === "post.tonemap.mode") {
        this.applyMode(store);
        requestRebuild();   // bake new renderer.toneMapping into material
      } else if (key === "post.tonemap.enabled") {
        this.applyMode(store);
        // enabled also triggers the PostStack-level rebuild via the
        // `post.*.enabled` subscription in PostStack — no requestRebuild needed.
      }
    });
  }

  build(input: ShaderNodeObject<Node>, _ctx: PassCtx): ShaderNodeObject<Node> {
    return input.mul(this.exposureU);
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
    ]) {
      const schema = store.schemaFor(key);
      if (!schema) throw new Error(`TonemapEffect.bindUI: schema ${key} missing`);
      bindParam(folder, store, modulator, schema, proxies);
    }
  }

  dispose(): void {
    this.unsub?.();
    this.unsub = null;
    // HMR order is teardown then build, so the new TonemapEffect's
    // applyMode() will restore renderer.toneMapping in registerParams.
    this.renderer.toneMapping = NoToneMapping;
  }

  private applyMode(store: ParamStore): void {
    const enabled = store.get("post.tonemap.enabled") as boolean;
    const modeIdx = store.get("post.tonemap.mode") as number;
    this.renderer.toneMapping = enabled ? (TONEMAP_TABLE[modeIdx] ?? NoToneMapping) : NoToneMapping;
  }
}
