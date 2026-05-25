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

  constructor(renderer: WebGPURenderer) {
    this.renderer = renderer;
  }

  registerParams(store: ParamStore, requestRebuild: () => void): void {
    // Seed renderer tone-mapping and exposure from store.
    this.applyMode(store);
    this.exposureU.value = store.get("post.tonemap.exposure") as number;

    store.subscribe((key, value) => {
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

  bindUI(folder: FolderApi, store: ParamStore): void {
    const b = {
      enabled: store.get("post.tonemap.enabled") as boolean,
      mode: store.get("post.tonemap.mode") as number,
      exposure: store.get("post.tonemap.exposure") as number,
    };
    folder
      .addBinding(b, "enabled", { label: "Enabled" })
      .on("change", (e: { value: boolean }) => store.set("post.tonemap.enabled", e.value));
    folder
      .addBinding(b, "mode", {
        label: "Mode",
        options: { None: 0, AgX: 1, ACES: 2, Neutral: 3 },
      })
      .on("change", (e: { value: number }) => store.set("post.tonemap.mode", e.value));
    folder
      .addBinding(b, "exposure", { label: "Exposure", min: 0, max: 4, step: 0.01 })
      .on("change", (e: { value: number }) => store.set("post.tonemap.exposure", e.value));
  }

  dispose(): void {
    this.renderer.toneMapping = NoToneMapping;
  }

  private applyMode(store: ParamStore): void {
    const enabled = store.get("post.tonemap.enabled") as boolean;
    const modeIdx = store.get("post.tonemap.mode") as number;
    this.renderer.toneMapping = enabled ? (TONEMAP_TABLE[modeIdx] ?? NoToneMapping) : NoToneMapping;
  }
}
