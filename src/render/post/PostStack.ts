import { PostProcessing } from "three/webgpu";
import { pass, mrt, output, transformedNormalView } from "three/tsl";
import type { PerspectiveCamera, Scene } from "three";
import type { Node, TextureNode, WebGPURenderer } from "three/webgpu";
import type { ShaderNodeObject } from "three/tsl";
import type { ParamStore } from "../../params/ParamStore";
import type { FolderApi } from "tweakpane";
import type { PostEffect, PassCtx } from "./PostEffect";

// Owns a single PostProcessing instance plus an ordered list of effects.
// Subscribes to ParamStore: any `post.*.enabled` change reads each effect's
// `enabled` from the store, recomputes MRT requirements, and rebuilds
// `outputNode`. Topology-changing param keys (e.g. tonemap mode) can also
// request rebuild via the `requestRebuild` callback passed to registerParams.
//
// PostProcessing has no dispose(); we keep one instance for the App lifetime
// and just reassign outputNode + flip needsUpdate on rebuild.
export class PostStack {
  private post: PostProcessing;
  private rebuildScheduled = false;
  private subStore: () => void;
  private effects: PostEffect[];

  constructor(
    renderer: WebGPURenderer,
    private scene: Scene,
    private camera: PerspectiveCamera,
    private store: ParamStore,
    effects: PostEffect[],
  ) {
    this.effects = effects;
    this.post = new PostProcessing(renderer);

    // Per-effect param subscriptions. Each effect can call requestRebuild()
    // when one of its keys changes graph topology (e.g. tonemap mode).
    for (const effect of this.effects) {
      effect.registerParams(this.store, () => this.scheduleRebuild());
    }

    // Toggle subscription — flips effect.enabled and triggers rebuild.
    this.subStore = this.store.subscribe((key, value) => {
      const m = key.match(/^post\.([^.]+)\.enabled$/);
      if (!m) return;
      const id = m[1];
      const effect = this.effects.find((e) => e.id === id);
      if (!effect) return;
      if (typeof value !== "boolean") return;
      effect.enabled = value;
      this.scheduleRebuild();
    });

    // Seed effect.enabled from current store values.
    for (const effect of this.effects) {
      effect.enabled = this.store.get(`post.${effect.id}.enabled`) as boolean;
    }
  }

  build(): void {
    const enabled = this.effects.filter((e) => e.enabled);
    const needsNormal = enabled.some((e) => e.needs.normal);

    const scenePass = pass(this.scene, this.camera);
    scenePass.setMRT(
      needsNormal
        ? mrt({ output, normal: transformedNormalView })
        : mrt({ output }),
    );

    // @types/three declares getTextureNode as ShaderNodeObject<Node>, but at
    // runtime it returns a TextureNode subclass — cast through unknown to
    // match the PassCtx contract (consumers index uvNode/levelNode/etc.).
    const sceneColor = scenePass.getTextureNode("output") as unknown as ShaderNodeObject<TextureNode>;
    const sceneNormal = needsNormal
      ? (scenePass.getTextureNode("normal") as unknown as ShaderNodeObject<TextureNode>)
      : null;
    const sceneDepth = scenePass.getTextureNode("depth") as unknown as ShaderNodeObject<TextureNode>;

    const ctx: PassCtx = {
      scene: this.scene,
      camera: this.camera,
      sceneNormal,
      sceneDepth,
    };

    let node: Node = sceneColor;
    for (const effect of enabled) node = effect.build(node, ctx);

    this.post.outputNode = node;
    this.post.needsUpdate = true;
  }

  bindUI(folder: FolderApi): void {
    for (const effect of this.effects) {
      const sub = folder.addFolder({ title: effect.label, expanded: false });
      effect.bindUI(sub, this.store);
    }
  }

  async renderAsync(): Promise<void> {
    await this.post.renderAsync();
  }

  dispose(): void {
    this.subStore();
    for (const effect of this.effects) effect.dispose();
    // PostProcessing has no dispose. Nothing to free per-instance.
  }

  private scheduleRebuild(): void {
    if (this.rebuildScheduled) return;
    this.rebuildScheduled = true;
    queueMicrotask(() => {
      this.rebuildScheduled = false;
      this.build();
    });
  }
}
