import type { PerspectiveCamera, Scene } from "three";
import type { ShaderNodeObject } from "three/tsl";
import type { Node } from "three/webgpu";
import type { ParamStore } from "../../params/ParamStore";
import type { Modulator } from "../../params/Modulator";
import type { ParamProxyRegistry } from "../../params/bindParam";
import type { FolderApi } from "tweakpane";

// Context handed to every effect's build() — the scene-pass texture nodes.
export interface PassCtx {
  scene: Scene;
  camera: PerspectiveCamera;
  // View-space normal. NULL if no enabled effect requested it.
  sceneNormal: ShaderNodeObject<Node> | null;
  // Always available.
  sceneDepth: ShaderNodeObject<Node>;
  // View-space Z reconstructed from depth (negative in front of the camera).
  sceneViewZ: ShaderNodeObject<Node>;
}

export interface PostEffect {
  readonly id: string;          // stable; matches param prefix (e.g. "ao")
  readonly label: string;       // panel folder title
  readonly needs: Readonly<{ normal?: boolean }>;

  // Read by PostStack at build time. Mutated by the post.<id>.enabled subscription.
  enabled: boolean;

  // Build this effect's node chain. `input` is the upstream color node.
  build(input: ShaderNodeObject<Node>, ctx: PassCtx): ShaderNodeObject<Node>;

  // Optional per-frame hook, called by PostStack right before render while
  // the effect is enabled — for uniforms driven by live scene state (e.g.
  // DOF auto-focus tracking the camera) rather than params.
  update?(): void;

  // Register param schemas + uniform subscriptions. Called once at construction.
  registerParams(store: ParamStore, requestRebuild: () => void): void;

  // Add UI widgets into the effect's sub-folder under "Post". `proxies`
  // collects per-key "re-pull the widget's value from the store" callbacks so
  // programmatic writes (preset load, reset) move the widgets instead of
  // leaving them stale — PostStack drives them.
  bindUI(
    folder: FolderApi,
    store: ParamStore,
    modulator: Modulator,
    proxies: ParamProxyRegistry,
  ): void;

  dispose(): void;
}
