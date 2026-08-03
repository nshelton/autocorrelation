import { Vector3 } from "three";
import { uniform } from "three/tsl";
import type { PerspectiveCamera } from "three";
import type { ShaderNodeObject } from "three/tsl";
import type { Node } from "three/webgpu";
import type { FolderApi } from "tweakpane";
// @ts-expect-error - local TSL node, no .d.ts
import { bokeh } from "../BokehNode.js";
import type { PostEffect, PassCtx } from "../PostEffect";
import type { ParamStore } from "../../../params/ParamStore";
import type { Modulator } from "../../../params/Modulator";
import { bindParam, type ParamProxyRegistry } from "../../../params/bindParam";

// blades = 0 means "circle" in the schema; a 48-gon is visually a circle and
// keeps the shader branch-free.
const CIRCLE_SIDES = 48;

export class DofEffect implements PostEffect {
  readonly id = "dof";
  readonly label = "DOF";
  readonly needs = {} as const;
  enabled = false;

  // We own the uniforms and hand them to bokeh(), so hot updates are plain
  // .value writes that survive rebuilds.
  private focusU = uniform(4.0);
  private apertureU = uniform(0.01);
  private bladesU = uniform(5);
  private rotationU = uniform(0);
  private boostU = uniform(1.0);
  private unsub: (() => void) | null = null;
  // Auto-focus drives focusU from the camera each frame (update() below),
  // overriding the focus slider while on. Camera is grabbed from build ctx.
  private autoFocus = false;
  private camera: PerspectiveCamera | null = null;
  private fwd = new Vector3();

  registerParams(store: ParamStore): void {
    this.unsub = store.subscribe((key, value) => {
      if (key === "post.dof.autoFocus" && typeof value === "boolean") {
        this.autoFocus = value;
        // Hand control back to the slider on toggle-off.
        if (!value) this.focusU.value = store.get("post.dof.focus") as number;
        return;
      }
      if (typeof value !== "number") return;
      if (key === "post.dof.focus") this.focusU.value = value;
      else if (key === "post.dof.aperture") this.apertureU.value = value;
      else if (key === "post.dof.blades") this.bladesU.value = value === 0 ? CIRCLE_SIDES : value;
      else if (key === "post.dof.rotation") this.rotationU.value = value;
      else if (key === "post.dof.boost") this.boostU.value = value;
    });
    this.focusU.value = store.get("post.dof.focus") as number;
    this.apertureU.value = store.get("post.dof.aperture") as number;
    const blades = store.get("post.dof.blades") as number;
    this.bladesU.value = blades === 0 ? CIRCLE_SIDES : blades;
    this.rotationU.value = store.get("post.dof.rotation") as number;
    this.boostU.value = store.get("post.dof.boost") as number;
    this.autoFocus = store.get("post.dof.autoFocus") as boolean;
  }

  // Focal-plane distance to the world origin: dot(origin − camPos, forward)
  // = −camPos·forward. Plane distance, not euclidean — origin stays sharp
  // even when swim/rotate aims the camera slightly off-center.
  update(): void {
    if (!this.autoFocus || !this.camera) return;
    this.camera.getWorldDirection(this.fwd);
    this.focusU.value = -this.fwd.dot(this.camera.position);
  }

  build(input: ShaderNodeObject<Node>, ctx: PassCtx): ShaderNodeObject<Node> {
    this.camera = ctx.camera;
    return bokeh(
      input, ctx.sceneViewZ,
      this.focusU, this.apertureU,
      this.bladesU, this.rotationU, this.boostU,
    ) as ShaderNodeObject<Node>;
  }

  bindUI(
    folder: FolderApi,
    store: ParamStore,
    modulator: Modulator,
    proxies: ParamProxyRegistry,
  ): void {
    for (const key of [
      "post.dof.enabled",
      "post.dof.autoFocus",
      "post.dof.focus",
      "post.dof.aperture",
      "post.dof.blades",
      "post.dof.rotation",
      "post.dof.boost",
    ]) {
      const schema = store.schemaFor(key);
      if (!schema) throw new Error(`DofEffect.bindUI: schema ${key} missing`);
      bindParam(folder, store, modulator, schema, proxies);
    }
  }

  dispose(): void {
    this.unsub?.();
    this.unsub = null;
  }
}
