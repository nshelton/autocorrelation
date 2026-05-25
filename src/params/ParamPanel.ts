import { FolderApi, Pane } from "tweakpane";
import { ParamStore } from "./ParamStore";
import { Modulator } from "./Modulator";
import { bindParam, type ParamProxyRegistry } from "./bindParam";

export class ParamPanel {
  public pane: Pane;
  public scenes: FolderApi;
  public camera: FolderApi;
  public post: FolderApi;
  private unsubscribe: () => void;
  private proxies: ParamProxyRegistry = new Map();

  constructor(store: ParamStore, modulator: Modulator, container?: HTMLElement) {
    this.pane = new Pane({ container });
    const folder = this.pane.addFolder({ title: "Analysis", expanded: false });

    // ParamPanel owns the DSP folder only. Component-toggle and
    // component-param schemas are rendered by ComponentManager.bindUI()
    // into their own per-component folders. We pass `this.proxies` so
    // bindParam registers a "re-pull proxy from store" callback per key;
    // the subscriber below calls those + pane.refresh() to restore the
    // existing Reset-snaps-sliders behavior for dsp.* params.
    for (const schema of store.schemasInOrder()) {
      if (!schema.key.startsWith("dsp.")) continue;
      bindParam(folder, store, modulator, schema, this.proxies);
    }

    // Gated on source==='user' so per-frame modulator notifies don't
    // jitter the UI.
    this.unsubscribe = store.subscribe((key, _value, source) => {
      if (source !== "user") return;
      const refresh = this.proxies.get(key);
      if (!refresh) return;
      refresh();
      this.pane.refresh();
    });

    this.scenes = this.pane.addFolder({ title: "Scenes" });
    this.camera = this.pane.addFolder({ title: "Camera", expanded: false });
    this.post = this.pane.addFolder({ title: "Post", expanded: false });
    this.pane.addButton({ title: "Reset to defaults" }).on("click", () => store.reset());
    this.pane.addButton({ title: "Reset modulation" }).on("click", () => {
      for (const schema of store.schemasInOrder()) {
        if (modulator.getBinding(schema.key)) modulator.setBinding(schema.key, null);
      }
    });
  }

  dispose(): void {
    this.unsubscribe();
    this.proxies.clear();
    this.pane.dispose();
  }
}
