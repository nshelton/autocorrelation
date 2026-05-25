import { FolderApi, Pane } from "tweakpane";
import { ParamStore, type ParamSchema, type ParamValue } from "./ParamStore";

export class ParamPanel {
  public pane: Pane;
  public scenes: FolderApi;
  public camera: FolderApi;
  public post: FolderApi;
  private bindings: Record<string, ParamValue> = {};
  private unsubscribe: () => void;

  constructor(store: ParamStore, container?: HTMLElement) {
    this.pane = new Pane({ container });
    const folder = this.pane.addFolder({ title: "Analysis", expanded: false });

    // ParamPanel owns the DSP folder only. Component-toggle and
    // component-param schemas are rendered by ComponentManager.bindUI()
    // into their own per-component folders. Without this filter, HMR
    // double-renders component widgets here AND in the component folders
    // (ParamStore.register is idempotent, so on the 2nd page-lifetime mount
    // it has already-registered component schemas in scope).
    for (const schema of store.schemasInOrder()) {
      if (!schema.key.startsWith("dsp.")) continue;
      this.bindings[schema.key] = store.get(schema.key);
      const widget = this.addWidget(folder, schema);
      widget.on("change", (e: { value: ParamValue }) => store.set(schema.key, e.value));
    }

    this.unsubscribe = store.subscribe((key, value, source) => {
      if (source !== "user") return;
      if (!key.startsWith("dsp.")) return;
      if (this.bindings[key] !== value) {
        this.bindings[key] = value;
        this.pane.refresh();
      }
    });

    this.scenes = this.pane.addFolder({ title: "Scenes" });
    this.camera = this.pane.addFolder({ title: "Camera", expanded: false });
    this.post = this.pane.addFolder({ title: "Post", expanded: false });
    this.pane.addButton({ title: "Reset to defaults" }).on("click", () => store.reset());
  }

  dispose(): void {
    this.unsubscribe();
    this.pane.dispose();
  }

  private addWidget(folder: ReturnType<Pane["addFolder"]>, schema: ParamSchema) {
    if (schema.kind === "boolean") {
      // tweakpane infers a checkbox from the boolean field type.
      return folder.addBinding(this.bindings, schema.key, { label: schema.label });
    }
    if (schema.kind === "discrete") {
      const labels = schema.optionLabels ?? schema.options.map(String);
      return folder.addBinding(this.bindings, schema.key, {
        label: schema.label,
        options: Object.fromEntries(schema.options.map((v, i) => [labels[i], v])),
      });
    }
    return folder.addBinding(this.bindings, schema.key, {
      label: schema.label,
      min: schema.min,
      max: schema.max,
      step: schema.step,
    });
  }
}
