import { Pane } from "tweakpane";
import { ParamStore, type ParamSchema, type ParamValue } from "./ParamStore";

export class ParamPanel {
  private pane: Pane;
  private bindings: Record<string, ParamValue> = {};
  private unsubscribe: () => void;

  constructor(store: ParamStore, container?: HTMLElement) {
    this.pane = new Pane({ title: "Analysis", container });
    const folder = this.pane.addFolder({ title: "DSP" });

    for (const schema of store.schemasInOrder()) {
      this.bindings[schema.key] = store.get(schema.key);
      const widget = this.addWidget(folder, schema);
      widget.on("change", (e: { value: ParamValue }) => store.set(schema.key, e.value));
    }

    this.unsubscribe = store.subscribe((key, value) => {
      if (this.bindings[key] !== value) {
        this.bindings[key] = value;
        this.pane.refresh();
      }
    });

    this.pane.addButton({ title: "Reset to defaults" }).on("click", () => store.reset());
  }

  dispose(): void {
    this.unsubscribe();
    this.pane.dispose();
  }

  private addWidget(folder: ReturnType<Pane["addFolder"]>, schema: ParamSchema) {
    if (schema.kind === "discrete") {
      return folder.addBinding(this.bindings, schema.key, {
        label: schema.label,
        options: Object.fromEntries(schema.options.map((v) => [String(v), v])),
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
