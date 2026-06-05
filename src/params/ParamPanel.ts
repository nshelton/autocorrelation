import { FolderApi, Pane } from "tweakpane";
import { ParamStore } from "./ParamStore";
import { Modulator } from "./Modulator";
import { bindParam, addGraph, type ParamProxyRegistry } from "./bindParam";
import { modPopover } from "./ModPopover";
import { persistFold } from "./foldState";

// Top-anchored, flex-wrap row of columns — one Pane per section so the panel
// spreads across the top instead of growing into one very long column. Each
// column scrolls independently if it gets tall.
export class ParamPanel {
  public scenes: FolderApi;
  public camera: FolderApi;
  public post: FolderApi;
  private container: HTMLDivElement;
  private panes: Pane[] = [];
  private unsubscribe: () => void;
  private proxies: ParamProxyRegistry = new Map();

  constructor(store: ParamStore, modulator: Modulator, host?: HTMLElement) {
    const mount = host ?? document.body;
    this.container = document.createElement("div");
    // pointer-events:none on the row so clicks fall through the gaps to the
    // canvas; each column re-enables them.
    this.container.style.cssText =
      "position:fixed; top:0; left:0; right:0; z-index:10; display:flex;" +
      " flex-wrap:wrap; align-items:flex-start; gap:8px; padding:8px;" +
      " pointer-events:none;";
    this.container.classList.add("gui-el");   // fades/hides with the rest of the GUI
    mount.appendChild(this.container);

    // ---- First column: System + Analysis + global reset buttons ----
    const firstPane = this.addColumn();

    // System: GUI-level settings. "gui transparency" fades the whole GUI — a
    // presentation control, registered for persistence but built as a plain
    // binding (no audio-mod button). Press 'h' to hide the GUI entirely.
    const system = firstPane.addFolder({ title: "System", expanded: true });
    persistFold(system, "System");
    store.register({
      key: "system.guiTransparency",
      label: "gui transparency",
      kind: "continuous",
      min: 0,
      max: 1,
      step: 0.01,
      default: 0,
      reconfig: false,
    });
    const applyGuiOpacity = (t: number) =>
      document.documentElement.style.setProperty("--gui-opacity", String(1 - t));
    const guiProxy = { value: store.get("system.guiTransparency") as number };
    system
      .addBinding(guiProxy, "value", { label: "gui transparency", min: 0, max: 1, step: 0.01 })
      .on("change", (e: { value: number }) => {
        store.set("system.guiTransparency", e.value);
        applyGuiOpacity(e.value);
      });
    applyGuiOpacity(guiProxy.value); // honor persisted value on load

    // Live DSP band signals (autogained 0..1), read through the modulator's
    // source accessor — same data the rms.low/mid/high modulation sources use.
    const signals = firstPane.addFolder({ title: "Signals", expanded: true });
    persistFold(signals, "Signals");
    addGraph(signals, "low", () => modulator.readSource("rms.low"), 3);
    addGraph(signals, "mid", () => modulator.readSource("rms.mid"), 3);
    addGraph(signals, "high", () => modulator.readSource("rms.high"), 3);

    const analysis = firstPane.addFolder({ title: "Analysis", expanded: true });
    persistFold(analysis, "Analysis");
    // ParamPanel owns the DSP folder only. Component / camera / post params are
    // rendered into their own columns by App. We pass `this.proxies` so bindParam
    // registers a "re-pull proxy from store" callback per dsp key; the subscriber
    // below calls those + pane.refresh() to keep Reset-snaps-sliders working.
    for (const schema of store.schemasInOrder()) {
      if (!schema.key.startsWith("dsp.")) continue;
      bindParam(analysis, store, modulator, schema, this.proxies);
    }
    firstPane.addButton({ title: "Reset to defaults" }).on("click", () => store.reset());
    firstPane.addButton({ title: "Reset modulation" }).on("click", () => {
      for (const schema of store.schemasInOrder()) {
        if (modulator.getBinding(schema.key)) modulator.setBinding(schema.key, null);
      }
    });

    // Gated on source==='user' so per-frame modulator notifies don't jitter the UI.
    this.unsubscribe = store.subscribe((key, _value, source) => {
      if (source !== "user") return;
      const refresh = this.proxies.get(key);
      if (!refresh) return;
      refresh();
      firstPane.refresh();
    });

    // ---- Remaining columns ----
    this.scenes = this.addColumn().addFolder({ title: "Scenes", expanded: true });
    this.camera = this.addColumn().addFolder({ title: "Camera", expanded: true });
    this.post = this.addColumn().addFolder({ title: "Post", expanded: true });
    persistFold(this.scenes, "Scenes");
    persistFold(this.camera, "Camera");
    persistFold(this.post, "Post");
  }

  private addColumn(): Pane {
    const col = document.createElement("div");
    // Transparent pane base so the columns float over the canvas. Folder title
    // bars keep a faint container background (set at :root) so each section
    // stays outlined.
    // overflow-x:hidden so a value cell that's a hair wider than the content
    // box (or the y-scrollbar eating width) never spawns a horizontal scrollbar.
    col.style.cssText =
      "pointer-events:auto; width:256px; max-height:100vh;" +
      " overflow-x:hidden; overflow-y:auto;" +
      " --tp-base-background-color:transparent;";
    this.container.appendChild(col);
    const pane = new Pane({ container: col });
    this.panes.push(pane);
    return pane;
  }

  dispose(): void {
    this.unsubscribe();
    this.proxies.clear();
    modPopover.close();
    for (const p of this.panes) p.dispose();
    this.panes = [];
    this.container.remove();
  }
}
