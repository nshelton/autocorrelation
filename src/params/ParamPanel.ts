import { FolderApi, Pane } from "tweakpane";
import { ParamStore } from "./ParamStore";
import { Modulator } from "./Modulator";
import { bindParam, addGraph, type ParamProxyRegistry } from "./bindParam";
import { modPopover } from "./ModPopover";
import { persistFold } from "./foldState";

const LAYOUT_KEY = "paramPanel.layout.v2";

// Panel ids per column, left to right. Empty until the user drags something.
function loadLayout(): string[][] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? "[]");
    if (!Array.isArray(raw)) return [];
    return raw
      .filter(Array.isArray)
      .map((col) => (col as unknown[]).filter((s): s is string => typeof s === "string"));
  } catch {
    return [];
  }
}

// Pointer travel before a press on a title becomes a drag rather than a fold.
const DRAG_SLOP_PX = 4;

let cssInjected = false;
function ensureCss(): void {
  if (cssInjected) return;
  cssInjected = true;
  const style = document.createElement("style");
  style.textContent = `
.pp-col {
  pointer-events: auto; width: 256px; max-height: 100vh;
  display: flex; flex-direction: column; gap: 6px;
  overflow-x: hidden; overflow-y: auto;
  --tp-base-background-color: transparent;
}
.pp-panel { flex: 0 0 auto; }
/* Folder titles are the drag handle, so advertise it. A plain click still
   folds — the drag only engages past the slop threshold. */
.pp-panel .tp-fldv_b { cursor: grab; }
.pp-dragging { opacity: 0.6; }
.pp-dragging .tp-fldv_b { cursor: grabbing; }
/* The trailing empty column is the "pull it out into its own column" target.
   Invisible until something is being dragged. */
.pp-col:empty { min-height: 56px; border-radius: 4px; }
body.pp-drag-active .pp-col:empty {
  outline: 1px dashed rgba(187, 188, 196, 0.35);
  outline-offset: -1px;
}
`;
  document.head.appendChild(style);
}

// Top-anchored, flex-wrap row of columns. Every section is its own Pane in its
// own panel div, and panels can be dragged by their folder title between and
// within columns — so several modules can share a column, or each can have its
// own. Layout persists in localStorage; the row wraps when it runs out of
// width, and each column scrolls independently if it gets tall.
export class ParamPanel {
  // Far-left box. App injects the system-preset thumbnail grid here.
  public system: FolderApi;
  public scenes: FolderApi;
  public camera: FolderApi;
  public post: FolderApi;
  private container: HTMLDivElement;
  private columns: HTMLDivElement[] = [];
  private panels = new Map<string, { el: HTMLDivElement; pane: Pane }>();
  private layout: string[][] = loadLayout();
  private unsubscribe: () => void;
  private proxies: ParamProxyRegistry = new Map();

  constructor(store: ParamStore, modulator: Modulator, host?: HTMLElement) {
    ensureCss();
    const mount = host ?? document.body;
    this.container = document.createElement("div");
    // pointer-events:none on the row so clicks fall through the gaps to the
    // canvas; each column re-enables them.
    this.container.style.cssText =
      "position:fixed; top:0; left:0; right:0; z-index:10; display:flex;" +
      " flex-wrap:wrap; align-items:flex-start; gap:8px; padding:8px;" +
      " pointer-events:none;";
    this.container.classList.add("gui-el"); // fades/hides with the rest of the GUI
    mount.appendChild(this.container);
    this.installDrag();

    // ---- System: GUI-level settings + global resets ----
    // "gui transparency" fades the whole GUI — a presentation control,
    // registered for persistence but built as a plain binding (no audio-mod
    // button). Press 'h' to hide the GUI entirely.
    const systemPane = this.addPanel("system");
    const system = systemPane.addFolder({ title: "System", expanded: true });
    this.system = system;
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

    // How long loading a preset takes to glide in. Registered in main.ts, so a
    // missing schema here means boot order changed, not a typo.
    const tweenSchema = store.schemaFor("system.presetTweenSecs");
    if (tweenSchema) bindParam(system, store, modulator, tweenSchema, this.proxies);

    system.addButton({ title: "Reset to defaults" }).on("click", () => store.reset());
    system.addButton({ title: "Reset modulation" }).on("click", () => {
      for (const schema of store.schemasInOrder()) {
        if (modulator.getBinding(schema.key)) modulator.setBinding(schema.key, null);
      }
    });

    // ---- Signals: live DSP band levels (autogained 0..1), read through the
    // modulator's source accessor — same data rms.low/mid/high modulation uses.
    const signals = this.addPanel("signals").addFolder({ title: "Signals", expanded: true });
    persistFold(signals, "Signals");
    addGraph(signals, "low", () => modulator.readSource("rms.low"), 3);
    addGraph(signals, "mid", () => modulator.readSource("rms.mid"), 3);
    addGraph(signals, "high", () => modulator.readSource("rms.high"), 3);

    // ---- Analysis: the dsp.* params. ParamPanel owns this folder only;
    // component / camera / post params are bound into their own panels by App.
    // `this.proxies` collects a "re-pull proxy from store" callback per dsp key;
    // the subscriber below calls those so Reset snaps the sliders.
    const dspPane = this.addPanel("analysis");
    const analysis = dspPane.addFolder({ title: "Analysis", expanded: true });
    persistFold(analysis, "Analysis");
    for (const schema of store.schemasInOrder()) {
      if (!schema.key.startsWith("dsp.")) continue;
      bindParam(analysis, store, modulator, schema, this.proxies);
    }

    // Gated on source==='user' so per-frame modulator notifies don't jitter the UI.
    this.unsubscribe = store.subscribe((key, _value, source) => {
      if (source !== "user") return;
      const refresh = this.proxies.get(key);
      if (!refresh) return;
      refresh();
      dspPane.refresh();
      systemPane.refresh();
    });

    this.scenes = this.addPanel("scenes").addFolder({ title: "Scenes", expanded: true });
    this.camera = this.addPanel("camera").addFolder({ title: "Camera", expanded: true });
    this.post = this.addPanel("post").addFolder({ title: "Post", expanded: true });
    persistFold(this.scenes, "Scenes");
    persistFold(this.camera, "Camera");
    persistFold(this.post, "Post");
  }

  // A panel per enabled scene, created and destroyed as scenes toggle. It lands
  // wherever the user last dragged it; failing that, in a column of its own.
  addScenePanel(id: string): { pane: Pane; dispose(): void } {
    const pane = this.addPanel(`scene:${id}`);
    return {
      pane,
      dispose: () => this.removePanel(`scene:${id}`),
    };
  }

  // Creates the panel's div + Pane and drops it into its remembered slot.
  private addPanel(id: string): Pane {
    const el = document.createElement("div");
    el.className = "pp-panel";
    el.dataset.panelId = id;
    this.placePanel(el, id);
    const pane = new Pane({ container: el });
    this.panels.set(id, { el, pane });
    return pane;
  }

  private removePanel(id: string): void {
    const panel = this.panels.get(id);
    if (!panel) return;
    this.panels.delete(id);
    panel.pane.dispose();
    panel.el.remove();
    this.tidyColumns();
  }

  // Saved layout wins; anything unknown gets a fresh column of its own, which
  // is what makes a newly enabled scene appear beside the others rather than
  // stacked under one of them.
  private placePanel(el: HTMLDivElement, id: string): void {
    const colIndex = this.layout.findIndex((c) => c.includes(id));
    if (colIndex < 0) {
      // Into the trailing empty column, which tidyColumns then replaces with a
      // fresh one. Math.max covers the very first panel, when none exist yet.
      this.columnAt(Math.max(0, this.columns.length - 1)).appendChild(el);
      this.tidyColumns();
      return;
    }
    const col = this.columnAt(colIndex);
    const wanted = this.layout[colIndex].indexOf(id);
    // Insert before the first panel whose saved position is later than ours.
    const after = [...col.children].find((c) => {
      const otherId = (c as HTMLElement).dataset.panelId ?? "";
      const at = this.layout[colIndex].indexOf(otherId);
      return at < 0 || at > wanted;
    });
    col.insertBefore(el, after ?? null);
    this.tidyColumns();
  }

  private columnAt(index: number): HTMLDivElement {
    while (this.columns.length <= index) {
      const col = document.createElement("div");
      col.className = "pp-col";
      this.container.appendChild(col);
      this.columns.push(col);
    }
    return this.columns[index];
  }

  // Drop empty columns and keep exactly one empty one at the end, which is the
  // drop target for pulling a panel out into a column of its own.
  private tidyColumns(): void {
    for (const col of [...this.columns]) {
      if (col.childElementCount > 0) continue;
      const i = this.columns.indexOf(col);
      if (i === this.columns.length - 1) continue; // keep the trailing one
      this.columns.splice(i, 1);
      col.remove();
    }
    const last = this.columns[this.columns.length - 1];
    if (!last || last.childElementCount > 0) this.columnAt(this.columns.length);
  }

  // Delegated because panels come and go: a press on any folder title starts a
  // potential drag of the panel that title belongs to. Past the slop it
  // live-moves between columns; the flex layout reflows as the DOM changes, so
  // there's no ghost element or placeholder.
  private installDrag(): void {
    this.container.addEventListener("pointerdown", (e) => {
      const target = e.target as HTMLElement | null;
      if (!target?.closest(".tp-fldv_b")) return;
      const el = target.closest<HTMLElement>(".pp-panel");
      if (!el) return;
      const startX = e.clientX;
      const startY = e.clientY;
      let dragging = false;

      const move = (ev: PointerEvent) => {
        if (!dragging) {
          if (Math.hypot(ev.clientX - startX, ev.clientY - startY) < DRAG_SLOP_PX) return;
          dragging = true;
          el.classList.add("pp-dragging");
          document.body.classList.add("pp-drag-active");
        }
        const col = this.columnUnder(ev.clientX, ev.clientY);
        if (!col) return;
        const before = [...col.children].find((c) => {
          if (c === el) return false;
          const r = c.getBoundingClientRect();
          return ev.clientY < r.top + r.height / 2;
        });
        if (before !== el) col.insertBefore(el, before ?? null);
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        if (!dragging) return; // never passed the slop: it was a fold click
        el.classList.remove("pp-dragging");
        document.body.classList.remove("pp-drag-active");
        this.tidyColumns();
        this.saveLayout();
        // Swallow the click this pointerup produces, so ending a drag on a
        // title doesn't also fold that section.
        window.addEventListener(
          "click",
          (c) => {
            c.stopPropagation();
            c.preventDefault();
          },
          { capture: true, once: true },
        );
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  }

  // Which column the pointer is over. Columns have wildly different heights, so
  // y must NOT gate the choice: dragging at the y of a tall panel across a short
  // column leaves the pointer below that column's bottom edge, and treating that
  // as "not this column" hands the drop to whichever taller column happened to
  // span that y. Horizontal distance decides (weighted 10x); vertical distance
  // only breaks ties between columns sharing an x range, which is exactly the
  // wrapped-row case. Distance is to the rect, so 0 while inside it.
  private columnUnder(x: number, y: number): HTMLDivElement | null {
    let best: HTMLDivElement | null = null;
    let bestScore = Infinity;
    for (const col of this.columns) {
      const r = col.getBoundingClientRect();
      const dx = x < r.left ? r.left - x : x > r.right ? x - r.right : 0;
      const dy = y < r.top ? r.top - y : y > r.bottom ? y - r.bottom : 0;
      const score = dx * 10 + dy;
      if (score < bestScore) {
        bestScore = score;
        best = col;
      }
    }
    return best;
  }

  private saveLayout(): void {
    this.layout = this.columns
      .map((col) => [...col.children].map((c) => (c as HTMLElement).dataset.panelId ?? ""))
      .filter((ids) => ids.length > 0);
    try {
      localStorage.setItem(LAYOUT_KEY, JSON.stringify(this.layout));
    } catch {
      // localStorage unavailable; layout lasts for this session only
    }
  }

  dispose(): void {
    this.unsubscribe();
    this.proxies.clear();
    modPopover.close();
    for (const p of this.panels.values()) {
      p.pane.dispose();
      p.el.remove();
    }
    this.panels.clear();
    this.columns = [];
    this.container.remove();
  }
}
