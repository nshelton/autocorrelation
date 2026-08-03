import type { FolderApi } from "tweakpane";
import type { PresetScope, PresetStore } from "./PresetStore";
import { persistFold } from "./foldState";

// Collapsible "Presets" folder appended to the bottom of a module's section.
// Same shape for every module — components, Camera, Post, Physics — the only
// per-module input is `scope` (which param prefixes the module owns):
//
//   [ swirly ][ chaos  ]   ← half-width chips, two per row; click to load
//   [ dense  ]                 white = loaded, gray = loaded + edited, black = not loaded
//   [ 💾 ][ + ][ 🗑 ]        ← overwrite / new / delete, one row
//
// Raw DOM rather than tweakpane blades: blades are full-width rows, which is
// what made this section taller than the params it belongs to.
export function addPresetSection(
  parent: FolderApi,
  presets: PresetStore,
  scope: PresetScope,
): { dispose(): void } {
  const folder = parent.addFolder({ title: "Presets", expanded: false });
  persistFold(folder, `presets:${scope.id}`);
  ensureCss();

  const content = folder.element.querySelector<HTMLElement>(".tp-fldv_c") ?? folder.element;
  const grid = document.createElement("div");
  grid.className = "pset-grid";
  const actions = document.createElement("div");
  actions.className = "pset-actions";
  content.append(grid, actions);

  const saveBtn = iconButton(ICON_SAVE, "save", "overwrite the loaded preset");
  const newBtn = iconButton(ICON_PLUS, "new preset", "save as a new preset");
  const delBtn = iconButton(ICON_TRASH, "delete", "delete the loaded preset");
  actions.append(saveBtn, newBtn, delBtn);

  saveBtn.addEventListener("click", () => {
    const name = presets.current(scope);
    if (name) presets.save(scope, name);
    else promptName(newBtn, (n) => presets.save(scope, n));
  });
  newBtn.addEventListener("click", () => promptName(newBtn, (n) => presets.save(scope, n)));
  delBtn.addEventListener("click", () => {
    const name = presets.current(scope);
    if (name) presets.remove(scope, name);
  });

  let chips: Array<{ el: HTMLButtonElement; name: string }> = [];

  const rebuild = () => {
    grid.textContent = "";
    chips = presets.list(scope).map((preset) => {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "pset-chip";
      el.textContent = preset.name;
      el.title = preset.name;
      el.addEventListener("click", () => presets.apply(scope, preset.name));
      grid.appendChild(el);
      return { el, name: preset.name };
    });
    paint();
  };

  // Fill encodes the state: white = this preset is loaded and matches the live
  // params, gray = loaded but edited since, black = not loaded.
  const paint = () => {
    const current = presets.current(scope);
    const dirty = current !== null && presets.isDirty(scope);
    for (const chip of chips) {
      const isCurrent = chip.name === current;
      chip.el.classList.toggle("current", isCurrent && !dirty);
      chip.el.classList.toggle("dirty", isCurrent && dirty);
    }
    delBtn.disabled = current === null;
  };

  rebuild();
  // Dirty can flip on any param write anywhere in the scope, and ParamStore has
  // no aggregate change event — poll, like the tweakpane monitor this replaced.
  const timer = setInterval(paint, 200);
  // Deferred: the store emits from inside a chip's own click handler, and
  // rebuild() replaces the chip that is mid-dispatch.
  const unsub = presets.subscribe((id) => {
    if (id === scope.id) queueMicrotask(rebuild);
  });

  return {
    dispose: () => {
      unsub();
      clearInterval(timer);
      grid.remove();
      actions.remove();
    },
  };
}

// Small floating name input anchored under a button. Shared with the
// system-preset grid.
export function promptName(anchor: HTMLElement, onOk: (name: string) => void): void {
  ensureCss();
  const root = document.createElement("div");
  root.className = "preset-prompt gui-el";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "preset name";
  input.spellcheck = false;
  const ok = document.createElement("button");
  ok.textContent = "save";
  const cancel = document.createElement("button");
  cancel.textContent = "cancel";
  const row = document.createElement("div");
  row.className = "preset-prompt-row";
  row.append(ok, cancel);
  root.append(input, row);
  document.body.appendChild(root);

  const close = () => {
    root.remove();
    document.removeEventListener("pointerdown", outside, true);
  };
  const commit = () => {
    const name = input.value.trim();
    if (name) onOk(name);
    close();
  };
  function outside(e: PointerEvent) {
    if (!root.contains(e.target as Node)) close();
  }
  document.addEventListener("pointerdown", outside, true);
  ok.addEventListener("click", commit);
  cancel.addEventListener("click", close);
  // The app's global shortcuts live on window in the bubble phase ('h' hides
  // the GUI, space moves the camera) — stop typing from reaching them.
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") commit();
    else if (e.key === "Escape") close();
  });

  const r = anchor.getBoundingClientRect();
  root.style.left = `${Math.min(r.left, window.innerWidth - root.offsetWidth - 4)}px`;
  root.style.top = `${Math.min(r.bottom + 2, window.innerHeight - root.offsetHeight - 4)}px`;
  input.focus();
}

function iconButton(svg: string, name: string, tip: string): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "pset-icon";
  b.title = tip;
  // Queried by name in tests and useful for assistive tech — the label itself
  // is never painted (the SVG is).
  b.setAttribute("aria-label", name);
  b.innerHTML = svg;
  return b;
}

// Inline SVG rather than glyphs or emoji: renders identically everywhere and
// inherits currentColor for the hover state.
const ICON_SAVE =
  '<svg viewBox="0 0 16 16" width="11" height="11" fill="none" stroke="currentColor" stroke-width="1.2">' +
  '<path d="M2.6 2.6h8.2l2.6 2.6v8.2H2.6z"/><path d="M5.4 2.6v4h4.4v-4"/><path d="M5 13.4V9.4h6v4"/></svg>';
const ICON_PLUS =
  '<svg viewBox="0 0 16 16" width="11" height="11" fill="none" stroke="currentColor" stroke-width="1.6">' +
  '<path d="M8 3.2v9.6M3.2 8h9.6"/></svg>';
const ICON_TRASH =
  '<svg viewBox="0 0 16 16" width="11" height="11" fill="none" stroke="currentColor" stroke-width="1.2">' +
  '<path d="M2.8 4.4h10.4"/><path d="M6.4 4.4V2.6h3.2v1.8"/><path d="M4.4 4.4l.6 9h6l.6-9"/></svg>';

let cssInjected = false;
function ensureCss(): void {
  if (cssInjected) return;
  cssInjected = true;
  const style = document.createElement("style");
  style.textContent = `
.pset-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 2px; padding: 2px 4px 0;
}
.pset-chip {
  padding: 1px 4px; min-width: 0; cursor: pointer;
  font: 10px/14px system-ui, sans-serif; text-align: left;
  color: #bbb; background: #000; border: 1px solid #3a3a3a; border-radius: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pset-chip:hover { border-color: #777; color: #fff; }
/* Loaded and unmodified. */
.pset-chip.current { background: #fff; color: #111; border-color: #fff; }
/* Loaded, but the live params have drifted from it. */
.pset-chip.dirty { background: #888; color: #111; border-color: #888; }
.pset-actions { display: flex; gap: 2px; padding: 3px 4px 4px; }
.pset-icon {
  flex: 1; height: 16px; padding: 0; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  color: #999; background: #2a2a2a; border: 1px solid #444; border-radius: 2px;
}
.pset-icon:hover { color: #fff; border-color: #777; }
.pset-icon:disabled { opacity: 0.35; cursor: default; }
.pset-icon:disabled:hover { color: #999; border-color: #444; }
.preset-prompt {
  position: fixed; z-index: 1001; width: 180px; padding: 6px;
  background: #1c1c1c; border: 1px solid #444; border-radius: 4px;
  display: flex; flex-direction: column; gap: 4px;
  font-family: system-ui, sans-serif; font-size: 11px;
}
.preset-prompt input {
  background: #2a2a2a; color: #ccc; border: 1px solid #444; border-radius: 3px;
  padding: 3px 5px; font: inherit; outline: none;
}
.preset-prompt input:focus { border-color: #1ea7e1; }
.preset-prompt-row { display: flex; gap: 4px; }
.preset-prompt button {
  flex: 1; padding: 3px 0; font: inherit; cursor: pointer;
  background: #2a2a2a; color: #ccc; border: 1px solid #444; border-radius: 3px;
}
.preset-prompt button:hover { border-color: #666; color: #fff; }
`;
  document.head.appendChild(style);
}
