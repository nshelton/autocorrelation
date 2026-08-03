import type { FolderApi } from "tweakpane";
import type { PresetScope, PresetStore } from "./PresetStore";
import { promptName } from "./PresetSection";

// The whole-system scope: empty prefixes means every registered param and
// every modulation, including each component's `enabled` flag — so a system
// preset restores which modules are on, not just their settings.
const SYSTEM: PresetScope = { id: "system", prefixes: [] };

// Thumbnail grid of full-system snapshots, injected into the far-left System
// folder. Raw DOM rather than tweakpane blades because tweakpane has no image
// button; the grid div is appended to the folder's content container first, so
// it renders above the save/overwrite buttons added after it.
export function addSystemPresets(
  folder: FolderApi,
  presets: PresetStore,
  capture: () => Promise<string>,
): { dispose(): void } {
  ensureCss();

  const content = folder.element.querySelector<HTMLElement>(".tp-fldv_c") ?? folder.element;
  const grid = document.createElement("div");
  grid.className = "sys-grid";
  content.appendChild(grid);

  const rebuild = () => {
    grid.textContent = "";
    const current = presets.current(SYSTEM);
    for (const preset of presets.list(SYSTEM)) {
      const square = document.createElement("button");
      square.type = "button";
      square.className = "sys-thumb";
      square.title = preset.name;
      if (preset.name === current) square.classList.add("active");
      if (preset.thumb) square.style.backgroundImage = `url(${preset.thumb})`;

      const caption = document.createElement("span");
      caption.className = "sys-cap";
      caption.textContent = preset.name;
      square.appendChild(caption);

      const del = document.createElement("span");
      del.className = "sys-del";
      del.textContent = "×";
      del.title = `delete "${preset.name}"`;
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        presets.remove(SYSTEM, preset.name);
      });
      square.appendChild(del);

      square.addEventListener("click", () => presets.apply(SYSTEM, preset.name));
      grid.appendChild(square);
    }
    if (presets.list(SYSTEM).length === 0) {
      const hint = document.createElement("div");
      hint.className = "sys-empty";
      hint.textContent = "no system presets yet";
      grid.appendChild(hint);
    }
  };
  rebuild();

  const saveAs = async (name: string) => presets.save(SYSTEM, name, await capture());

  const saveBtn = folder.addButton({ title: "save system" });
  saveBtn.on("click", () => {
    const name = presets.current(SYSTEM);
    // Overwriting re-captures the thumbnail so the image tracks the state.
    if (name) void saveAs(name);
    else promptName(saveBtn.element, (n) => void saveAs(n));
  });

  const newBtn = folder.addButton({ title: "+ new system preset" });
  newBtn.on("click", () => promptName(newBtn.element, (n) => void saveAs(n)));

  // Deferred: the store emits from inside a square's own click handler, and
  // rebuild() clears the grid those squares live in.
  const unsub = presets.subscribe((id) => {
    if (id === SYSTEM.id) queueMicrotask(rebuild);
  });

  return {
    dispose: () => {
      unsub();
      grid.remove();
    },
  };
}

let cssInjected = false;
function ensureCss(): void {
  if (cssInjected) return;
  cssInjected = true;
  const style = document.createElement("style");
  style.textContent = `
.sys-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(52px, 1fr));
  gap: 4px; padding: 4px 4px 6px;
}
.sys-thumb {
  position: relative; aspect-ratio: 1; padding: 0; cursor: pointer;
  background: #000 center/cover no-repeat;
  border: 1px solid #444; border-radius: 3px; overflow: hidden;
}
.sys-thumb:hover { border-color: #888; }
.sys-thumb.active { border-color: #1ea7e1; box-shadow: 0 0 0 1px #1ea7e1; }
.sys-cap {
  position: absolute; left: 0; right: 0; bottom: 0; padding: 1px 3px;
  font: 9px/1.3 system-ui, sans-serif; color: #ddd; text-align: left;
  background: rgba(0, 0, 0, 0.6);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sys-del {
  position: absolute; top: 1px; right: 2px; width: 12px; height: 12px;
  font: 11px/11px system-ui, sans-serif; color: #ccc; text-align: center;
  background: rgba(0, 0, 0, 0.55); border-radius: 2px; opacity: 0;
}
.sys-thumb:hover .sys-del { opacity: 1; }
.sys-del:hover { color: #fff; background: #c0392b; }
.sys-empty {
  grid-column: 1 / -1; padding: 2px 2px 4px;
  font: 10px system-ui, sans-serif; color: #777;
}
`;
  document.head.appendChild(style);
}
