import type { ButtonApi, FolderApi } from "tweakpane";
import type { PresetScope, PresetStore } from "./PresetStore";
import { persistFold } from "./foldState";

// Collapsible "Presets" folder appended to the bottom of a module's section.
// Same shape for every module — components, Camera, Post — the only per-module
// input is `scope` (which param prefixes the module owns). Contents:
//
//   preset   swirly *      ← current name, "*" while edited
//   ● swirly               ← one button per saved preset; click loads it
//     chaos
//   save                   ← overwrite current (or name one if there's none)
//   + new preset           ← name dialog, saves a new one
//   delete                 ← drop the current preset
//
// The button list is rebuilt from scratch whenever the store changes; there's
// no incremental blade bookkeeping to get wrong.
export function addPresetSection(
  parent: FolderApi,
  presets: PresetStore,
  scope: PresetScope,
): { dispose(): void } {
  const folder = parent.addFolder({ title: "Presets", expanded: false });
  persistFold(folder, `presets:${scope.id}`);

  // Polled by tweakpane (readonly monitor) rather than pushed, so slider drags
  // flip the "*" without every param write touching this section.
  const status = {
    get value(): string {
      const name = presets.current(scope);
      if (!name) return "(none)";
      return presets.isDirty(scope) ? `${name} *` : name;
    },
  };

  const rebuild = () => {
    for (const child of [...folder.children]) child.dispose();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    folder.addBinding(status, "value", {
      label: "preset",
      readonly: true,
      interval: 200,
    } as any);

    const current = presets.current(scope);
    for (const preset of presets.list(scope)) {
      const active = preset.name === current;
      const b = folder.addButton({ title: active ? `● ${preset.name}` : preset.name });
      b.on("click", () => presets.apply(scope, preset.name));
    }

    const saveBtn = folder.addButton({ title: "save" });
    saveBtn.on("click", () => {
      const name = presets.current(scope);
      if (name) presets.save(scope, name);
      else promptName(saveBtn, (n) => presets.save(scope, n));
    });

    const newBtn = folder.addButton({ title: "+ new preset" });
    newBtn.on("click", () => promptName(newBtn, (n) => presets.save(scope, n)));

    const delBtn = folder.addButton({ title: "delete" });
    delBtn.disabled = current === null;
    delBtn.on("click", () => {
      const name = presets.current(scope);
      if (name) presets.remove(scope, name);
    });
  };

  rebuild();
  // Deferred: the store emits from inside a button's own click handler, and
  // rebuild() disposes that button.
  const unsub = presets.subscribe((s) => {
    if (s === scope.id) queueMicrotask(rebuild);
  });

  return { dispose: unsub };
}

function promptName(anchor: ButtonApi, onOk: (name: string) => void): void {
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

  const r = anchor.element.getBoundingClientRect();
  root.style.left = `${Math.min(r.left, window.innerWidth - root.offsetWidth - 4)}px`;
  root.style.top = `${Math.min(r.bottom + 2, window.innerHeight - root.offsetHeight - 4)}px`;
  input.focus();
}

let cssInjected = false;
function ensureCss(): void {
  if (cssInjected) return;
  cssInjected = true;
  const style = document.createElement("style");
  style.textContent = `
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
