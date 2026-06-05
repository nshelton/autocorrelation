import { FolderApi } from "tweakpane";

// Per-section collapsed/expanded memory. Tweakpane folders are recreated every
// page load (and on HMR), so their open state lives in localStorage keyed by a
// stable id. Default is expanded — first load shows everything open; any toggle
// the user makes is remembered.
const KEY = "paramPanel.folds";

function load(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}");
  } catch {
    return {};
  }
}

export function persistFold(folder: FolderApi, id: string): void {
  const state = load();
  if (id in state) folder.expanded = state[id];
  folder.on("fold", (ev) => {
    const s = load();
    s[id] = ev.expanded;
    localStorage.setItem(KEY, JSON.stringify(s));
  });
}
