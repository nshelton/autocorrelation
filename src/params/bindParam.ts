import type { FolderApi, ButtonApi } from "tweakpane";
import { ParamStore, type ParamSchema, type ParamValue } from "./ParamStore";
import { Modulator, MOD_SOURCE_KEYS } from "./Modulator";
import { modPopover } from "./ModPopover";

// Sentinel value for "no modulation" in the source dropdown.
const NONE = "none";

// Optional map a caller can pass to collect "re-pull from store into proxy"
// callbacks per param key. Only ParamPanel uses this for dsp.* sliders. Other
// call sites omit it (their sliders don't visually refresh on external writes).
export type ParamProxyRegistry = Map<string, () => void>;

// Wraps the recurring `folder.addBinding(proxy, ...).on("change", store.set)`
// pattern. For continuous, non-dsp.* schemas it injects a small inline "∿"
// button at the right of the row that opens the modulation popover (source +
// depth/power/smoothing + driving-signal graph) — replacing the old always-on
// `↳ mod` sub-folder so each param is a single row.
//
// The visible slider is bound to a LOCAL proxy. Tweakpane writes the user's
// drag into it and fires `change`; we forward to store.set. Modulator
// `notify()` calls do NOT touch the proxy; the live driven value is mirrored
// onto the slider via subscribeValue below (suppressed mid-drag).
export function bindParam(
  folder: FolderApi,
  store: ParamStore,
  modulator: Modulator,
  schema: ParamSchema,
  refreshRegistry?: ParamProxyRegistry,
): void {
  const proxy: { value: ParamValue } = { value: store.get(schema.key) };
  const binding = makeWidget(folder, proxy, schema);
  // True while the user is mid-drag — gates the live modulated refresh below so
  // it doesn't fight the drag. `last` is true on the final change of an
  // interaction (and on single click/type), false during a drag.
  let interacting = false;
  binding.on("change", (e: { value: ParamValue; last: boolean }) => {
    interacting = !e.last;
    store.set(schema.key, e.value);
  });

  refreshRegistry?.set(schema.key, () => {
    proxy.value = store.get(schema.key);
  });

  // Narrowing form (not a `modulatable` boolean) so `schema` is typed as the
  // continuous variant below — buildModControls needs its min/max/step.
  if (schema.kind !== "continuous" || schema.key.startsWith("dsp.")) return;

  // Inline mod button → floating popover. Controls are rebuilt fresh on each
  // open (reading current state), so they hold no long-lived subscription.
  const btn = injectModButton(binding.element, (b) =>
    modPopover.toggle(b, (pane) => {
      const f = pane.addFolder({ title: schema.label, expanded: true });
      buildModControls(f, modulator, schema.key, schema.min, schema.max, schema.step);
    }),
  );
  const refreshTint = () => {
    const on = modulator.getBinding(schema.key) !== null;
    btn.classList.toggle("active", on);
    // Tint the whole row (label + slider fill) accent-blue when a modulation is
    // hooked up, matching the active mod button.
    binding.element.classList.toggle("modulated", on);
  };
  refreshTint();
  void modulator.subscribe((key) => {
    if (key === schema.key) refreshTint();
  });

  // Live-update the slider to the actual driven value while modulated. Routed
  // through the Modulator (recreated + cleared per HMR) so we don't leak a
  // subscriber on the page-lifetime ParamStore. Suppressed mid-drag.
  void modulator.subscribeValue((key, value) => {
    if (key !== schema.key || interacting) return;
    proxy.value = value;
    binding.refresh();
  });
}

// Build the modulation controls for `key` into `folder`. No long-lived
// subscriptions — the popover rebuilds this each time it opens, so anything
// registered on the Modulator would accumulate. Disposed with the folder.
export function buildModControls(
  folder: FolderApi,
  modulator: Modulator,
  key: string,
  min: number,
  max: number,
  step: number,
): void {
  const existing = modulator.getBinding(key);
  const modProxy = {
    source: existing?.source ?? NONE,
    lo: existing?.lo ?? min,
    hi: existing?.hi ?? max,
    power: existing?.power ?? 1,
    smoothing: existing?.smoothing ?? 0,
  };
  const write = () => {
    if (modProxy.source === NONE) {
      modulator.setBinding(key, null);
    } else {
      modulator.setBinding(key, {
        source: modProxy.source,
        lo: modProxy.lo,
        hi: modProxy.hi,
        power: modProxy.power,
        smoothing: modProxy.smoothing,
      });
    }
  };
  folder.addBinding(modProxy, "source", { label: "source", options: sourceDropdownOptions() }).on("change", write);
  // lo/hi define the output range the signal sweeps, in the param's own units.
  folder.addBinding(modProxy, "lo", { label: "lo", min, max, step }).on("change", write);
  folder.addBinding(modProxy, "hi", { label: "hi", min, max, step }).on("change", write);
  addFilterControls(folder, modProxy, write);
  // Graph shows the smoothed + power-curved signal driving the modulation.
  addSourceMonitor(folder, () => modulator.processedValue(key));
}

// Build trigger controls for a button. Same source / power / smoothing filtering
// as continuous modulation — only the response differs (threshold edge instead
// of a lo/hi range map).
export function buildTriggerControls(
  folder: FolderApi,
  modulator: Modulator,
  triggerKey: string,
): void {
  const existing = modulator.getTrigger(triggerKey);
  const trigProxy = {
    source: existing?.source ?? NONE,
    threshold: existing?.threshold ?? 0.5,
    power: existing?.power ?? 1,
    smoothing: existing?.smoothing ?? 0,
  };
  const write = () => {
    if (trigProxy.source === NONE) {
      modulator.setTrigger(triggerKey, null);
    } else {
      modulator.setTrigger(triggerKey, {
        source: trigProxy.source,
        threshold: trigProxy.threshold,
        power: trigProxy.power,
        smoothing: trigProxy.smoothing,
      });
    }
  };
  folder.addBinding(trigProxy, "source", { label: "source", options: sourceDropdownOptions() }).on("change", write);
  folder.addBinding(trigProxy, "threshold", { label: "threshold", min: 0, max: 1, step: 0.01 }).on("change", write);
  addFilterControls(folder, trigProxy, write);
  // Threshold compares the smoothed + power-curved signal, so graph that.
  addSourceMonitor(folder, () => modulator.processedValue(triggerKey));
}

// Shared power + smoothing sliders (the source filtering common to mod bindings
// and triggers), wired to `onChange`.
function addFilterControls(
  folder: FolderApi,
  proxy: { power: number; smoothing: number },
  onChange: () => void,
): void {
  folder.addBinding(proxy, "power", { label: "power", min: 0.1, max: 10, step: 0.1 }).on("change", onChange);
  folder.addBinding(proxy, "smoothing", { label: "smoothing", min: 0, max: 1, step: 0.01 }).on("change", onChange);
}

// Inline-button trigger UI for a pushbutton: injects the "∿" button into the
// button's row (opens the trigger popover), styles the action button as a gray
// outline, and returns a flash() that pulses it white — call flash() whenever
// the action fires so audio-driven triggers are visible. Envelope = the
// trig-flash CSS animation fading white back to the outline.
export function bindTrigger(
  button: ButtonApi,
  modulator: Modulator,
  triggerKey: string,
): () => void {
  const btn = injectModButton(button.element, (b) =>
    modPopover.toggle(b, (pane) => {
      const f = pane.addFolder({ title: "trigger", expanded: true });
      buildTriggerControls(f, modulator, triggerKey);
    }),
  );
  const refreshTint = () =>
    btn.classList.toggle("active", modulator.getTrigger(triggerKey) !== null);
  refreshTint();
  void modulator.subscribe((key) => {
    if (key === triggerKey) refreshTint();
  });

  const main = button.element.querySelector<HTMLElement>(".tp-btnv_b");
  main?.classList.add("trig-btn");
  return () => {
    if (!main) return;
    main.classList.remove("trig-flash");
    void main.offsetWidth; // reflow → restart the flash envelope from white
    main.classList.add("trig-flash");
  };
}

// Source dropdown options shared by mod + trigger UIs: "none" sentinel first,
// then every curated audio source.
function sourceDropdownOptions(): Record<string, string> {
  const o: Record<string, string> = { [NONE]: NONE };
  for (const k of MOD_SOURCE_KEYS) o[k] = k;
  return o;
}

// Live 0..1 graph bound to a getter that Tweakpane polls on its own interval to
// feed the scrolling graph — no per-frame wiring, nothing to dispose. `rows`
// sets the graph's visual height.
export function addGraph(
  folder: FolderApi,
  label: string,
  read: () => number,
  rows = 1,
): void {
  const target = {
    get value(): number {
      return read();
    },
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  folder.addBinding(target, "value", {
    label,
    readonly: true,
    view: "graph",
    min: 0,
    max: 1,
    interval: 1000 / 30,
    rows,
  } as any);
}

function addSourceMonitor(folder: FolderApi, read: () => number): void {
  addGraph(folder, "level", read);
}

// Inject the small "∿" modulation button at the right edge of a tweakpane row
// (`binding.element` / `button.element`). Returns the button so the caller can
// toggle its `.active` tint. Pads the value cell so the slider doesn't underlap.
function injectModButton(
  row: HTMLElement,
  onClick: (btn: HTMLButtonElement) => void,
): HTMLButtonElement {
  ensureModCss();
  row.style.position = "relative";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "mod-btn";
  btn.textContent = "∿";
  btn.title = "audio modulation";
  // Stop the row's own pointer handlers (slider drag) from seeing this; the
  // popover's document capture listener still fires (capture precedes target).
  btn.addEventListener("pointerdown", (e) => e.stopPropagation());
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    e.preventDefault();
    onClick(btn);
  });
  row.appendChild(btn);
  const valueCell = row.querySelector<HTMLElement>(".tp-lblv_v");
  if (valueCell) valueCell.style.paddingRight = "20px";
  return btn;
}

let cssInjected = false;
function ensureModCss(): void {
  if (cssInjected) return;
  cssInjected = true;
  const style = document.createElement("style");
  style.textContent = `
.mod-btn {
  position: absolute; right: 2px; top: 50%; transform: translateY(-50%);
  width: 16px; height: 16px; padding: 0; line-height: 14px;
  font-size: 12px; text-align: center;
  background: #2a2a2a; color: #888; border: 1px solid #444; border-radius: 3px;
  cursor: pointer; z-index: 2;
}
.mod-btn:hover { color: #ccc; border-color: #666; }
.mod-btn.active { color: #1ea7e1; border-color: #1ea7e1; }
/* A param row with a modulation hooked up: accent-blue slider fill bar. Label
   and value text keep their normal color (only .tp-sldv_k::before is tinted,
   not --in-fg which also colors the number text). */
.modulated .tp-sldv_k::before { background-color: #1ea7e1; }
/* Trigger action buttons: gray outline, no fill. Flash filled on fire and fade
   back (the envelope) via the trig-flash animation. Selector is .tp-btnv_b.trig-btn
   (specificity beats tweakpane's .tp-btnv_b) so we avoid !important — an
   !important author rule would outrank the animation and block the fill. */
.tp-btnv_b.trig-btn {
  background-color: transparent;
  border: 1px solid rgba(204, 204, 204, 0.4);
  box-shadow: none;
}
.tp-btnv_b.trig-btn.trig-flash {
  animation: trig-flash-kf 0.45s ease-out;
}
@keyframes trig-flash-kf {
  from { background-color: #fff; border-color: #fff; color: #111; }
  to   { background-color: transparent; border-color: rgba(204, 204, 204, 0.4); }
}
`;
  document.head.appendChild(style);
}

function makeWidget(
  folder: FolderApi,
  proxy: { value: ParamValue },
  schema: ParamSchema,
) {
  if (schema.kind === "boolean") {
    return folder.addBinding(proxy, "value", { label: schema.label });
  }
  if (schema.kind === "discrete") {
    const labels = schema.optionLabels ?? schema.options.map(String);
    return folder.addBinding(proxy, "value", {
      label: schema.label,
      options: Object.fromEntries(schema.options.map((v, i) => [labels[i], v])),
    });
  }
  return folder.addBinding(proxy, "value", {
    label: schema.label,
    min: schema.min,
    max: schema.max,
    step: schema.step,
  });
}
