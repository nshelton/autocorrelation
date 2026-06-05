import type { FolderApi } from "tweakpane";
import { ParamStore, type ParamSchema, type ParamValue } from "./ParamStore";
import { Modulator, MOD_SOURCE_KEYS } from "./Modulator";

// Sentinel value for "no modulation" in the source dropdown.
const NONE = "none";

// Optional map a caller can pass to collect "re-pull from store into proxy"
// callbacks per param key. Only ParamPanel uses this — see Task 6. Other
// call sites omit it (matches current behavior: their sliders don't
// visually refresh on external writes like Reset).
export type ParamProxyRegistry = Map<string, () => void>;

// Wraps the recurring `folder.addBinding(proxy, ...).on("change", store.set)`
// pattern. For continuous, non-dsp.* schemas, also appends a collapsed
// `↳ mod` sub-folder with a source dropdown + depth slider that drive
// Modulator.setBinding.
//
// The visible slider/dropdown/checkbox is bound to a LOCAL proxy object
// owned by this helper's closure. Tweakpane writes the user's drag into
// the proxy and fires `change`; we forward to store.set.
//
// External writes to the store (e.g. Reset to defaults) do NOT auto-refresh
// the proxy by default. Callers that need that — currently only ParamPanel
// for dsp.* sliders — pass a `ParamProxyRegistry` Map and bindParam
// registers a refresh callback for the key. The caller then calls the
// callback + `pane.refresh()` from its own subscriber.
//
// Modulator `notify()` calls DO NOT touch the proxy — see ParamStore.notify
// docs and Task 3 — so the slider stays anchored to the base value when
// modulation is active.
//
// Modulator subscription is added below; it lives until the Modulator is
// disposed (which happens in App.dispose, before panel teardown). No HMR
// leak because the modulator is recreated each cycle.
export function bindParam(
  folder: FolderApi,
  store: ParamStore,
  modulator: Modulator,
  schema: ParamSchema,
  refreshRegistry?: ParamProxyRegistry,
): void {
  const proxy: { value: ParamValue } = { value: store.get(schema.key) };
  const binding = makeWidget(folder, proxy, schema);
  // True while the user is mid-drag on this widget — gates the live modulated
  // refresh below so it doesn't fight the drag. `last` is true on the final
  // change of an interaction (and on single click/type), false during a drag.
  let interacting = false;
  binding.on("change", (e: { value: ParamValue; last: boolean }) => {
    interacting = !e.last;
    store.set(schema.key, e.value);
  });

  refreshRegistry?.set(schema.key, () => {
    proxy.value = store.get(schema.key);
  });

  const modulatable =
    schema.kind === "continuous" && !schema.key.startsWith("dsp.");
  if (!modulatable) return;

  const sub = folder.addFolder({ title: "↳ mod", expanded: false });

  const existing = modulator.getBinding(schema.key);
  const modProxy = {
    source: existing?.source ?? NONE,
    depth: existing?.depth ?? 0,
    power: existing?.power ?? 1,
    smoothing: existing?.smoothing ?? 0,
  };

  const sourceBinding = sub.addBinding(modProxy, "source", {
    label: "source",
    options: sourceDropdownOptions(),
  });
  const depthBinding = sub.addBinding(modProxy, "depth", {
    label: "depth",
    min: 0,
    max: 1,
    step: 0.01,
  });
  const powerBinding = sub.addBinding(modProxy, "power", {
    label: "power",
    min: 0.1,
    max: 10,
    step: 0.1,
  });
  const smoothingBinding = sub.addBinding(modProxy, "smoothing", {
    label: "smoothing",
    min: 0,
    max: 1,
    step: 0.01,
  });

  const writeBinding = () => {
    if (modProxy.source === NONE) {
      modulator.setBinding(schema.key, null);
    } else {
      modulator.setBinding(schema.key, {
        source: modProxy.source,
        depth: modProxy.depth,
        power: modProxy.power,
        smoothing: modProxy.smoothing,
      });
    }
  };
  sourceBinding.on("change", writeBinding);
  depthBinding.on("change", writeBinding);
  powerBinding.on("change", writeBinding);
  smoothingBinding.on("change", writeBinding);

  // Two-way sync from modulator changes (e.g. persisted-on-load).
  // Unsubscribe is intentionally discarded — Modulator.dispose() clears all
  // UI subs at HMR teardown, and the modulator is recreated per cycle.
  void modulator.subscribe((key) => {
    if (key !== schema.key) return;
    const current = modulator.getBinding(schema.key);
    modProxy.source = current?.source ?? NONE;
    modProxy.depth = current?.depth ?? 0;
    modProxy.power = current?.power ?? 1;
    modProxy.smoothing = current?.smoothing ?? 0;
    sub.refresh();
  });

  // Graph shows the smoothed + power-curved signal actually driving the
  // modulation (0..1), not the raw source.
  addSourceMonitor(sub, () => modulator.processedValue(schema.key));

  // Live-update the slider to the actual driven value while this param is
  // modulated. Routed through the Modulator (recreated + cleared per HMR) so
  // we don't leak a subscriber on the page-lifetime ParamStore. Suppressed
  // mid-drag so the user can still set the base value.
  void modulator.subscribeValue((key, value) => {
    if (key !== schema.key || interacting) return;
    proxy.value = value;
    binding.refresh();
  });
}

// Source dropdown options shared by mod + trigger UIs: "none" sentinel first,
// then every curated audio source.
function sourceDropdownOptions(): Record<string, string> {
  const o: Record<string, string> = { [NONE]: NONE };
  for (const k of MOD_SOURCE_KEYS) o[k] = k;
  return o;
}

// Live 0..1 graph driven by `read`. Bound to a getter that Tweakpane polls on
// its own interval to feed the scrolling graph — no per-frame wiring, nothing
// to dispose. Lives inside the (collapsed) mod/trig sub-folder, so it's only
// visible when that sub-folder is expanded.
function addSourceMonitor(folder: FolderApi, read: () => number): void {
  const target = {
    get value(): number {
      return read();
    },
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  folder.addBinding(target, "value", {
    label: "level",
    readonly: true,
    view: "graph",
    min: 0,
    max: 1,
    interval: 1000 / 30,
  } as any);
}

// Trigger UI for a pushbutton: a collapsed "↳ trig" sub-folder with a source
// dropdown, a threshold slider, and the live source graph. On a rising edge
// across threshold the Modulator fires the button's registered action.
export function bindTrigger(
  folder: FolderApi,
  modulator: Modulator,
  triggerKey: string,
): void {
  const sub = folder.addFolder({ title: "↳ trig", expanded: false });

  const existing = modulator.getTrigger(triggerKey);
  const trigProxy = {
    source: existing?.source ?? NONE,
    threshold: existing?.threshold ?? 0.5,
  };

  const sourceBinding = sub.addBinding(trigProxy, "source", {
    label: "source",
    options: sourceDropdownOptions(),
  });
  const thresholdBinding = sub.addBinding(trigProxy, "threshold", {
    label: "threshold",
    min: 0,
    max: 1,
    step: 0.01,
  });

  const writeTrigger = () => {
    if (trigProxy.source === NONE) {
      modulator.setTrigger(triggerKey, null);
    } else {
      modulator.setTrigger(triggerKey, {
        source: trigProxy.source,
        threshold: trigProxy.threshold,
      });
    }
  };
  sourceBinding.on("change", writeTrigger);
  thresholdBinding.on("change", writeTrigger);

  // Two-way sync (persisted-on-load / external clears); same disposal story as
  // the mod folder's subscriber.
  void modulator.subscribe((key) => {
    if (key !== triggerKey) return;
    const current = modulator.getTrigger(triggerKey);
    trigProxy.source = current?.source ?? NONE;
    trigProxy.threshold = current?.threshold ?? 0.5;
    sub.refresh();
  });

  // Triggers compare the raw source against the threshold, so the graph shows
  // the raw source value (no smoothing/power for triggers).
  addSourceMonitor(sub, () => modulator.readSource(trigProxy.source));
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
