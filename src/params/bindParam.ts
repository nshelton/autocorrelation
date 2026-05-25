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
  binding.on("change", (e: { value: ParamValue }) => store.set(schema.key, e.value));

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
  };

  const sourceOptions: Record<string, string> = { [NONE]: NONE };
  for (const k of MOD_SOURCE_KEYS) sourceOptions[k] = k;

  const sourceBinding = sub.addBinding(modProxy, "source", {
    label: "source",
    options: sourceOptions,
  });
  const depthBinding = sub.addBinding(modProxy, "depth", {
    label: "depth",
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
      });
    }
  };
  sourceBinding.on("change", writeBinding);
  depthBinding.on("change", writeBinding);

  // Two-way sync from modulator changes (e.g. persisted-on-load).
  modulator.subscribe((key) => {
    if (key !== schema.key) return;
    const current = modulator.getBinding(schema.key);
    modProxy.source = current?.source ?? NONE;
    modProxy.depth = current?.depth ?? 0;
    sub.refresh();
  });
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
