import type { FolderApi } from "tweakpane";
import type { Component, ComponentClass, ComponentDeps } from "./Component";
import { Modulator } from "../../params/Modulator";
import { bindParam, bindTrigger, type ParamProxyRegistry } from "../../params/bindParam";

// Per-component runtime state. paramsBag is null for components that don't
// declare static paramDefaults. The bag is allocated once per page lifetime
// and kept across toggle cycles so tweakpane bindings (which hold a
// *reference*) stay live.
interface Slot {
  cls: ComponentClass;
  paramsBag: Record<string, number> | null;
  instance: Component | null;
  enabledKey: string;
}

export class ComponentManager {
  private slots: Slot[] = [];
  private storeUnsub: (() => void) | null = null;
  // Mix of real tweakpane folders/bindings and our store-subscription
  // unsubscribers. Both expose .dispose(), so we treat them uniformly.
  private paneTeardowns: Array<{ dispose: () => void }> = [];

  constructor(
    private deps: ComponentDeps,
    private classes: readonly ComponentClass[],
  ) {}

  // Register schemas, allocate bags, and construct components whose enabled
  // flag is true. Must be called once before update() or bindUI().
  start(): void {
    const { paramStore } = this.deps;

    for (const cls of this.classes) {
      const enabledKey = `components.${cls.id}.enabled`;
      paramStore.register({
        key: enabledKey,
        label: `${cls.label} enabled`,
        kind: "boolean",
        default: true,
        reconfig: false,
      });

      const paramsBag = this.allocateBag(cls);
      const slot: Slot = { cls, paramsBag, instance: null, enabledKey };
      this.slots.push(slot);

      if (paramStore.get(enabledKey) === true) {
        this.construct(slot);
      }
    }

    this.storeUnsub = paramStore.subscribe((key, value) => {
      const slot = this.slots.find((s) => s.enabledKey === key);
      if (slot) {
        if (value === true && !slot.instance) this.construct(slot);
        else if (value === false && slot.instance) this.destroy(slot);
        return;
      }
      // Mirror external param changes into the stable bag so the live
      // instance picks them up next frame (it reads from the same object).
      for (const s of this.slots) {
        if (!s.paramsBag) continue;
        const prefix = `${s.cls.paramPrefix ?? s.cls.id}.`;
        if (!key.startsWith(prefix)) continue;
        const localKey = key.slice(prefix.length);
        if (localKey in s.paramsBag && typeof value === "number") {
          s.paramsBag[localKey] = value;
        }
      }
    });
  }

  // Add one tweakpane folder per component: enable checkbox first, then
  // (if applicable) one slider per param bound to the stable bag. Must
  // be called AFTER start() — slots are populated there.
  bindUI(parent: FolderApi, modulator: Modulator): void {
    const { paramStore } = this.deps;

    for (const slot of this.slots) {
      const enabled = paramStore.get(slot.enabledKey) === true;
      const folder = parent.addFolder({ title: slot.cls.label, expanded: enabled });
      this.paneTeardowns.push(folder);

      const enabledProxy: { enabled: boolean } = { enabled };
      const enabledBinding = folder.addBinding(enabledProxy, "enabled", {
        label: "enabled",
      });
      enabledBinding.on("change", (e: { value: boolean }) => {
        paramStore.set(slot.enabledKey, e.value);
      });
      // Mirror external enable changes (e.g. from `Reset to defaults`)
      // back into the checkbox UI, and collapse the folder when a component
      // is disabled so disabled scenes don't clutter the panel.
      const unsub = paramStore.subscribe((key, value) => {
        if (key === slot.enabledKey && typeof value === "boolean") {
          if (enabledProxy.enabled !== value) {
            enabledProxy.enabled = value;
            folder.refresh();
          }
          folder.expanded = value;
        }
      });
      this.paneTeardowns.push({ dispose: unsub });

      // Per-component reset button. Only touches THIS component's params;
      // does not reset the enabled flag or any other component's params.
      // Setting each key through paramStore.set() flows through the mirror
      // subscriber (which updates slot.paramsBag), but tweakpane caches
      // the displayed value, so we folder.refresh() at the end to re-pull.
      if (slot.paramsBag && slot.cls.paramDefaults) {
        const defaults = slot.cls.paramDefaults;
        const prefix = slot.cls.paramPrefix ?? slot.cls.id;
        const resetBtn = folder.addButton({ title: "Reset to defaults" });
        resetBtn.on("click", () => {
          for (const [k, def] of Object.entries(defaults)) {
            paramStore.set(`${prefix}.${k}`, def);
          }
          folder.refresh();
        });
      }

      if (slot.cls.paramButtons) {
        const btnPrefix = slot.cls.paramPrefix ?? slot.cls.id;
        for (const btn of slot.cls.paramButtons) {
          const b = folder.addButton({ title: btn.title });
          // Same action is reachable from audio: register it under a stable key
          // and inject the trigger UI. bindTrigger returns a flash() that pulses
          // the button white so audio-driven fires are visible.
          const triggerKey = `${btnPrefix}.button.${btn.title.replace(/\s+/g, "")}`;
          const flash = bindTrigger(b, modulator, triggerKey);
          const fire = () => {
            flash();
            btn.onClick(paramStore);
            folder.refresh();
          };
          b.on("click", fire);
          modulator.registerTriggerCallback(triggerKey, fire);
        }
      }

      if (!slot.paramsBag) continue;
      // Slider/dropdown bindings are not pushed into paneTeardowns
      // explicitly; tweakpane's folder.dispose() cascades to child
      // bindings and their change listeners, and the folder IS in
      // paneTeardowns.
      const allKeys = new Set<string>([
        ...Object.keys(slot.cls.paramOpts ?? {}),
        ...Object.keys(slot.cls.paramDefaults ?? {}),
      ]);
      // Collect per-key "re-pull proxy from store" callbacks so programmatic
      // store writes (e.g. the SH tween behind "Randomize SH", or "Reset to
      // defaults") snap the sliders to the new values instead of going stale.
      const proxies: ParamProxyRegistry = new Map();
      for (const k of allKeys) {
        const fullKey = `${slot.cls.paramPrefix ?? slot.cls.id}.${k}`;
        const schema = paramStore.schemaFor(fullKey);
        if (!schema) continue;
        bindParam(folder, paramStore, modulator, schema, proxies);
      }
      // Gated on source==='user' so per-frame modulator notifies don't jitter
      // the UI; matches ParamPanel's dsp.* slider refresh.
      const refreshUnsub = paramStore.subscribe((key, _value, source) => {
        if (source !== "user") return;
        const refresh = proxies.get(key);
        if (!refresh) return;
        refresh();
        folder.refresh();
      });
      this.paneTeardowns.push({ dispose: refreshUnsub });
    }
  }

  // No-op after dispose() (slots cleared); safe to call from a paused
  // RAF loop without guarding.
  update(): void {
    for (const slot of this.slots) {
      slot.instance?.update();
    }
  }

  dispose(): void {
    this.storeUnsub?.();
    this.storeUnsub = null;
    for (const slot of this.slots) {
      if (slot.instance) {
        slot.instance.dispose();
        slot.instance = null;
      }
    }
    for (const t of this.paneTeardowns) {
      try {
        t.dispose();
      } catch {
        // Some entries are subscription unsubs, others are real tweakpane
        // folders. Both expose .dispose(); neither should throw.
      }
    }
    this.paneTeardowns = [];
    this.slots = [];
  }

  // Allocate (or null) the params bag. Must be called BEFORE construct() —
  // the bag is the second constructor arg.
  private allocateBag(cls: ComponentClass): Record<string, number> | null {
    if (!cls.paramDefaults) return null;
    const { paramStore } = this.deps;
    const bag: Record<string, number> = {};
    const prefix = cls.paramPrefix ?? cls.id;
    for (const [k, def] of Object.entries(cls.paramDefaults)) {
      const fullKey = `${prefix}.${k}`;
      const kind = cls.paramKinds?.[k] ?? "continuous";
      if (kind === "discrete") {
        const options = cls.paramDiscreteOptions?.[k];
        if (!options) {
          throw new Error(
            `ComponentManager: ${cls.id}.${k} declared discrete but paramDiscreteOptions[${k}] is missing`,
          );
        }
        paramStore.register({
          key: fullKey,
          label: k,
          kind: "discrete",
          reconfig: false,
          default: def,
          options,
          optionLabels: cls.paramDiscreteLabels?.[k],
        });
      } else {
        const opts = cls.paramOpts?.[k];
        paramStore.register({
          key: fullKey,
          label: k,
          kind: "continuous",
          reconfig: false,
          default: def,
          min: opts?.min ?? 0,
          max: opts?.max ?? 1,
          step: opts?.step ?? 0.01,
        });
      }
      const v = paramStore.get(fullKey);
      bag[k] = typeof v === "number" ? v : def;
    }
    return bag;
  }

  private construct(slot: Slot): void {
    slot.instance = slot.paramsBag
      ? new slot.cls(this.deps, slot.paramsBag)
      : new slot.cls(this.deps);
  }

  private destroy(slot: Slot): void {
    slot.instance?.dispose();
    slot.instance = null;
  }
}
