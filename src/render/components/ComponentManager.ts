import type { Pane } from "tweakpane";
import type { Component, ComponentClass, ComponentDeps } from "./Component";

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
  // (if applicable) one slider per param bound to the stable bag.
  bindUI(pane: Pane): void {
    const { paramStore } = this.deps;

    for (const slot of this.slots) {
      const folder = pane.addFolder({ title: slot.cls.label });
      this.paneTeardowns.push(folder);

      const enabledProxy: { enabled: boolean } = {
        enabled: paramStore.get(slot.enabledKey) === true,
      };
      const enabledBinding = folder.addBinding(enabledProxy, "enabled", {
        label: "enabled",
      });
      enabledBinding.on("change", (e: { value: boolean }) => {
        paramStore.set(slot.enabledKey, e.value);
      });
      // Mirror external enable changes (e.g. from `Reset to defaults`)
      // back into the checkbox UI.
      const unsub = paramStore.subscribe((key, value) => {
        if (key === slot.enabledKey && typeof value === "boolean") {
          if (enabledProxy.enabled !== value) {
            enabledProxy.enabled = value;
            pane.refresh();
          }
        }
      });
      this.paneTeardowns.push({ dispose: unsub });

      if (!slot.paramsBag || !slot.cls.paramOpts) continue;
      for (const [k, opts] of Object.entries(slot.cls.paramOpts)) {
        const fullKey = `${slot.cls.paramPrefix ?? slot.cls.id}.${k}`;
        const slider = folder.addBinding(slot.paramsBag, k, {
          ...opts,
          step: opts.step ?? (opts.max - opts.min) / 100,
        });
        slider.on("change", (e: { value: number }) => {
          paramStore.set(fullKey, e.value);
        });
      }
    }
  }

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
