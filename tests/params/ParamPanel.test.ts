import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ParamStore } from "../../src/params/ParamStore";
import { ParamPanel } from "../../src/params/ParamPanel";

describe("ParamPanel", () => {
  let container: HTMLElement;

  beforeEach(() => {
    localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    container.remove();
  });

  it("only renders dsp.* schemas, ignoring component-namespaced ones", () => {
    const store = new ParamStore();
    store.register({
      key: "dsp.alpha",
      label: "Alpha",
      kind: "continuous",
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.5,
      reconfig: false,
    });
    // Component schemas — must NOT appear in the DSP folder.
    store.register({
      key: "components.boxView.enabled",
      label: "Box View enabled",
      kind: "boolean",
      default: true,
      reconfig: false,
    });
    store.register({
      key: "boxView.pull",
      label: "pull",
      kind: "continuous",
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.3,
      reconfig: false,
    });

    const panel = new ParamPanel(store, container);

    // The DSP folder should have only 1 binding (dsp.alpha), not 3 (which would
    // include the two component-namespaced params). We verify by checking if
    // external store.set() calls to non-dsp keys don't crash (subscribe is
    // filtered). If the binding was there, it would try to update a non-existent
    // binding key on refresh.
    // A robust way: count how many inputs/controls are in the DOM under the pane.
    // Tweakpane creates a control per binding, so 1 dsp.alpha binding should
    // produce 1 control (plus the reset button). Component params should not be here.
    const controls = container.querySelectorAll("input, select");
    // 1 control for dsp.alpha (slider)
    expect(controls.length).toBe(1);

    panel.dispose();
  });

  it("ignores external sets to non-dsp keys (no pane refresh for them)", () => {
    const store = new ParamStore();
    store.register({
      key: "dsp.alpha",
      label: "Alpha",
      kind: "continuous",
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.5,
      reconfig: false,
    });
    store.register({
      key: "components.boxView.enabled",
      label: "Box View enabled",
      kind: "boolean",
      default: true,
      reconfig: false,
    });

    const panel = new ParamPanel(store, container);
    // Should not throw / should not corrupt bindings.
    store.set("components.boxView.enabled", false);
    expect(store.get("components.boxView.enabled")).toBe(false);

    panel.dispose();
  });
});
