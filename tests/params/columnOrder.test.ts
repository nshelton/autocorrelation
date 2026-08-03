import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ParamStore } from "../../src/params/ParamStore";
import { ParamPanel } from "../../src/params/ParamPanel";
import { Modulator } from "../../src/params/Modulator";
import { FeatureStore } from "../../src/store/FeatureStore";

function makePanel(container: HTMLElement) {
  const store = new ParamStore();
  const modulator = new Modulator(store, new FeatureStore());
  return new ParamPanel(store, modulator, container);
}

// [[panelId, ...], ...] per column, mirroring the persisted layout shape.
const layout = (container: HTMLElement) =>
  [...container.querySelectorAll<HTMLElement>(".pp-col")].map((col) =>
    [...col.children].map((c) => (c as HTMLElement).dataset.panelId),
  );

describe("panel layout", () => {
  let container: HTMLElement;

  beforeEach(() => {
    localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => container.remove());

  it("puts each fixed section in its own column by default", () => {
    const panel = makePanel(container);
    expect(layout(container)).toEqual([
      ["system"],
      ["signals"],
      ["analysis"],
      ["scenes"],
      ["camera"],
      ["environment"],
      ["post"],
      [], // trailing drop target
    ]);
    panel.dispose();
  });

  it("uses folder titles as handles, adding no extra row", () => {
    const panel = makePanel(container);
    expect(container.querySelector(".col-grip")).toBeNull();
    expect(container.querySelector(".pp-panel .tp-fldv_b")).not.toBeNull();
    panel.dispose();
  });

  it("keeps exactly one empty column at the end", () => {
    const panel = makePanel(container);
    const cols = layout(container);
    expect(cols.filter((c) => c.length === 0)).toHaveLength(1);
    expect(cols[cols.length - 1]).toEqual([]);
    panel.dispose();
  });

  it("gives a newly enabled scene a column of its own", () => {
    const panel = makePanel(container);
    const scene = panel.addScenePanel("spawner");
    const cols = layout(container);
    expect(cols).toContainEqual(["scene:spawner"]);
    scene.dispose();
    expect(layout(container)).not.toContainEqual(["scene:spawner"]);
    panel.dispose();
  });

  it("restores a saved layout, stacking several panels in one column", () => {
    localStorage.setItem(
      "paramPanel.layout.v2",
      JSON.stringify([
        ["system", "scenes", "scene:spawner"],
        ["analysis", "signals"],
        ["camera", "post"],
      ]),
    );
    const panel = makePanel(container);
    expect(layout(container)).toEqual([
      ["system", "scenes"],
      ["analysis", "signals"],
      ["camera", "post"],
      // Not in the saved layout, so it gets a fresh column of its own.
      ["environment"],
      [],
    ]);

    // A scene switched on later joins the column it was left in, in position.
    const scene = panel.addScenePanel("spawner");
    expect(layout(container)[0]).toEqual(["system", "scenes", "scene:spawner"]);
    scene.dispose();
    panel.dispose();
  });

  // happy-dom reports zero-size rects, so the geometry is degenerate — what
  // these cover is the wiring: slop threshold, live move, persistence.
  function press(el: Element, x: number, y = 0) {
    el.dispatchEvent(new MouseEvent("pointerdown", { bubbles: true, clientX: x, clientY: y }));
  }
  const moveTo = (x: number, y = 0) =>
    window.dispatchEvent(new MouseEvent("pointermove", { bubbles: true, clientX: x, clientY: y }));
  const release = () => window.dispatchEvent(new MouseEvent("pointerup", { bubbles: true }));

  it("a click under the slop threshold neither moves nor persists", () => {
    const panel = makePanel(container);
    const before = layout(container);
    press(container.querySelector(".pp-panel .tp-fldv_b")!, 0);
    moveTo(2);
    release();
    expect(layout(container)).toEqual(before);
    expect(localStorage.getItem("paramPanel.layout.v2")).toBeNull();
    panel.dispose();
  });

  it("dragging a title past the threshold moves the panel and persists", () => {
    const panel = makePanel(container);
    const first = container.querySelector<HTMLElement>(".pp-panel")!;
    expect(first.dataset.panelId).toBe("system");

    press(first.querySelector(".tp-fldv_b")!, 0);
    moveTo(400);
    release();

    // Which column it lands in is geometry, which happy-dom can't give us; what
    // this pins down is that the drag ran and persisted the resulting DOM.
    const saved = JSON.parse(localStorage.getItem("paramPanel.layout.v2") ?? "[]");
    expect(saved).toEqual(layout(container).filter((c) => c.length > 0));
    panel.dispose();
  });
});
