import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Pane } from "tweakpane";
import { ParamStore, type ParamSchema } from "../../src/params/ParamStore";
import { Modulator } from "../../src/params/Modulator";
import { PresetStore, type PresetScope } from "../../src/params/PresetStore";
import { addPresetSection } from "../../src/params/PresetSection";
import { FeatureStore } from "../../src/store/FeatureStore";

const RATE: ParamSchema = {
  key: "spawner.rate",
  label: "rate",
  kind: "continuous",
  min: 0,
  max: 10,
  step: 0.1,
  default: 1,
  reconfig: false,
};

const SPAWNER: PresetScope = { id: "spawner", prefixes: ["spawner"] };

function buttonTitles(el: HTMLElement): string[] {
  return [...el.querySelectorAll("button")].map((b) => b.textContent?.trim() ?? "");
}

describe("addPresetSection", () => {
  let container: HTMLElement;
  let pane: Pane;

  beforeEach(() => {
    localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    pane = new Pane({ container });
  });

  afterEach(() => {
    pane.dispose();
    container.remove();
  });

  function setup() {
    const store = new ParamStore();
    store.register(RATE);
    const mod = new Modulator(store, new FeatureStore());
    const presets = new PresetStore(store, mod);
    const folder = pane.addFolder({ title: "Spawner" });
    const section = addPresetSection(folder, presets, SPAWNER);
    return { store, presets, folder, section };
  }

  it("renders the action buttons and no preset entries when empty", () => {
    const { folder } = setup();
    // "Spawner" is the parent folder's own title bar button.
    const titles = buttonTitles(folder.element);
    expect(titles).toEqual(["Spawner", "Presets", "save", "+ new preset", "delete"]);
  });

  it("lists saved presets and marks the current one", async () => {
    const { presets, folder } = setup();
    presets.save(SPAWNER, "fast");
    await Promise.resolve(); // rebuild is deferred to a microtask

    expect(buttonTitles(folder.element)).toContain("● fast");
  });

  it("loads a preset when its button is clicked", async () => {
    const { store, presets, folder } = setup();
    presets.save(SPAWNER, "one");
    store.set("spawner.rate", 8);
    presets.save(SPAWNER, "two");
    await Promise.resolve();

    const one = [...folder.element.querySelectorAll("button")].find(
      (b) => b.textContent?.trim() === "one",
    );
    one!.click();
    expect(store.get("spawner.rate")).toBe(1);
    expect(presets.current(SPAWNER)).toBe("one");
  });

  it("save overwrites the current preset in place", async () => {
    const { store, presets, folder } = setup();
    presets.save(SPAWNER, "one");
    store.set("spawner.rate", 5);
    await Promise.resolve();

    const save = [...folder.element.querySelectorAll("button")].find(
      (b) => b.textContent?.trim() === "save",
    );
    save!.click();
    expect(presets.list(SPAWNER)).toHaveLength(1);
    expect(presets.isDirty(SPAWNER)).toBe(false);
  });

  it("+ new preset opens a name dialog that saves on Enter", async () => {
    const { presets, folder } = setup();
    const add = [...folder.element.querySelectorAll("button")].find(
      (b) => b.textContent?.trim() === "+ new preset",
    );
    add!.click();

    const input = document.querySelector<HTMLInputElement>(".preset-prompt input");
    expect(input).not.toBeNull();
    input!.value = "swirly";
    input!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));

    expect(presets.list(SPAWNER).map((p) => p.name)).toEqual(["swirly"]);
    expect(document.querySelector(".preset-prompt")).toBeNull();
    await Promise.resolve();
    expect(buttonTitles(folder.element)).toContain("● swirly");
  });

  it("delete is disabled with no current preset and removes it otherwise", async () => {
    const { presets, folder } = setup();
    const del = () =>
      [...folder.element.querySelectorAll("button")].find(
        (b) => b.textContent?.trim() === "delete",
      )!;
    expect(del().disabled).toBe(true);

    presets.save(SPAWNER, "one");
    await Promise.resolve();
    del().click();
    expect(presets.list(SPAWNER)).toHaveLength(0);
  });
});
