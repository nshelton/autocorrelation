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

  const chips = () => [...container.querySelectorAll<HTMLElement>(".pset-chip")];
  const icon = (name: string) =>
    container.querySelector<HTMLButtonElement>(`.pset-icon[aria-label="${name}"]`)!;

  it("renders the three icon actions on one row and no chips when empty", () => {
    setup();
    const row = container.querySelector(".pset-actions")!;
    expect([...row.querySelectorAll("button")].map((b) => b.getAttribute("aria-label"))).toEqual([
      "save",
      "new preset",
      "delete",
    ]);
    expect(chips()).toHaveLength(0);
    expect(icon("delete").disabled).toBe(true);
  });

  it("shows one chip per preset, white for the loaded one", async () => {
    const { presets } = setup();
    presets.save(SPAWNER, "one");
    presets.save(SPAWNER, "two");
    await Promise.resolve();

    expect(chips().map((c) => c.textContent)).toEqual(["one", "two"]);
    expect(chips()[0].className).not.toContain("current");
    expect(chips()[1].className).toContain("current");
    expect(chips()[1].className).not.toContain("dirty");
  });

  it("marks the loaded chip dirty once params drift", async () => {
    const { store, presets, section } = setup();
    presets.save(SPAWNER, "one");
    await Promise.resolve();
    expect(chips()[0].className).toContain("current");

    store.set("spawner.rate", 5);
    // paint() is polled; drive it directly rather than waiting on the timer.
    await new Promise((r) => setTimeout(r, 250));
    expect(chips()[0].className).toContain("dirty");
    expect(chips()[0].className).not.toContain("current");
    section.dispose();
  });

  it("loads a preset when its chip is clicked", async () => {
    const { store, presets } = setup();
    presets.save(SPAWNER, "one");
    store.set("spawner.rate", 8);
    presets.save(SPAWNER, "two");
    await Promise.resolve();

    chips()[0].click();
    expect(store.get("spawner.rate")).toBe(1);
    expect(presets.current(SPAWNER)).toBe("one");
  });

  it("save overwrites the loaded preset in place", async () => {
    const { store, presets } = setup();
    presets.save(SPAWNER, "one");
    store.set("spawner.rate", 5);
    await Promise.resolve();

    icon("save").click();
    expect(presets.list(SPAWNER)).toHaveLength(1);
    expect(presets.isDirty(SPAWNER)).toBe(false);
  });

  it("the + icon opens a name dialog that saves on Enter", async () => {
    const { presets } = setup();
    icon("new preset").click();

    const input = document.querySelector<HTMLInputElement>(".preset-prompt input");
    expect(input).not.toBeNull();
    input!.value = "swirly";
    input!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));

    expect(presets.list(SPAWNER).map((p) => p.name)).toEqual(["swirly"]);
    expect(document.querySelector(".preset-prompt")).toBeNull();
    await Promise.resolve();
    expect(chips().map((c) => c.textContent)).toEqual(["swirly"]);
  });

  it("the trash icon deletes the loaded preset", async () => {
    const { presets } = setup();
    presets.save(SPAWNER, "one");
    await Promise.resolve();
    icon("delete").click();
    expect(presets.list(SPAWNER)).toHaveLength(0);
  });

  it("dispose stops the dirty poll and removes the DOM", async () => {
    const { presets, section } = setup();
    presets.save(SPAWNER, "one");
    await Promise.resolve();
    section.dispose();
    expect(container.querySelector(".pset-grid")).toBeNull();
    expect(container.querySelector(".pset-actions")).toBeNull();
  });
});
