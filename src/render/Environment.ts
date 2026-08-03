import { Color, EquirectangularReflectionMapping } from "three";
import type { Scene, Texture } from "three";
import { RGBELoader } from "three/examples/jsm/loaders/RGBELoader.js";
import type { FolderApi } from "tweakpane";
import type { ParamStore } from "../params/ParamStore";
import type { Modulator } from "../params/Modulator";
import { bindParam, type ParamProxyRegistry } from "../params/bindParam";
import { HDRI_FILES } from "../params/envSchemas";
import { setLitLook } from "./components/litMaterial";

// Page-lifetime texture cache: HMR rebuilds a fresh App (and Environment),
// but switching HDRIs must never re-fetch a multi-MB file.
const hdriCache = new Map<string, Promise<Texture>>();

function loadHdri(file: string): Promise<Texture> {
  let p = hdriCache.get(file);
  if (!p) {
    p = new RGBELoader().loadAsync(`/hdri/${file}`).then((tex) => {
      tex.mapping = EquirectangularReflectionMapping;
      return tex;
    });
    hdriCache.set(file, p);
  }
  return p;
}

// Owns scene.background + scene.environment. Color mode: flat backdrop, no
// IBL — the analytic lights are all the lighting. HDRI mode: the equirect
// lights every lit material via scene.environment and draws as the backdrop.
export class Environment {
  private unsub: () => void;
  private bgColor = new Color();
  // Monotonic token: only the latest applyMode() may install its async-loaded
  // texture, so rapid dropdown flips can't resolve out of order.
  private generation = 0;

  constructor(private scene: Scene, private store: ParamStore) {
    this.unsub = store.subscribe((key) => {
      if (!key.startsWith("env.")) return;
      // Levels are hot uniform-ish writes (modulatable per-frame); mode/color/
      // hdri go through the async texture path.
      if (key === "env.mode" || key === "env.color" || key === "env.hdri") this.applyMode();
      else this.applyLevels();
    });
    this.applyLevels();
    this.applyMode();
  }

  private applyLevels(): void {
    const s = this.store;
    setLitLook(s.get("env.roughness") as number, s.get("env.metalness") as number);
    this.scene.environmentIntensity = s.get("env.intensity") as number;
    this.scene.backgroundIntensity = s.get("env.bgIntensity") as number;
    this.scene.backgroundBlurriness = s.get("env.bgBlur") as number;
  }

  private applyMode(): void {
    const s = this.store;
    const gen = ++this.generation;
    if ((s.get("env.mode") as number) === 0) {
      this.scene.background = this.bgColor.setHex(s.get("env.color") as number);
      this.scene.environment = null;
      return;
    }
    const file = HDRI_FILES[s.get("env.hdri") as number] ?? HDRI_FILES[0];
    void loadHdri(file).then((tex) => {
      if (gen !== this.generation) return;
      this.scene.background = tex;
      this.scene.environment = tex;
    });
  }

  bindUI(folder: FolderApi, modulator: Modulator, proxies: ParamProxyRegistry): void {
    for (const key of [
      "env.mode",
      "env.color",
      "env.hdri",
      "env.intensity",
      "env.bgIntensity",
      "env.bgBlur",
      "env.roughness",
      "env.metalness",
    ]) {
      const schema = this.store.schemaFor(key);
      if (!schema) throw new Error(`Environment.bindUI: schema ${key} missing`);
      bindParam(folder, this.store, modulator, schema, proxies);
    }
  }

  dispose(): void {
    this.unsub();
  }
}
