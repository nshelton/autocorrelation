import { Scene, Object3D } from "three";
import type { FeatureStore } from "../store/FeatureStore";
import type { ParamStore } from "../params/ParamStore";

export interface BoxViewDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
}

export class BoxView {
  private scene: Scene;
  private store: FeatureStore;
  private paramStore: ParamStore;

  constructor(private deps: BoxViewDeps) {
    this.scene = deps.scene;
    this.store = deps.store;
    this.paramStore = deps.paramStore;

    
  }

  update(): void {}
  dispose(): void {}
}
