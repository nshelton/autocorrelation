import { PointLight as ThreePointLight, Mesh, MeshBasicMaterial, SphereGeometry } from "three";
import type { Component, ComponentDeps } from "./Component";

// Gizmo sphere radius — a visible marker for where the (otherwise invisible)
// light sits.
const MARKER_RADIUS = 0.07;
// Intensity at which the marker reaches full brightness; higher intensities
// just clamp at 1 rather than blowing the color out.
const MARKER_FULL_BRIGHTNESS_INTENSITY = 10;

// Module-lifetime singleton, added to the scene once and never removed —
// r170's WebGPU renderer doesn't reliably rebuild already-compiled material
// pipelines (Spawner, BoxView, ...) when a light is added/removed at runtime,
// so a light toggled via scene.add()/remove() on enable/disable would go
// invisible on already-live meshes. Matches App.ts's directional/ambient
// lights: the Scenes-panel checkbox drives intensity to 0 instead.
let sharedLight: ThreePointLight | null = null;

export class PointLightScene implements Component {
  static id = "pointLight";
  static label = "Point Light";
  static paramOpts = {
    intensity: { min: 0, max: 50, step: 0.1 },
    distance: { min: 0, max: 20, step: 0.1 },
    decay: { min: 0, max: 4, step: 0.1 },
    x: { min: -10, max: 10, step: 0.1 },
    y: { min: -10, max: 10, step: 0.1 },
    z: { min: -10, max: 10, step: 0.1 },
  };
  static paramDefaults = {
    color: 0xffffff,
    intensity: 8,
    distance: 0,
    decay: 2,
    x: 0,
    y: 2,
    z: 0,
  };
  static paramKinds = {
    color: "color" as const,
  };

  private params: Record<string, number>;
  private light: ThreePointLight;
  private scene: ComponentDeps["scene"];
  // Unlit, so it stays visible however dim/bright the actual light is — a
  // gizmo, not a scene fixture. Ordinary component lifecycle (unlike the
  // light itself): it doesn't feed any other material's lighting model, so
  // add/remove on enable/disable is safe here.
  private marker: Mesh;

  constructor(deps: ComponentDeps, params: Record<string, number>) {
    this.params = params;
    this.scene = deps.scene;
    if (!sharedLight) sharedLight = new ThreePointLight();
    this.light = sharedLight;
    deps.scene.add(this.light); // no-op if already attached
    this.marker = new Mesh(new SphereGeometry(MARKER_RADIUS, 12, 8), new MeshBasicMaterial());
    deps.scene.add(this.marker);
    this.apply();
  }

  private apply(): void {
    const p = this.params;
    this.light.color.setHex(p.color);
    this.light.intensity = p.intensity;
    this.light.distance = p.distance;
    this.light.decay = p.decay;
    this.light.position.set(p.x, p.y, p.z);
    this.marker.position.set(p.x, p.y, p.z);
    const brightness = Math.min(p.intensity / MARKER_FULL_BRIGHTNESS_INTENSITY, 1);
    (this.marker.material as MeshBasicMaterial).color.setHex(p.color).multiplyScalar(brightness);
  }

  update(): void {
    this.apply();
  }

  dispose(): void {
    // The light stays in the scene; goes dark instead of scene.remove() — see
    // the comment on sharedLight above. The marker has no such constraint.
    this.light.intensity = 0;
    this.scene.remove(this.marker);
    this.marker.geometry.dispose();
    (this.marker.material as MeshBasicMaterial).dispose();
  }
}
