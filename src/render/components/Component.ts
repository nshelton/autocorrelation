import type { Scene } from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { FeatureStore } from "../../store/FeatureStore";
import type { ParamStore } from "../../params/ParamStore";

// Shared dependencies passed to every component constructor. App builds this
// once at start and reuses it for every component instance.
export interface ComponentDeps {
  scene: Scene;
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
  // Required for components that dispatch WebGPU compute (e.g. OrbitalCloud).
  // Existing components ignore it.
  renderer: WebGPURenderer;
}

// A component is a class with update() + dispose(). update() takes no args;
// if a future component needs dt, App's RAF loop already tracks it and we add
// the argument then.
export interface Component {
  update(): void;
  dispose(): void;
}

// A component CLASS is the registry entry. Static metadata replaces a separate
// entry-wrapper struct. id/label are required; the param trio is optional and
// opts the component into ParamStore-backed live-tunable params (App owns a
// stable params bag and passes it as the second constructor arg).
export interface ComponentClass {
  new (deps: ComponentDeps, params?: Record<string, number>): Component;
  id: string;
  label: string;
  paramPrefix?: string;
  paramOpts?: Record<string, { min: number; max: number; step?: number }>;
  paramDefaults?: Record<string, number>;
  // Per-key kind override. Absent or "continuous" → continuous (uses paramOpts
  // min/max/step). "discrete" → uses paramDiscreteOptions for the value set.
  paramKinds?: Record<string, "continuous" | "discrete">;
  // For each key whose paramKinds entry is "discrete", the allowed values.
  // Must be present when paramKinds[key] === "discrete".
  paramDiscreteOptions?: Record<string, number[]>;
}
