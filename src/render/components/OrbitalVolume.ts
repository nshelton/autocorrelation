import type { Component, ComponentDeps } from "./Component";

// Discrete option sets (kept module-level so the static config below can
// reference them without recomputing).
const VOLUME_STEPS_OPTIONS = [16, 32, 48, 64, 96, 128] as const;
const SHADOW_STEPS_OPTIONS = [0, 4, 8, 16, 24] as const;

function buildParamOpts(): Record<string, { min: number; max: number; step?: number }> {
  return {
    volumeSteps:  { min: 0, max: 0, step: 0 },  // discrete; ignored
    shadowSteps:  { min: 0, max: 0, step: 0 },  // discrete; ignored
    density:      { min: 0.1, max: 500, step: 0.1 },
    boundsRadius: { min: 1, max: 20, step: 0.1 },
  };
}

function buildParamDefaults(): Record<string, number> {
  return {
    volumeSteps:  48,
    shadowSteps:  8,
    density:      50,
    boundsRadius: 8,
  };
}

export class OrbitalVolume implements Component {
  static id = "orbitalVolume";
  static label = "Orbital Volume";
  static paramPrefix = "orbitalVolume";
  static paramOpts = buildParamOpts();
  static paramDefaults = buildParamDefaults();
  static paramKinds = {
    volumeSteps: "discrete" as const,
    shadowSteps: "discrete" as const,
  };
  static paramDiscreteOptions = {
    volumeSteps: VOLUME_STEPS_OPTIONS as unknown as number[],
    shadowSteps: SHADOW_STEPS_OPTIONS as unknown as number[],
  };

  constructor(_deps: ComponentDeps, _params: Record<string, number>) {
    // wired up in Task 4
  }

  update(): void {}
  dispose(): void {}
}
