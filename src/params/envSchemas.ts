import type { ParamSchema } from "./ParamStore";

// Bundled equirect HDRs under public/hdri/ (Poly Haven, CC0). The env.hdri
// dropdown indexes this list — drop a file in and add it here.
export const HDRI_FILES = [
  "studio_small_08_2k.hdr",
  "moonless_golf_2k.hdr",
  "venice_sunset_2k.hdr",
];

const HDRI_LABELS = HDRI_FILES.map((f) => f.replace(/_2k\.hdr$/, ""));

export const envSchemas: ParamSchema[] = [
  { key: "env.mode",  label: "mode", kind: "discrete",
    options: [0, 1], optionLabels: ["Color", "HDRI"], default: 0, reconfig: false },
  { key: "env.color", label: "bg color", kind: "color", default: 0x0a0a0a, reconfig: false },
  { key: "env.hdri",  label: "hdri", kind: "discrete",
    options: HDRI_FILES.map((_, i) => i), optionLabels: HDRI_LABELS, default: 0, reconfig: false },
  // scene.environmentIntensity — IBL contribution on the lit materials.
  { key: "env.intensity",   label: "light intensity", kind: "continuous", min: 0, max: 4, step: 0.01, default: 1, reconfig: false },
  // Background draw brightness/blur, independent of the lighting — a dim or
  // blurred backdrop with full-strength IBL is the usual look here.
  { key: "env.bgIntensity", label: "bg intensity",    kind: "continuous", min: 0, max: 2, step: 0.01, default: 1, reconfig: false },
  { key: "env.bgBlur",      label: "bg blur",         kind: "continuous", min: 0, max: 1, step: 0.01, default: 0, reconfig: false },
  // Shared look of every lit component material (see litMaterial.ts).
  { key: "env.roughness",   label: "roughness",       kind: "continuous", min: 0, max: 1, step: 0.01, default: 0.5, reconfig: false },
  { key: "env.metalness",   label: "metalness",       kind: "continuous", min: 0, max: 1, step: 0.01, default: 0, reconfig: false },
];
