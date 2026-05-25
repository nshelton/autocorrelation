import type { ParamSchema } from "./ParamStore";

export const postSchemas: ParamSchema[] = [
  // AO
  { key: "post.ao.enabled",      label: "AO on",            kind: "boolean",    default: true,  reconfig: false },
  { key: "post.ao.radius",       label: "AO radius",        kind: "continuous", min: 0.05, max: 2.0, step: 0.05, default: 0.25, reconfig: false },
  { key: "post.ao.intensity",    label: "AO intensity",     kind: "continuous", min: 0.0,  max: 2.0, step: 0.05, default: 1.0,  reconfig: false },

  // Bloom
  { key: "post.bloom.enabled",   label: "Bloom on",         kind: "boolean",    default: false, reconfig: false },
  { key: "post.bloom.strength",  label: "Bloom strength",   kind: "continuous", min: 0.0, max: 3.0,  step: 0.01, default: 0.5,  reconfig: false },
  { key: "post.bloom.radius",    label: "Bloom radius",     kind: "continuous", min: 0.0, max: 1.0,  step: 0.01, default: 0.4,  reconfig: false },
  { key: "post.bloom.threshold", label: "Bloom threshold",  kind: "continuous", min: 0.0, max: 2.0,  step: 0.01, default: 0.85, reconfig: false },

  // Tonemap
  { key: "post.tonemap.enabled", label: "Tonemap on",       kind: "boolean",    default: true,  reconfig: false },
  // Integer-coded: 0=None, 1=AgX, 2=ACES, 3=Neutral. Mapped to three's
  // ToneMapping constants inside TonemapEffect via TONEMAP_TABLE.
  { key: "post.tonemap.mode",    label: "Tonemap mode",     kind: "discrete",
    options: [0, 1, 2, 3], optionLabels: ["None", "AgX", "ACES", "Neutral"],
    default: 0, reconfig: false },
  { key: "post.tonemap.exposure",label: "Tonemap exposure", kind: "continuous", min: 0.0, max: 4.0, step: 0.01, default: 1.0, reconfig: false },

  // Lens: combined barrel distortion + chromatic aberration + vignette,
  // implemented as a single resample pass (LensNode uses convertToTexture
  // internally so it can re-sample the upstream chain at warped UVs).
  { key: "post.lens.enabled",        label: "Lens on",          kind: "boolean",    default: false, reconfig: false },
  { key: "post.lens.distortion",     label: "Lens distortion",  kind: "continuous", min: -0.5, max: 0.5,  step: 0.01,  default: 0.0,   reconfig: false },
  { key: "post.lens.chromatic",      label: "Lens chromatic",   kind: "continuous", min: 0.0,  max: 0.05, step: 0.001, default: 0.005, reconfig: false },
  { key: "post.lens.vignette",       label: "Lens vignette",    kind: "continuous", min: 0.0,  max: 1.0,  step: 0.01,  default: 0.5,   reconfig: false },
  { key: "post.lens.vignetteRadius", label: "Lens vig. radius", kind: "continuous", min: 0.0,  max: 1.0,  step: 0.01,  default: 0.4,   reconfig: false },
];
