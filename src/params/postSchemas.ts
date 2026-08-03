import type { ParamSchema } from "./ParamStore";

export const postSchemas: ParamSchema[] = [
  // AO
  { key: "post.ao.enabled",      label: "AO on",            kind: "boolean",    default: true,  reconfig: false },
  { key: "post.ao.radius",       label: "AO radius",        kind: "continuous", min: 0.05, max: 2.0, step: 0.05, default: 0.25, reconfig: false },
  { key: "post.ao.intensity",    label: "AO intensity",     kind: "continuous", min: 0.0,  max: 2.0, step: 0.05, default: 1.0,  reconfig: false },

  // DOF: 49-tap polygonal-aperture bokeh (local BokehNode), per-pixel jittered
  // so the sparse kernel dithers to noise instead of visible tap rings.
  // `focus` is view-space distance in world units (camera presets orbit ~4
  // out). Defocus is inverse-distance — blur = aperture·clamp(1 − focus/dist,
  // ±1) — so `aperture` is simply the far-field blur radius in UV units of
  // screen width: 0 = everything sharp, larger = narrower in-focus zone.
  // blades 0 = circular aperture. boost weights taps by luminance² so bright
  // HDR highlights resolve into the aperture shape.
  { key: "post.dof.enabled",  label: "DOF on",         kind: "boolean",    default: false, reconfig: false },
  // autoFocus keeps the world origin on the focal plane (tracks the camera
  // per frame); the focus slider is overridden while it's on.
  { key: "post.dof.autoFocus",label: "DOF focus origin", kind: "boolean",  default: false, reconfig: false },
  { key: "post.dof.focus",    label: "DOF focus",      kind: "continuous", min: 0.1, max: 20.0, step: 0.1,    default: 4.0,  reconfig: false },
  { key: "post.dof.aperture", label: "DOF aperture",   kind: "continuous", min: 0.0, max: 0.05, step: 0.0005, default: 0.01, reconfig: false },
  { key: "post.dof.blades",   label: "DOF blades",     kind: "discrete",
    options: [0, 3, 4, 5, 6, 7, 8], optionLabels: ["Circle", "3", "4", "5", "6", "7", "8"],
    default: 5, reconfig: false },
  { key: "post.dof.rotation", label: "DOF rotation",   kind: "continuous", min: 0.0, max: 3.14, step: 0.01,  default: 0.0,  reconfig: false },
  { key: "post.dof.boost",    label: "DOF highlights", kind: "continuous", min: 0.0, max: 4.0,  step: 0.05,  default: 1.0,  reconfig: false },

  // Bloom
  { key: "post.bloom.enabled",   label: "Bloom on",         kind: "boolean",    default: false, reconfig: false },
  { key: "post.bloom.strength",  label: "Bloom strength",   kind: "continuous", min: 0.0, max: 3.0,  step: 0.01, default: 0.5,  reconfig: false },
  { key: "post.bloom.radius",    label: "Bloom radius",     kind: "continuous", min: 0.0, max: 1.0,  step: 0.01, default: 0.4,  reconfig: false },
  // Threshold is in linear HDR units — with lights driven past 1 the
  // interesting cutoffs sit well above the old LDR-ish 2.0 cap.
  { key: "post.bloom.threshold", label: "Bloom threshold",  kind: "continuous", min: 0.0, max: 4.0,  step: 0.01, default: 0.85, reconfig: false },

  // Tonemap: exposure -> contrast (power about mid-grey) -> saturation in
  // linear HDR, then the selected curve — all in-chain, see TonemapEffect.
  { key: "post.tonemap.enabled", label: "Tonemap on",       kind: "boolean",    default: true,  reconfig: false },
  // Integer-coded: 0=None, 1=AgX, 2=ACES, 3=Neutral. Mapped to three's
  // ToneMapping constants inside TonemapEffect via TONEMAP_TABLE.
  { key: "post.tonemap.mode",    label: "Tonemap mode",     kind: "discrete",
    options: [0, 1, 2, 3], optionLabels: ["None", "AgX", "ACES", "Neutral"],
    default: 0, reconfig: false },
  { key: "post.tonemap.exposure",  label: "Tonemap exposure", kind: "continuous", min: 0.0, max: 4.0, step: 0.01, default: 1.0, reconfig: false },
  { key: "post.tonemap.contrast",  label: "Tonemap contrast", kind: "continuous", min: 0.5, max: 2.0, step: 0.01, default: 1.0, reconfig: false },
  { key: "post.tonemap.saturation",label: "Tonemap saturation", kind: "continuous", min: 0.0, max: 2.0, step: 0.01, default: 1.0, reconfig: false },

  // Lens: combined barrel distortion + chromatic aberration + vignette,
  // implemented as a single resample pass (LensNode uses convertToTexture
  // internally so it can re-sample the upstream chain at warped UVs).
  { key: "post.lens.enabled",        label: "Lens on",          kind: "boolean",    default: false, reconfig: false },
  { key: "post.lens.distortion",     label: "Lens distortion",  kind: "continuous", min: -0.5, max: 0.5,  step: 0.01,  default: 0.0,   reconfig: false },
  { key: "post.lens.chromatic",      label: "Lens chromatic",   kind: "continuous", min: 0.0,  max: 0.05, step: 0.001, default: 0.005, reconfig: false },
  { key: "post.lens.vignette",       label: "Lens vignette",    kind: "continuous", min: 0.0,  max: 1.0,  step: 0.01,  default: 0.5,   reconfig: false },
  { key: "post.lens.vignetteRadius", label: "Lens vig. radius", kind: "continuous", min: 0.0,  max: 1.0,  step: 0.01,  default: 0.4,   reconfig: false },
  { key: "post.lens.zoom",           label: "Lens zoom",        kind: "continuous", min: 0.0,  max: 0.5,  step: 0.01,  default: 0.0,   reconfig: false },

  // Grain: film grain via Interleaved Gradient Noise (Jimenez, COD:AW).
  // Visually approximates blue noise; per-frame `frameId` shift gives temporal motion.
  { key: "post.grain.enabled",  label: "Grain on",       kind: "boolean",    default: false, reconfig: false },
  { key: "post.grain.strength", label: "Grain strength", kind: "continuous", min: 0.0, max: 0.5, step: 0.005, default: 0.08, reconfig: false },
  { key: "post.grain.scale",    label: "Grain scale",    kind: "continuous", min: 0.1, max: 4.0, step: 0.05,  default: 1.0,  reconfig: false },
];
