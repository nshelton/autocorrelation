import type { ParamSchema } from "./ParamStore";

export const analysisSchemas: ParamSchema[] = [
  {
    key: "dsp.windowSize",
    label: "Window size",
    kind: "discrete",
    options: [512, 1024, 2048, 4096],
    default: 2048,
    reconfig: true,
  },
  {
    key: "dsp.rmsHistoryLen",
    label: "RMS history length",
    kind: "discrete",
    options: [256, 512, 1024],
    default: 512,
    reconfig: true,
  },
  {
    key: "dsp.hopSize",
    label: "Hop size",
    kind: "discrete",
    options: [256, 512, 1024, 2048],
    default: 1024,
    reconfig: false,
  },
  {
    key: "dsp.smoothingTauSecs",
    label: "Smoothing τ (s)",
    kind: "continuous",
    min: 0.01,
    max: 1.0,
    step: 0.005,
    default: 0.0956,
    reconfig: false,
  },
  {
    key: "dsp.dbFloor",
    label: "Spectrum dB floor",
    kind: "continuous",
    min: -120,
    max: -40,
    step: 1,
    default: -100,
    reconfig: false,
  },
];
