import type { ParamSchema } from "./ParamStore";

export const lightSchemas: ParamSchema[] = [
  { key: "light.directional.enabled", label: "Directional light", kind: "boolean", default: true, reconfig: false },
];
