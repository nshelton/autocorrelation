import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import { ParticleView } from "./ParticleView";
import type { ComponentClass } from "./Component";

// Order = render order in the scene (insertion order). Also drives the
// order of folders in the tweakpane panel. Add a new component: import it
// here and append to this array.
export const COMPONENTS: readonly ComponentClass[] = [
  DebugView,
  BoxView as unknown as ComponentClass,
  ParticleView as unknown as ComponentClass,
];
