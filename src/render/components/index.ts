import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import { ParticleView } from "./ParticleView";
import { OrbitalCloud } from "./OrbitalCloud";
import type { ComponentClass } from "./Component";

// Order = render order in the scene (insertion order). Also drives the
// order of folders in the tweakpane panel. Add a new component: import it
// here and append to this array.
//
// OrbitalVolume is temporarily NOT registered — the ray-march shader hangs
// the render thread on first frame. Investigating; re-add once fixed.
export const COMPONENTS: readonly ComponentClass[] = [
  DebugView,
  BoxView as unknown as ComponentClass,
  ParticleView as unknown as ComponentClass,
  OrbitalCloud as unknown as ComponentClass,
];
