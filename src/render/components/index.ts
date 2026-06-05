import { DebugView } from "../debug/DebugView";
import { BoxView } from "./BoxView";
import { ParticleView } from "./ParticleView";
import { OrbitalCloud } from "./OrbitalCloud";
import { OrbitalVolume } from "./OrbitalVolume";
import { Spawner } from "./Spawner";
import { Grid } from "./Grid";
import type { ComponentClass } from "./Component";

// Order = render order in the scene (insertion order). Also drives the
// order of folders in the tweakpane panel.
export const COMPONENTS: readonly ComponentClass[] = [
  DebugView,
  BoxView as unknown as ComponentClass,
  ParticleView as unknown as ComponentClass,
  OrbitalCloud as unknown as ComponentClass,
  OrbitalVolume as unknown as ComponentClass,
  Spawner as unknown as ComponentClass,
  Grid as unknown as ComponentClass,
];
