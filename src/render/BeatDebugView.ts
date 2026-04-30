import { Scene } from "three";
import { BeatGridRenderer } from "./BeatGridRenderer";
import { BeatGridScrollingRenderer } from "./BeatGridScrollingRenderer";
import { BeatPulseSquares } from "./BeatPulseSquares";
import type { FeatureStore } from "../store/FeatureStore";

/**
 * Composes the three beat-debug renderers (static autocorr grid, scrolling
 * rms-history grid, 2×2 pulse squares). Sub-renderers are not yet
 * constructed — lazy-init refactor is deferred.
 */
export class BeatDebugView {
  private grid?: BeatGridRenderer;
  private gridScrolling?: BeatGridScrollingRenderer;
  private pulseSquares?: BeatPulseSquares;

  constructor(_scene: Scene, _store: FeatureStore) {}

  update(): void {
    this.grid?.update();
    this.gridScrolling?.update();
    this.pulseSquares?.update();
  }

  dispose(): void {
    this.grid?.dispose();
    this.gridScrolling?.dispose();
    this.pulseSquares?.dispose();
    this.grid = undefined;
    this.gridScrolling = undefined;
    this.pulseSquares = undefined;
  }
}
