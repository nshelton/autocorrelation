import { Scene } from "three";
import { BeatGridRenderer } from "./BeatGridRenderer";
import { BeatGridScrollingRenderer } from "./BeatGridScrollingRenderer";
import { BeatPulseSquares } from "./BeatPulseSquares";
import type { FeatureStore } from "../store/FeatureStore";

export interface BeatDebugSizes {
  rmsLen: number;
  onsetAcfLen: number;
  beatGridLen: number;
  beatPulsesLen: number;
  beatStateLen: number;
}

/**
 * Composes the three beat-debug renderers (static autocorr grid, scrolling
 * rms-history grid, 2×2 pulse squares). Reads `beatGrid` / `beatPulses` /
 * `beatState` from the FeatureStore (App owns those store entries); App
 * calls {@link applyConfigured} whenever sizes change to rebuild renderers.
 *
 * Layout constants are baked in here — these renderers are only meaningful
 * when overlaid on the rms-history / autocorr-accumulator areas built by
 * App, so co-locating their geometry keeps the App-level wiring trivial.
 */
export class BeatDebugView {
  private grid?: BeatGridRenderer;
  private gridScrolling?: BeatGridScrollingRenderer;
  private pulseSquares?: BeatPulseSquares;

  constructor(private scene: Scene, private store: FeatureStore) {}

  /**
   * Tear down any existing renderers and rebuild them at the new sizes.
   * App owns store buffers; this only handles renderer (re)construction.
   * Idempotent — safe to call multiple times.
   */
  applyConfigured(sizes: BeatDebugSizes): void {
    this.dispose();

    // Linear x-mapping shared by both grid renderers — matches `linearLayout`
    // and `PeakMarkers` so the grid lines pixel-align with the chart lines.
    const linearX = (n: number) => (idx: number) => (n <= 1 ? 0 : (idx / (n - 1)) * 2 - 1);

    // Static grid: top band of the autocorr section (yCenter -1.0, ySpan ±0.4
    // ⇒ top edge at -0.6). Lines extend down 0.2 units.
    this.grid = new BeatGridRenderer({
      source: () => this.store.get("beatGrid"),
      // ceil(rmsAcfLen / MIN_PEAK_LAG=10) is the densest possible grid; +headroom.
      maxLines: 32,
      lagDomain: sizes.onsetAcfLen,
      xForLag: linearX(sizes.onsetAcfLen),
      color: 0xffff66,
    });
    this.scene.add(this.grid.object3d);

    // Scrolling grid: top band of the rms-history section (yCenter -0.5,
    // ySpan ±0.4 ⇒ top edge at -0.1).
    this.gridScrolling = new BeatGridScrollingRenderer({
      source: () => this.store.get("beatGrid"),
      maxLines: 64,
      domain: sizes.rmsLen,
      yTop: -0.5 + 0.4,
      yBottom: -0.5 + 0.4 - 0.2,
      xForIndex: linearX(sizes.rmsLen),
      color: 0xffff66,
    });
    this.scene.add(this.gridScrolling.object3d);
 
    this.pulseSquares = new BeatPulseSquares({
      source: () => this.store.get("beatPulses")
    });

    this.pulseSquares.object3d.position.set(1.2, -0.5, 0);
    this.scene.add(this.pulseSquares.object3d);
  }

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
