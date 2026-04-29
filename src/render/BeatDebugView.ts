import { Scene } from "three";
import { BeatGridRenderer } from "./BeatGridRenderer";
import { BeatGridScrollingRenderer } from "./BeatGridScrollingRenderer";
import { BeatPulseSquares } from "./BeatPulseSquares";
import type { FeatureStore } from "../store/FeatureStore";

export interface BeatDebugSizes {
  rmsLen: number;
  rmsAcfLen: number;
  beatGridLen: number;
  beatPulsesLen: number;
}

export interface BeatDebugFeatures {
  beatGrid?: Float32Array;
  beatPulses?: Float32Array;
}

/**
 * Composes the three beat-debug renderers (static autocorr grid, scrolling
 * rms-history grid, 2×2 pulse squares) along with their shared buffer
 * allocations and feature ingestion. Owns its own subset of the FeatureStore
 * keys: `beatGrid` and `beatPulses`. App passes through `applyFeatures()`
 * for each "features" message and `applyConfigured()` whenever sizes change.
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

  applyFeatures(msg: BeatDebugFeatures): void {
    if (msg.beatGrid) this.store.set("beatGrid", msg.beatGrid);
    if (msg.beatPulses) this.store.set("beatPulses", msg.beatPulses);
  }

  /**
   * Tear down any existing renderers, allocate fresh NaN-filled buffers in
   * the store, and rebuild all three renderers at the new sizes. Idempotent
   * — safe to call multiple times.
   */
  applyConfigured(sizes: BeatDebugSizes): void {
    this.dispose();

    const beatGridInit = new Float32Array(sizes.beatGridLen);
    beatGridInit.fill(NaN);
    this.store.set("beatGrid", beatGridInit);
    const beatPulsesInit = new Float32Array(sizes.beatPulsesLen);
    beatPulsesInit.fill(NaN);
    this.store.set("beatPulses", beatPulsesInit);

    // Linear x-mapping shared by both grid renderers — matches `linearLayout`
    // and `PeakMarkers` so the grid lines pixel-align with the chart lines.
    const linearX = (n: number) => (idx: number) => (n <= 1 ? 0 : (idx / (n - 1)) * 2 - 1);

    // Static grid: top band of the autocorr section (yCenter -1.0, ySpan ±0.4
    // ⇒ top edge at -0.6). Lines extend down 0.2 units.
    this.grid = new BeatGridRenderer({
      source: () => this.store.get("beatGrid"),
      // ceil(rmsAcfLen / MIN_PEAK_LAG=10) is the densest possible grid; +headroom.
      maxLines: 32,
      lagDomain: sizes.rmsAcfLen,
      yTop: -1.0 + 0.4,
      yBottom: -1.0 + 0.4 - 0.2,
      xForLag: linearX(sizes.rmsAcfLen),
      color: 0xffff66,
    });
    this.scene.add(this.grid.object3d);

    // Scrolling grid: top band of the rms-history section (yCenter -0.5,
    // ySpan ±0.4 ⇒ top edge at -0.1).
    this.gridScrolling = new BeatGridScrollingRenderer({
      source: () => this.store.get("beatGrid"),
      // Densest possible grid is rmsLen / MIN_PEAK_LAG (~52 at default).
      maxLines: 64,
      domain: sizes.rmsLen,
      yTop: -0.5 + 0.4,
      yBottom: -0.5 + 0.4 - 0.2,
      xForIndex: linearX(sizes.rmsLen),
      color: 0xffff66,
    });
    this.scene.add(this.gridScrolling.object3d);

    // 2×2 pulse squares just past the right edge of the rms area (x ∈ [-1, 1]),
    // vertically centered on the rms section (yCenter = -0.5).
    this.pulseSquares = new BeatPulseSquares({
      source: () => this.store.get("beatPulses"),
      count: 4,
      centerX: 1.6,
      centerY: -0.2,
      cellSize: 0.5,
      gap: 0.02,
    });
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
