import { Scene } from "three";
import type { FeatureStore } from "../../store/FeatureStore";

/**
 * Stub. The pre-Phase-2 version composed three beat-debug renderers (static
 * autocorr grid, scrolling rms-history grid, 2×2 pulse squares) via an
 * applyConfigured(sizes) method that App.ts called on `configured` messages.
 * That protocol is gone — sub-renderers are no longer wired. Lazy-init
 * refactor (each renderer self-detects size on first update) lands later.
 */
export class BeatDebugView {
  constructor(_scene: Scene, _store: FeatureStore) {}

  update(): void {}

  dispose(): void {}
}
