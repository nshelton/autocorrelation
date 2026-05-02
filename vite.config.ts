import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  worker: {
    plugins: () => [wasm()],
  },
  resolve: {
    dedupe: ["three"],
  },
  server: {
    port: 5173,
    headers: {
      // Cross-origin isolation unlocks high-resolution performance.now() in
      // AudioWorkletGlobalScope (~5 µs in Chrome vs 1 ms otherwise). Used by
      // crates/dsp/src/perf.rs for dspPerf measurements.
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
