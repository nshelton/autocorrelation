# Audio Visualizer — Design

**Date:** 2026-04-27
**Status:** Approved (pending spec review)

## Overview

A web-based, GPU-accelerated audio visualizer with a game-engine-style architecture. High-performance DSP analysis (FFT, autocorrelation) runs in WebAssembly; visuals run entirely on the GPU through Three.js's WebGPU renderer. The architecture treats this less like a "demo page" and more like a small game engine, so visual ambition (particles, 3D scenes, procedural camera animation) is additive rather than a rewrite.

## Goals

- Real-time visualization of live audio from multiple sources (mic, tab audio, file).
- Support multiple analysis methods: FFT spectrum, autocorrelation on the raw audio buffer, autocorrelation on the per-frame RMS envelope (for beat detection).
- 3D scene with a camera abstraction supporting named presets, smooth transitions between them, and procedural per-frame control.
- Minimal CPU work in steady state — feature data lands on the GPU; everything visual happens in shaders.
- Architecture that allows particle systems, post-processing, and richer scenes to be added without restructuring.

## Non-goals

- Native or installable app.
- Audio editing, recording, or DAW-style features.
- Server-side processing.
- Music information retrieval beyond beat/period detection.
- True OS-wide system audio capture (requires virtual audio devices; out of scope).

## Stack

- **Rendering:** Three.js with `WebGPURenderer` and TSL (Three.js Shading Language)
- **DSP:** Rust crate at `crates/dsp/`, compiled to WebAssembly with SIMD enabled, via `wasm-pack`
- **Real-time audio:** `AudioWorkletProcessor` bridging Web Audio → WASM → main thread
- **App layer:** TypeScript
- **Bundler:** Vite, with `vite-plugin-wasm` and `vite-plugin-top-level-await`
- **Runtime parameter knobs:** `tweakpane`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Main thread (TS)                       │
│   Three.js + WebGPURenderer                                 │
│   App, CameraRig, LineRenderer(s), FeatureStore, UI         │
└──────────────▲──────────────────────────────────────────────┘
               │  postMessage with transferable Float32Array
               │  (FFT bins, autocorr lags, RMS, beat phase)
┌──────────────┴──────────────────────────────────────────────┐
│                  AudioWorklet thread                        │
│   DSPWorklet: buffers analysis windows, calls into WASM     │
└──────────────▲──────────────────────────────────────────────┘
               │
┌──────────────┴──────────────────────────────────────────────┐
│              Rust/WASM (compiled with SIMD)                 │
│   process(window) -> Features                               │
│   - realfft for spectrum                                    │
│   - autocorrelation via Wiener-Khinchin                     │
│   - RMS-envelope autocorr for beat detection                │
└─────────────────────────────────────────────────────────────┘
```

Three layers, one job each. The DSP→GPU contract is "here's a `Float32Array` of features per frame; upload it to a 1D texture or storage buffer." After that, everything is shaders.

We do **not** start with `SharedArrayBuffer`. At our data rate (~60 Hz × a few hundred floats), `postMessage` with transferable Float32Arrays is sufficient and avoids forcing COOP/COEP headers on the dev server. SAB remains an option if profiling later demands it.

## Components

### `AudioSource`

Unified interface over input methods. Returns a Web Audio source node connected to the worklet. Handles the user-gesture requirement to start `AudioContext`.

- **v1:** microphone via `getUserMedia({ audio: true })`
- **v2:** tab audio via `getDisplayMedia({ audio: true })`; file upload decoded via `AudioContext.decodeAudioData`

### `DSPWorklet` (AudioWorkletProcessor)

Buffers raw audio frames (the worklet's native 128-sample chunks) into analysis windows (default 2048 samples, 50% overlap). Calls into the WASM module's `process` function. Posts the resulting features to the main thread via `postMessage` with transferable `Float32Array`s.

- **v1:** passthrough — emits the raw analysis window as the only feature.
- **v2+:** real feature extraction.

### `WasmDsp` (Rust crate)

Single public entry point: `process(window: &[f32]) -> Features`, where `Features` is a struct of `Float32Array`-shaped buffers exposed to JS. Compiled with SIMD128 enabled.

- **v1:** identity — copies window to output.
- **v2:** FFT (via `realfft`).
- **v3:** buffer autocorrelation via Wiener-Khinchin (`autocorr(x) = IFFT(|FFT(x)|²)`).
- **v4:** RMS envelope tracker + autocorrelation on the envelope; emits detected period and beat phase.

### `FeatureStore`

Main-thread holder of the most recent feature buffers. Subscribes to the worklet's port, swaps in new typed arrays as they arrive, exposes named accessors (`getWaveform()`, `getSpectrum()`, etc.) to renderers.

### `LineRenderer`

Generic 3D line that consumes a `Float32Array` source. The reusable visualization primitive.

```ts
class LineRenderer {
  constructor(opts: {
    source: () => Float32Array;
    layout?: (i: number, n: number, value: number) => Vector3;
    color?: ColorRepresentation;
    width?: number;
  });
  update(): void;       // called each frame
  object3d: Object3D;
}
```

The `layout` callback is the extensibility hook. Default is "linear X axis, value → Y." Swapping the callback turns the same renderer into a circular oscilloscope, a helix, an autocorrelation curve, or a polar spectrum — without subclassing.

### `CameraRig`

Wraps a `PerspectiveCamera`. Three mutually exclusive operating modes: static preset, tweening between presets, or procedural per-frame override.

```ts
class CameraRig {
  camera: PerspectiveCamera;
  addPreset(name: string, pose: { position: Vector3; target: Vector3; fov?: number }): void;
  goTo(name: string, opts?: { duration?: number; easing?: EasingFn }): Promise<void>;
  setProceduralController(fn: ((dt: number, camera: PerspectiveCamera) => void) | null): void;
  update(dt: number): void;
}
```

`update(dt)` is called once per frame from the render loop. If a tween is active it advances; if a procedural controller is set it runs; otherwise the camera holds its preset pose.

**v1 presets:**
- `"front"` — default, looking at scene origin from `+Z`
- `"side"` — 90° rotation around Y, used to exercise the tween path

### `App`

Top-level coordinator. Owns the `requestAnimationFrame` loop and wires data flow:

```
AudioSource → AudioContext → DSPWorklet → FeatureStore → renderers
                                                        ↘ CameraRig.update
                                                        ↘ WebGPURenderer.render
```

## Data flow (per frame)

1. Worklet accumulates samples until an analysis window is full.
2. Worklet calls `wasm.process(window)`, receives `Features`.
3. Worklet posts features to main thread (transferable, zero-copy).
4. `FeatureStore` swaps in the new buffers.
5. On the next `requestAnimationFrame`:
   - `CameraRig.update(dt)`
   - Each `LineRenderer.update()` reads its source buffer and updates GPU geometry.
   - `WebGPURenderer.render(scene, camera)`.

## Milestones

### v1 — Pipeline proof (definition of done)

- [ ] `npm run dev` boots a Vite app.
- [ ] User clicks "Start" → mic permission prompt → audio flowing.
- [ ] WASM module loaded, worklet calls `process` (passthrough).
- [ ] `LineRenderer` shows the live waveform on a flat plane.
- [ ] `CameraRig` has two presets; a keyboard key tweens between them smoothly.
- [ ] `WebGPURenderer` rendering at a stable 60 fps on a recent laptop.

No FFT, no autocorrelation, no particles. v1 proves the pipeline; everything later is additive.

### v2 — Spectrum + tab audio + file upload

- Add `realfft` to the Rust crate; emit spectrum bins as a feature.
- Second `LineRenderer` instance bound to the spectrum.
- `getDisplayMedia` source for tab audio.
- File upload source via `decodeAudioData`.

### v3 — Autocorrelation on raw buffer

- Wiener-Khinchin autocorrelation in the Rust crate.
- Third `LineRenderer` for the autocorrelation curve.
- Larger analysis window (4096+) if needed for useful lag range.

### v4 — Beat detection

- Per-frame RMS tracker.
- Autocorrelation of RMS envelope over a multi-second window.
- Detected period and beat phase exposed as scalar features.
- Visual indicator driven by beat phase.

### v5+ — Visual ambition

- GPU-resident particle system (compute shader, audio features as input).
- Post-processing chain (bloom, tone-mapping).
- Additional camera presets and procedural controllers (orbits, lookat-tracking, hand-authored sequences).
- Optional: glTF environment models.

## Out of scope (v1)

- File upload UI, tab audio, system audio.
- FFT, autocorrelation of any kind.
- Particles, lighting beyond ambient, post-processing.
- Multiple `LineRenderer` instances.
- Persisting camera presets to disk.

## Open questions to revisit during implementation

- **Analysis window size.** 2048 is the v1 default; autocorrelation work in v3+ may want 4096+ for useful lag range. Confirm with measurements.
- **Tween library.** Hand-rolled easing covers v1's two-preset case. If procedural sequences in v5 grow elaborate, consider `gsap` or a small dedicated tween lib.
- **`SharedArrayBuffer`.** Not needed initially. Revisit if `postMessage` overhead shows up in profiling.
- **Worklet ↔ WASM linkage.** WASM in `AudioWorkletGlobalScope` requires synchronous instantiation of the wasm module bytes; verify the `wasm-pack` output works without dynamic import in that context.
