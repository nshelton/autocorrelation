# Audio Visualizer v2 — FFT Spectrum + Tab Audio + FPS — Design

**Date:** 2026-04-27
**Status:** Approved (pending spec review)
**Builds on:** `2026-04-27-audio-visualizer-design.md`, v1 implementation tagged `v1.0.0`

## Overview

v2 turns the v1 pipeline-proof into something genuinely useful: a real FFT spectrum visualization driven by music (via tab audio capture) or live mic input, with a frame-rate counter for development feedback. The architecture from v1 is unchanged in shape — the Rust DSP crate gains an FFT, the audio source layer gains a second factory, and the render layer gains a second `LineRenderer`. No structural rework.

## Goals

- Real-time FFT spectrum, log-frequency, smoothed, rendered as a 3D line below the waveform.
- Tab audio capture as a first-class audio source alongside the mic.
- FPS counter visible during development and demo use.
- Take advantage of the existing `LineRenderer` reusability (no new render component).
- Drive-by cleanup of the small papercuts noted at v1 ship.

## Non-goals

- Spectrum bars (sticking with the line — fits the project's "everyone just uses bars" instinct).
- File upload as a third audio source.
- Mid-run source switching (reload to change source).
- Buffer or RMS-envelope autocorrelation (still v3 / v4).
- Beat detection (v4).
- Particles, post-processing (v5+).

## Stack additions

- Rust: `realfft` crate (real-input FFT, smaller and faster than full complex FFT for this use case).
- TS/JS: `stats.js` (the Three.js community FPS counter; ~3KB, drops onto a div).

## Architecture

The shape from v1 is unchanged:

```
Rust/WASM ──► AudioWorklet ──► Three.js (WebGPURenderer)
   DSP        Real-time      Scene + camera rig + 2× LineRenderer
              orchestration  + FPS overlay
```

Three small structural changes:

1. **DSP output is multi-feature.** The Rust `process()` previously emitted a single `Vec<f32>`. It now emits an aggregate that JS can read multiple named features from. The simplest WASM-friendly shape is a struct with two getter methods: `waveform()` and `spectrum()`, each returning a `Vec<f32>`. JS passes the aggregate's keyed buffers into the `FeatureStore` under `"waveform"` and `"spectrum"` keys.

2. **`App.start` becomes parameterized.** It takes an `AudioSourceFactory` — a function returning `Promise<AudioSourceBundle>`. `main.ts` wires two buttons, each calling `App.start` with a different factory.

3. **Per-feature LineRenderer instantiation.** Two `LineRenderer` instances, one per feature key, each with its own `layout` function and Y offset. No code change to the `LineRenderer` class.

## Component changes

### `crates/dsp/src/lib.rs` — add FFT and spectrum smoothing

- Depend on `realfft = "3"`.
- The `Dsp` struct gains:
  - `fft_planner: RealFftPlanner<f32>`
  - `fft_buffer: Vec<f32>` (sized 2048 for input)
  - `freq_buffer: Vec<Complex<f32>>` (sized 1025 for output)
  - `spectrum: Vec<f32>` (sized 1024 for the magnitude output, post-smoothing)
  - `smoothing_alpha: f32` (default 0.2 — i.e. 20% new, 80% history)
- `process(input)` now:
  1. Copies input into `fft_buffer` (with optional Hann window for cleaner spectra).
  2. Runs the FFT in place.
  3. Computes magnitude in dB, clipped to `[-100, 0]`, normalized to `[0, 1]`.
  4. Applies exponential smoothing per bin: `spectrum[i] = α·new + (1−α)·spectrum[i]`.
  5. Returns a `Features` aggregate exposing `waveform()` and `spectrum()`.
- Window function: Hann. Pre-computed once at construction.

### `src/audio/AudioSource.ts` — add tab audio factory

- New export: `createTabSource(): Promise<AudioSourceBundle>`. Calls `navigator.mediaDevices.getDisplayMedia({ audio: true, video: true })` (browsers require `video: true` to even show the tab picker; we drop the video tracks immediately after).
- Validates that the returned stream has at least one audio track. If not (user picked a video-only window or a tab with no audio), throws a clear error: `"Selected source has no audio. Please pick a tab that's playing audio."`
- Returns the same `AudioSourceBundle` shape as `createMicSource` so downstream code is identical.

### `src/audio/dsp-worklet.ts` — post both features

- Reads the `Features` aggregate from `dsp.process(window)`.
- Posts `{ type: "features", waveform: Float32Array, spectrum: Float32Array }` (single message with both arrays as transferables).
- Slight rename of message type from `"waveform"` to `"features"` to reflect the new content.

### `src/store/FeatureStore.ts` — no code change

- Already keyed by string. v2 just sets `"waveform"` and `"spectrum"` instead of just `"waveform"`.

### `src/render/LineRenderer.ts` — no code change

- v2 just instantiates it twice with different `layout` functions.

### `src/render/LineLayouts.ts` — new

A small helper module that exports both layout functions used in v2: the log-frequency layout for the spectrum and a vertically-offset layout for the waveform. Kept separate so they're testable and clearly named.

```ts
export function waveformLayout(yOffset: number, height: number): LineLayoutFn;
export function logSpectrumLayout(yOffset: number, height: number): LineLayoutFn;
```

Both map sample index → x in `[-1, 1]` (linear for waveform, log2 for spectrum), and value → y in `[yOffset, yOffset + height]`. The default `LineRenderer` layout from v1 stays available as the fallback for ad-hoc uses.

### `src/ui/Stats.ts` — new

Thin wrapper around `stats.js`. Mounts a div in the top-right corner, exposes `begin()` and `end()` methods called from the render loop. Hot path is two function calls per frame; trivial cost.

```ts
export class FpsOverlay {
  mount(parent?: HTMLElement): void;
  begin(): void;
  end(): void;
  unmount(): void;
}
```

### `index.html` — two buttons

Replace the single `<button id="start">` with two: `<button id="start-mic">Mic</button>` and `<button id="start-tab">Tab Audio</button>`. Both styled the same as v1's Start button.

### `src/main.ts` — wire the two buttons

```ts
import { createMicSource, createTabSource } from "./audio/AudioSource";
import { App } from "./App";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const startMic = document.getElementById("start-mic") as HTMLButtonElement;
const startTab = document.getElementById("start-tab") as HTMLButtonElement;

let started = false;
const start = async (factory: () => Promise<AudioSourceBundle>) => {
  if (started) return;
  started = true;
  startMic.disabled = true;
  startTab.disabled = true;
  startMic.textContent = "Running…";
  const app = new App();
  await app.start(canvas, factory);
};

startMic.addEventListener("click", () => start(createMicSource));
startTab.addEventListener("click", () => start(createTabSource));
window.addEventListener("keydown", () => start(createMicSource), { once: true });
```

The "any keypress starts" behavior defaults to mic, since pressing Space or arrow keys to start tab capture would be unintuitive.

### `src/App.ts` — parameterized source

`App.start(canvas, sourceFactory)`. Body unchanged except `await createMicSource()` becomes `await sourceFactory()`. Adds:

- Two more `LineRenderer` instantiations (waveform with `waveformLayout(0.6)`, spectrum with `logSpectrumLayout(-0.6, 0.5)`).
- `FpsOverlay` mounted before the loop, `begin()` at top of loop, `end()` at bottom.
- Third camera preset: `"spectrum"` framing the lower line. Keys 1/2/3 jump to "front"/"side"/"spectrum". Existing Space toggle between front and side preserved.

### `vite.config.ts` — Three.js dedupe

```ts
resolve: {
  dedupe: ["three"],
},
```

Silences the "Multiple instances of Three.js" warning that comes from `three/webgpu` resolving Three core through a different path under some Vite resolver paths.

### `src/audio/dsp-worklet.ts` — louder failure on misconfig

Replace the constructor's missing-wasmModule branch:

```ts
if (!wasmModule) {
  throw new Error("[dsp-worklet] missing wasmModule in processorOptions");
}
```

Was `console.error + return`. The throw propagates as a `processorerror` event on the AudioWorkletNode, which is much louder than a console message buried in worklet output.

## Data flow (per frame)

1. Worklet accumulates a 2048-sample window.
2. Worklet calls `dsp.process(window)`, receives `Features` aggregate.
3. Worklet reads both `waveform()` and `spectrum()` views, copies into transferable Float32Arrays, posts as one message.
4. Main-thread port handler dispatches:
   - `store.set("waveform", msg.waveform)`
   - `store.set("spectrum", msg.spectrum)`
5. Render loop:
   - `fps.begin()`
   - `rig.update(dt)`
   - waveform `LineRenderer.update()` — reads `store.get("waveform")`
   - spectrum `LineRenderer.update()` — reads `store.get("spectrum")`
   - `renderer.render(scene, camera)`
   - `fps.end()`

## Spectrum specifics

- FFT size: 2048 (matches analysis window). Real FFT yields 1025 complex bins; we drop the DC bin (index 0) and use 1024.
- Magnitude conversion: `20 * log10(|bin|)`, clipped to `[-100, 0]` dB, then mapped to `[0, 1]`.
- Smoothing: per-bin exponential moving average with α = 0.2. Tunable later via `tweakpane` if/when we add it.
- Window function: Hann, applied to the input copy before FFT.
- Log frequency layout in JS, not Rust — keeps DSP output canonical so different layouts (linear, mel, etc.) can be tried without rebuilding WASM.

## Camera additions

- New preset `"spectrum"`: positioned just below the v1 "front" preset, looking at the lower line.
- Number keys `1`/`2`/`3` jump to `"front"`/`"side"`/`"spectrum"` respectively, all with the existing 0.8s tween.
- Space still toggles between front and side (preserved from v1).

## Drive-by cleanup folded in

- `vite.config.ts`: `resolve.dedupe = ["three"]`.
- `dsp-worklet.ts`: throw on missing wasmModule.

These were noted as v2 follow-ups at v1 ship and are small enough to land alongside the spectrum work without their own milestone.

## v2 milestone — definition of done

- [ ] `realfft` integrated in Rust crate; FFT runs at audio rate without dropouts.
- [ ] Spectrum line visible below the waveform, log-frequency, visibly responsive to musical input.
- [ ] Two buttons in `index.html`: "Mic" and "Tab Audio". Either path works end-to-end.
- [ ] Tab Audio button reliably captures audio from a tab playing music; clear error if user picks a no-audio source.
- [ ] FPS counter visible in the top-right corner; reads ~60fps under normal load.
- [ ] No "Multiple instances of Three.js" warning in console.
- [ ] Worklet throws (not console.errors) on missing wasmModule.
- [ ] Camera presets `"front"`, `"side"`, `"spectrum"` exist; keys 1/2/3 jump between them; Space still toggles front/side.
- [ ] All v1 unit tests still pass.
- [ ] At least one new unit test for `logSpectrumLayout` verifying the bin→x mapping.

## Out of scope (deferred)

- File upload as a source.
- Bars instead of line.
- Mid-run source switching.
- Spectrum color mapping (color by magnitude).
- Camera presets that move on beat.
- Tweakpane controls for smoothing α / window function / dB range.

## Open questions to revisit during implementation

- **Mono vs stereo for tab audio.** Tab capture often delivers stereo. Worklet currently reads `input[0][0]` (left channel only). For v2 we keep that — left channel is fine for analysis. v3+ may want a mono mix or per-channel spectra.
- **Hann window cost.** Multiplying 2048 samples by a precomputed window each frame is trivial; flagged for completeness only.
- **Tab audio on macOS.** `getDisplayMedia` audio capture is per-tab on macOS (no system audio), as already established. The error path for "user picked entire screen" should be exercised in manual verification.
