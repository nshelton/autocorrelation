# Audio Visualizer v2 — FFT Spectrum + Tab Audio + FPS — Design

**Date:** 2026-04-27
**Status:** Approved (pending spec review)
**Builds on:** `2026-04-27-audio-visualizer-design.md`, v1 implementation tagged `v1.0.0`

## Overview

v2 turns the v1 pipeline-proof into something genuinely useful: a real FFT spectrum visualization driven by music (via tab audio capture) or live mic input, with a frame-rate counter for development feedback. The architecture from v1 is unchanged in shape — the Rust DSP crate gains an FFT, the audio source layer gains a second factory, and the render layer gains a second `LineRenderer`. No structural rework.

## Goals

- Real-time FFT spectrum, log-frequency, smoothed, rendered as a 3D line below the waveform.
- Rolling RMS history (per-window loudness over the last ~11s) as a third line below the spectrum. Sets up v4's beat-detection autocorrelation cleanly: the buffer is already there, beat detection just adds an autocorrelation pass over it.
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
   DSP        Real-time      Scene + camera rig + 3× LineRenderer
              orchestration  + FPS overlay
```

Three small structural changes:

1. **DSP output is multi-feature.** The Rust `process()` previously emitted a single `Vec<f32>`. It now emits an aggregate that JS can read multiple named features from. The simplest WASM-friendly shape is a struct with three getter methods: `waveform()`, `spectrum()`, and `rms_history()`, each returning a `Vec<f32>`. JS passes the aggregate's keyed buffers into the `FeatureStore` under `"waveform"`, `"spectrum"`, and `"rms"` keys.

2. **`App.start` becomes parameterized.** It takes an `AudioSourceFactory` — a function returning `Promise<AudioSourceBundle>`. `main.ts` wires two buttons, each calling `App.start` with a different factory.

3. **Per-feature LineRenderer instantiation.** Two `LineRenderer` instances, one per feature key, each with its own `layout` function and Y offset. No code change to the `LineRenderer` class.

## Component changes

### `crates/dsp/src/lib.rs` — add FFT, spectrum smoothing, and rolling RMS

- Depend on `realfft = "3"`.
- The `Dsp` struct gains:
  - `fft_planner: RealFftPlanner<f32>`
  - `fft_buffer: Vec<f32>` (sized 2048 for input)
  - `freq_buffer: Vec<Complex<f32>>` (sized 1025 for output)
  - `spectrum: Vec<f32>` (sized 1024 for the magnitude output, post-smoothing)
  - `rms_history: Vec<f32>` (sized 256 — ~11s of history at ~23 windows/sec)
  - `hann: Vec<f32>` (sized 2048, pre-computed at construction)
  - `smoothing_alpha: f32` (default 0.2 — i.e. 20% new, 80% history)
- `process(input)` now:
  1. Computes RMS over the input window: `sqrt(mean(input[i]²))`.
  2. Shifts `rms_history` left by one and appends the new RMS at the end. Implementation: `rms_history.copy_within(1.., 0); rms_history[N-1] = rms;`. With N=256 this is a trivial memmove (~1KB) per window.
  3. Copies input × Hann into `fft_buffer`.
  4. Runs the real FFT in place.
  5. Computes magnitude in dB from bins 1..1024 (DC dropped), clipped to `[-100, 0]`, normalized to `[0, 1]`.
  6. Applies exponential smoothing per bin: `spectrum[i] = α·new + (1−α)·spectrum[i]`.
  7. Returns a `Features` aggregate exposing `waveform()`, `spectrum()`, and `rms_history()`.
- Window function: Hann. Pre-computed once at construction.

**Why shifted Vec instead of circular buffer for `rms_history`:** the buffer is small (256 floats), the cost of shifting is negligible (~6000 float-moves/sec), and the buffer hands directly to JS in temporal order (oldest → newest) without an unroll step. Autocorrelation in v4 will be the textbook formula `Σᵢ buf[i]·buf[i+lag]` rather than the modulo-indexed version. Simpler code, same math.

### `src/audio/AudioSource.ts` — add tab audio factory

- New export: `createTabSource(): Promise<AudioSourceBundle>`. Calls `navigator.mediaDevices.getDisplayMedia({ audio: true, video: true })` (browsers require `video: true` to even show the tab picker; we drop the video tracks immediately after).
- Validates that the returned stream has at least one audio track. If not (user picked a video-only window or a tab with no audio), throws a clear error: `"Selected source has no audio. Please pick a tab that's playing audio."`
- Returns the same `AudioSourceBundle` shape as `createMicSource` so downstream code is identical.

### `src/audio/dsp-worklet.ts` — post all three features

- Reads the `Features` aggregate from `dsp.process(window)`.
- Posts `{ type: "features", waveform: Float32Array, spectrum: Float32Array, rms: Float32Array }` (single message with all three arrays as transferables).
- Slight rename of message type from `"waveform"` to `"features"` to reflect the new content.

### `src/store/FeatureStore.ts` — no code change

- Already keyed by string. v2 sets three keys: `"waveform"`, `"spectrum"`, and `"rms"`.

### `src/render/LineRenderer.ts` — no code change

- v2 instantiates it three times with different `layout` functions and Y offsets.

### `src/render/LineLayouts.ts` — new

A small helper module that exports the layout functions used in v2: linear for the waveform and RMS history, log-frequency for the spectrum. Kept separate so they're testable and clearly named.

```ts
export function linearLayout(yOffset: number, height: number): LineLayoutFn;
export function logSpectrumLayout(yOffset: number, height: number): LineLayoutFn;
```

`linearLayout` maps sample index → x in `[-1, 1]`, value → y in `[yOffset, yOffset + height]` — used for both the waveform and the RMS history (same shape, different yOffset and source data). `logSpectrumLayout` does the same but with a log2 mapping on x. The default `LineRenderer` layout from v1 stays available as the fallback for ad-hoc uses.

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

- Three `LineRenderer` instantiations:
  - waveform: `linearLayout(0.6, 0.5)`, source `() => store.get("waveform")`
  - spectrum: `logSpectrumLayout(0.0, 0.5)`, source `() => store.get("spectrum")`
  - rms history: `linearLayout(-0.6, 0.5)`, source `() => store.get("rms")`
- `FpsOverlay` mounted before the loop, `begin()` at top of loop, `end()` at bottom.
- Camera presets: `"front"` (overview, all three lines visible), `"side"`, `"spectrum"` (frames the spectrum line), `"rms"` (frames the rms history line). Keys 1/2/3/4 jump to them respectively. Existing Space toggle between front and side preserved.
- The store needs all three keys primed with empty Float32Arrays before the LineRenderers are constructed (same workaround as v1 for the WebGPU length-change issue): `store.set("waveform", new Float32Array(2048))`, `store.set("spectrum", new Float32Array(1024))`, `store.set("rms", new Float32Array(256))`.

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
3. Worklet reads `waveform()`, `spectrum()`, and `rms_history()` views, copies into transferable Float32Arrays, posts as one message.
4. Main-thread port handler dispatches:
   - `store.set("waveform", msg.waveform)`
   - `store.set("spectrum", msg.spectrum)`
   - `store.set("rms", msg.rms)`
5. Render loop:
   - `fps.begin()`
   - `rig.update(dt)`
   - waveform `LineRenderer.update()` — reads `store.get("waveform")`
   - spectrum `LineRenderer.update()` — reads `store.get("spectrum")`
   - rms `LineRenderer.update()` — reads `store.get("rms")`
   - `renderer.render(scene, camera)`
   - `fps.end()`

## Spectrum specifics

- FFT size: 2048 (matches analysis window). Real FFT yields 1025 complex bins; we drop the DC bin (index 0) and use 1024.
- Magnitude conversion: `20 * log10(|bin|)`, clipped to `[-100, 0]` dB, then mapped to `[0, 1]`.
- Smoothing: per-bin exponential moving average with α = 0.2. Tunable later via `tweakpane` if/when we add it.
- Window function: Hann, applied to the input copy before FFT.
- Log frequency layout in JS, not Rust — keeps DSP output canonical so different layouts (linear, mel, etc.) can be tried without rebuilding WASM.

## RMS history specifics

- Buffer size: 256 samples (one per analysis window, ~11s at ~23 windows/sec).
- Computed in Rust as `sqrt(mean(samples²))` over each 2048-sample window.
- No additional smoothing applied — each sample is already the integral of 2048 audio samples, which is the smoothing.
- No transformation in JS; the `LineRenderer` reads the buffer directly and plots oldest-on-left, newest-on-right via `linearLayout`.
- Frequency resolution: each pixel in x represents one window (~42.7 ms). 60 BPM = 1Hz beat = ~23 samples between consecutive beats; 240 BPM = ~6 samples. Both are clearly visible.

## Camera additions

- New presets `"spectrum"` and `"rms"`: each frames its respective line tightly. `"front"` shows all three lines.
- Number keys `1`/`2`/`3`/`4` jump to `"front"`/`"side"`/`"spectrum"`/`"rms"` respectively, all with the existing 0.8s tween.
- Space still toggles between front and side (preserved from v1).

## Drive-by cleanup folded in

- `vite.config.ts`: `resolve.dedupe = ["three"]`.
- `dsp-worklet.ts`: throw on missing wasmModule.

These were noted as v2 follow-ups at v1 ship and are small enough to land alongside the spectrum work without their own milestone.

## v2 milestone — definition of done

- [ ] `realfft` integrated in Rust crate; FFT runs at audio rate without dropouts.
- [ ] Spectrum line visible (middle line), log-frequency, visibly responsive to musical input.
- [ ] RMS history line visible (bottom line), 256 samples wide, visibly tracking loudness over the last ~11 seconds.
- [ ] Two buttons in `index.html`: "Mic" and "Tab Audio". Either path works end-to-end.
- [ ] Tab Audio button reliably captures audio from a tab playing music; clear error if user picks a no-audio source.
- [ ] FPS counter visible in the top-right corner; reads ~60fps under normal load.
- [ ] No "Multiple instances of Three.js" warning in console.
- [ ] Worklet throws (not console.errors) on missing wasmModule.
- [ ] Camera presets `"front"`, `"side"`, `"spectrum"`, `"rms"` exist; keys 1/2/3/4 jump between them; Space still toggles front/side.
- [ ] All v1 unit tests still pass.
- [ ] New unit tests for `linearLayout` and `logSpectrumLayout` verifying the index→x mapping.
- [ ] New unit test for the Rust RMS computation (table-driven against known inputs).

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
