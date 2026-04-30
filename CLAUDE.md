# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- `npm run dev` — Vite dev server on port 5173.
- `npm run build` / `npm run preview` — production build / preview.
- `npm test` — run TS tests once (vitest, `happy-dom` env, `tests/**/*.test.ts`).
- `npm run test:watch` — vitest watch.
- `npm run wasm` — rebuild Rust DSP crate to WASM. Runs `wasm-pack build --target web --out-dir ../../src/wasm-pkg` from `crates/dsp`. **Run this whenever `crates/dsp/` changes** — `src/wasm-pkg/` is gitignored and not produced by `npm run dev`/`build`.
- Single TS test: `npx vitest run tests/render/CameraRig.test.ts` (or `-t "<test name pattern>"`).
- Rust tests: `cargo test -p dsp` from repo root.

## Architecture

Real-time audio visualizer. Audio flows: **source → AudioWorklet (Rust/WASM DSP) → main thread FeatureStore → Three.js WebGPU LineRenderers**. All DSP runs off the main thread; rendering and feature storage stay on it.

### Audio path (`src/audio/` + `crates/dsp/`)

- `AudioSource.ts` exposes three `AudioSourceBundle` factories: `createMicSource` (getUserMedia, no AGC/NS/EC), `createTabSource` (getDisplayMedia, drops video tracks immediately — picker requires both audio+video to surface tab-audio capture), `createTestSource` (in-process oscillator, no permissions — used for fast iteration; press `T` before any source is started).
- `dsp-worklet.ts` is the AudioWorklet processor. It receives a precompiled `WebAssembly.Module` via `processorOptions` (compiled on the main thread, passed in — workers cannot `compileStreaming` their own URL). It maintains a sliding window (`windowSize`, default 2048) and fires the FFT every `hopSize` samples (default 1024 = 50 % overlap → ~47 Hz update rate at 48 kHz). Output buffers are transferred (zero-copy) to the main thread via `postMessage`.
- `worklet-polyfills.ts` stubs `TextDecoder`/`TextEncoder` in `AudioWorkletGlobalScope` because wasm-pack `--target web` glue instantiates a `TextDecoder` at module load. **Must be imported before** the wasm-pack glue — keep the `import "./worklet-polyfills"` line first in `dsp-worklet.ts`.
- `crates/dsp/src/lib.rs` (`Dsp` struct) does Hann-windowed real FFT (`realfft`), magnitude → dBFS → normalized [0,1] → temporally smoothed (α≈0.2 at default settings, configurable via `dsp.smoothingTauSecs`). `mag_scale = 2/sum(hann)` so a unit-amplitude sine peaks at ~1.0. Bin 0 (DC) is dropped from the output spectrum. The same window also produces `rms_history` (full-band envelope) plus `low_rms_history` / `mid_rms_history` / `high_rms_history` from Parseval-correct band-energy summation.

### Beat detection pipeline (`crates/dsp/src/lib.rs`)

This is the bottom half of `Dsp::process` — runs after the FFT/RMS stages and feeds the beat-related output buffers.

```
rms_history (detrended) ──autocorrelate──▶  rms_acf
                                              │ EMA τ=accumTauSecs
                                              ▼
                                       rms_acf_accum  ──pick_acf_peaks──▶  acf_peaks
                                              │                            (top-N, [lag, mag,
                                              │                             sharpness] stride 3)
                                              │                                   │
                                              │                                   ▼
                                              │                          BeatTracker.observe
                                              │                            (state estimator)
                                              │                                   │
                                              ▼                                   ▼
                                    fit_beat_phase  ◀────── bpm_period ────  beat_grid
                                              │                              [period, phase, conf]
                                              │                              beat_state
                                              ▼                              [bpm, bpm_conf,
                                       update_beat_pulses                     beats_per_measure,
                                              │                               measure_conf]
                                              ▼
                                       beat_pulses  (4 saws at 1×/2×/4×/8× period)
```

- `pick_acf_peaks` writes `[lag, mag, sharpness]` triples (stride 3). Sharpness = `-(y₀ - 2y₁ + y₂)` from parabolic interp — large for narrow real beats, small for broad smeared lobes; the tracker uses it to prefer pointy peaks at `m·period` for measure detection.
- `BeatTracker` is a state estimator: holds smoothed `bpm_period` (continuous, EMA with confidence-gated rate `α(c) = α_max · c`) and discrete `beats_per_measure` (changes only with 1.3× hysteresis margin). Initial state: 120 BPM, 4/4. Bounded by `BEAT_TRACKER_MIN_BPM`/`MAX_BPM` (80–190) — observations whose `lag/k` falls outside this range are filtered, and `bpm_period` is clamped, so sub-period peaks (hi-hats, 16th-note patterns) can't pull the tracker into reporting absurd tempos.
- `fit_beat_phase` returns NaN when `rms_history` is silent; `update_beat_pulses` then **free-runs** the saws by advancing `beat_position += 1/period` each frame instead of locking to a bogus phase=0. This keeps the squares pulsing at the tracker's default rate during silence rather than sticking on full white.
- `acf_peaks` is consumed by both the tracker (BPM/measure observations) and `PeakMarkers` (rendered markers on the autocorr lane). Length = `3 * MAX_PEAKS` = 30 — keep stride-3 indexing in any new consumer.

### Rendering path (`src/render/`, `src/store/`, `src/App.ts`)

- `Scene.ts` creates a `WebGPURenderer` (from `three/webgpu`) — **not** the default WebGL renderer. `await renderer.init()` is required before first render.
- `App.ts` is the **thin orchestrator** (~130 lines): scene + camera + `CameraRig` presets (`1`=front, `2`=side, `3`=spectrum, `4`=rms, `5`=buffer-acf, `6`=rms-acf; space toggles front↔side), keyboard/resize listeners, the RAF loop, and worklet message routing. Each `features` and `configured` message is forwarded straight to `DebugView` — App owns no per-feature state.
- `DebugView` (`src/render/DebugView.ts`) owns every visualization layer: waveform / spectrum / buffer-ACF / multi-band RMS (full + low/mid/high, with autogain) / rms-ACF lines, peak markers, and a nested `BeatDebugView`. `applyConfigured` rebuilds renderers + allocates `FeatureStore` buffers, `applyFeatures` ingests the worklet's payload, `update` ticks all renderers, `dispose` tears down. Layout constants (Y offsets, colors) are baked in here.
- `BeatDebugView` owns the three beat-debug renderers: `BeatGridRenderer` (vertical lines at `k·period` over the autocorr lane), `BeatGridScrollingRenderer` (same but in rms-history-buffer-index space, scrolling left with phase), `BeatPulseSquares` (4 grayscale squares pulsing at `BEAT_PULSE_CYCLES = [1, 2, 4, 8]` cycles of the period — saw waves whose brightness encodes phase within each cycle).
- Per-channel **autogain** lives on the TS side (`DebugView.applyAutoGain`), not in the DSP. The 4 RMS-history lines (full + 3 bands) are normalized by a running peak per channel, decayed at `dsp.autoGain` τ (seconds). The renderer reads from parallel `${key}Auto` store entries; raw `rmsLow` / `rmsMid` etc. stay untouched.
- `LineRenderer` reads its `Float32Array` via a `source: () => Float32Array` callback (so the buffer can be swapped in the store without re-wiring), maps each sample through a `LineLayoutFn` into a `BufferAttribute` with `DynamicDrawUsage`, and reallocates if buffer length changes.
- `LineLayouts.ts` provides `linearLayout` (used for waveform & RMS) and `logSpectrumLayout` (log2 x-axis for the spectrum). All grid/marker renderers (`BeatGridRenderer`, `PeakMarkers`, …) use the same `xForLag = (i, n) => (i / (n-1)) * 2 - 1` formula so overlays stay pixel-aligned with the chart lines.
- `CameraRig` supports named presets, eased tweens (`goTo`), and an optional procedural controller; `goTo` returns a promise that resolves on tween completion.
- `FeatureStore` is intentionally a thin `Map<string, Float32Array>` — buffers in, buffers out; no events.

### Worklet ↔ main message protocol

Three message types (declared in `src/audio/dsp-worklet.ts`):

- **`configured`** (worklet → main): emitted on boot and whenever the worklet's `applyConfigure` runs. Carries every output buffer's length (`waveformLen`, `spectrumLen`, `rmsLen`, `rmsAcfLen`, `acfPeaksLen`, `beatGridLen`, `beatPulsesLen`, `beatStateLen`, …). The worklet **caches the last `configured` payload** so it can re-emit without recreating `Dsp`.
- **`features`** (worklet → main, ~47 Hz): one `Float32Array` per output buffer. Buffers are zero-copied via `transfer` — the worklet allocates fresh arrays each frame.
- **`sync`** (main → worklet, HMR-only): triggers re-emission of the cached `configured` payload. Posted by `App.start()` after the new instance wires `onmessage`. This is what lets HMR rebuild every line renderer at the right sizes **without resetting any DSP state** (rms_history accumulator, beat tracker, etc.) — the page-lifetime `Dsp` keeps running while the App-lifetime renderers tear down and rebuild.

### Param store & WorkletBridge (`src/params/`)

- `ParamStore` holds `dsp.*` keys with continuous/discrete schemas (`schemas.ts`). `WorkletBridge` subscribes to changes and forwards them to the worklet — `windowSize`/`rmsHistoryLen` are **reconfig** (rebuild Dsp); `hopSize`/`smoothingTauSecs`/`dbFloor`/`accumTauSecs` are **hot keys** (in-place setter on the live Dsp).
- `dsp.autoGain` is **TS-only** — it controls the autogain τ in `DebugView`, never reaches the worklet. Don't add it to `HOT_KEYS`.
- `bridge.bootstrap()` is called once per page-lifetime in `main.ts`. On HMR the bridge is recreated but `bootstrap()` does **not** re-run — the worklet keeps its current Dsp + params; the App's new renderers come up via the `sync` mechanism above.

### Build pipeline notes

- `vite-plugin-wasm` + `vite-plugin-top-level-await` handle `import "...?url"` of wasm and worklet sources. `vite.config.ts` also applies `wasm()` to the `worker` plugin chain (audio worklets count). The wasm URL and worklet URL are imported via `?url` / `?worker&url` and passed to `addModule` / `WebAssembly.compileStreaming`.
- `resolve.dedupe: ["three"]` is intentional — without it Three.js double-loads (the `three/webgpu` subpath imports its own copy). A "Multiple instances of Three.js" warning is a known leftover and is on the roadmap.
- `src/wasm-pkg/` is generated; never edit by hand and never check it in.

## Conventions

- Comments explain **why** (non-obvious invariants, hidden constraints, pitfalls like the polyfill ordering or `mag_scale` derivation), not what. Existing code is the reference for tone — match it.
- TypeScript is strict (`noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`). Keep it that way.
- Specs and plans live in `docs/superpowers/specs/` and `docs/superpowers/plans/`. The roadmap (`ROADMAP.md`) tracks deferred work — check it before starting a new feature.

## Pitfalls / non-obvious invariants

- **`acf_peaks` is stride 3.** Triples `[lag, mag, sharpness]`. Length 30 (= 3 × `MAX_PEAKS`). Anything iterating peaks must step in 3s. NaN means "empty slot."
- **Worklet `sync` is HMR-only.** App posts it once on `start()`. Don't post it from feature-update paths or you'll churn line renderers each frame.
- **HMR teardown order matters.** `App.dispose()` sets `port.onmessage = null` before disposing `DebugView` — otherwise an in-flight features message could land on a half-disposed renderer.
- **BeatTracker bounds are hard.** `bpm_period` is clamped to ~14.8–35.2 lags (190–80 BPM at default sr/hop). Music outside this range is reported at the nearest octave, not at its true tempo.
- **Beat phase NaN ≠ broken.** `fit_beat_phase` returns NaN under silence; renderers + `update_beat_pulses` handle this by free-running. Don't "fix" the NaN by defaulting to 0 — that locks the saws on full white.
- **`port.onmessage = null` then `= handler` does NOT replay queued messages.** Anything posted between dispose and the next `start()` is dropped. Counted on for HMR; don't try to recover it.
