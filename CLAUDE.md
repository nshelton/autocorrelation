# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- `npm run dev` — Vite dev server on port 5173.
- `npm run build` / `npm run preview` — production build / preview.
- `npm test` — run TS tests once (vitest, `happy-dom` env, `tests/**/*.test.ts`).
- `npm run test:watch` — vitest watch.
- `npm run wasm` — rebuild Rust DSP crate to WASM. Runs `wasm-pack build --target web --out-dir ../../src/wasm-pkg` from `crates/dsp`. **Run this whenever `crates/dsp/` changes** — `src/wasm-pkg/` is gitignored and not produced by `npm run dev`/`build`.
- Single TS test: `npx vitest run tests/render/<X>.test.ts` (or `-t "<test name pattern>"`).
- Rust tests: `cargo test -p dsp` from repo root.

## Architecture

Real-time audio visualizer. Audio flows: **source → AudioWorklet (Rust/WASM DSP) → main thread FeatureStore → Three.js WebGPU renderers**. All DSP runs off the main thread; rendering and feature storage stay on it.

### Audio path (`src/audio/` + `crates/dsp/`)

- `AudioSource.ts` exposes two `AudioSourceBundle` factories: `createMicSource` (getUserMedia, no AGC/NS/EC) and `createTabSource` (getDisplayMedia, drops video tracks immediately — picker requires both audio+video to surface tab-audio capture). Both require a permission prompt; there is no permission-free test source, so the app cannot be driven end-to-end headlessly. Any key pressed on the start screen picks the mic.
- `dsp-worklet.ts` is the AudioWorklet processor. It receives a precompiled `WebAssembly.Module` via `processorOptions` (compiled on the main thread, passed in — workers cannot `compileStreaming` their own URL). It maintains a sliding window (`windowSize`, default 2048) and fires the FFT every `hopSize` samples (default 1024 = 50 % overlap → ~47 Hz update rate at 48 kHz). Output buffers go zero-copy via `postMessage` transfer list.
- `worklet-polyfills.ts` provides UTF-8 `TextDecoder` / `TextEncoder` polyfills for `AudioWorkletGlobalScope` in browsers that lack them. wasm-pack `--target web` glue uses both at module load AND for every Rust↔JS string round-trip (e.g. `Dsp::buffer_names()`), so a no-op stub silently corrupts strings to `""`. **Must be imported before** the wasm-pack glue — keep `import "./worklet-polyfills"` first in `dsp-worklet.ts`.

### DSP crate layout (`crates/dsp/src/`)

Seven modules. Each pipeline stage owns its state struct; `lib.rs` is a thin orchestrator.

- **`lib.rs`** — `Dsp` struct (the wasm-bindgen surface), short `process()` that sequences the stages, plus integration tests.
- **`buffers.rs`** — `Buffers` struct: 14 named output `Vec<f32>` fields with **camelCase Rust field names** (`bufferAcf`, `rmsLow`, `onsetAcfEnhanced`, `dspPerf`, etc., via `#[allow(non_snake_case)]`) so the Rust field, the registry-lookup match arm, the worklet message field, and the FeatureStore key are all the same string. `Buffers::get(name) -> Option<&[f32]>` and `Buffers::descriptors() -> Vec<(&'static str, usize)>` are the only string-keyed entry points; stages use direct field access. `dspPerf` is diagnostic data — `[totalMs, freqHz]` EMAs from `crates/dsp/src/perf.rs`, not a visual time-series signal. Sub-ms effective resolution is recovered by EMA smoothing because the underlying `now_us()` source (browser `performance.now()` or `Date.now()`) is typically 1 ms quantized inside `AudioWorkletGlobalScope`.
- **`spectrum.rs`** — `SpectrumState`: windowed real FFT (`realfft`), magnitude → dBFS → normalized [0,1] → temporally smoothed (α derived from `dsp.smoothingTauSecs`). `mag_scale = 2/sum(w)` so a unit-amplitude sine peaks at ~1.0. The window is an **asymmetric Hann** built by `build_window(n, fall)`: a rising half-Hann over the first `n - fall` samples, a falling half-Hann over the last `fall`. `fall = n/2` (the `dsp.windowFall` default, clamped) is the ordinary symmetric Hann; shorter values move the peak next to the NEWEST sample, trading spectral leakage for onset latency — see `docs/latency.md`. `sum(w)` and `sum(w²)` are invariant in `fall`, so amplitude calibration never moves. Bin 0 (DC) dropped from output. Same FFT also produces low/mid/high-band RMS via Parseval-correct band-energy summation. Returns `(low_rms, mid_rms, high_rms, flux)` per call where `flux` is the spectral-flux onset signal.
- **`acf.rs`** — `AcfState`: generalized autocorrelation (Percival & Tzanetakis 2014 §II-B.2, `|X|^0.5` magnitude compression) on the onset history → smoothes along the lag axis with a Gaussian kernel (σ in lag bins, configurable via `dsp.acfSmoothingSigma`) → harmonic-enhanced ACF (sum of acf[τ] + acf[2τ] + acf[4τ]). Smoothing happens **before** harmonic enhancement so the enhanced output inherits the lag-axis broadening. Module also hosts the free functions `compute_gen_acf`, `compute_harmonic_enhanced`, `autocorrelate` (time-domain, used for `bufferAcf`), `bin_for_hz`.
- **`autogain.rs`** — `AutoGain`: scalar running-peak normalizer with `exp(-dt/τ)` decay, one instance per normalized channel (`rms`, `rmsLow`/`Mid`/`High`, `onset`). τ shared via the `autoGain` param.
- **`perf.rs`** — `now_us()` plus the EMA counters behind the `dspPerf` buffer.
- **`beat.rs`** — `BeatState`: comb-scored tempo search → phase comb → EMA smoothing → beat outputs. Diagram below.

### Beat detection pipeline (`beat.rs`)

All of this lives in `BeatState::process`; there are no separate candidate-picking
or TEA-accumulator stages.

```
onset_acf_enhanced ──comb sum per lag τ ∈ [tau_min, tau_max)──▶ tau_scores[τ]
                      (Σ acf[k·τ] for k ≥ 1, then ÷ √taps so
                       short lags don't win on tap count alone)
                                  │ argmax → best_tau
                                  ▼
                      parabolic_refine(tau_scores) → sub-lag best_tau
                                  │
                                  ▼
                      EMA (tea_alpha, τ = dsp.teaTauSecs) → tau_smoothed
                                  │
                                  ▼
   onset ──comb per phase φ ∈ [0, round(tau))──▶ phase_score_inst[φ]
             (Σ onset[p] − onset[p−1] at p = last − φ − k·tau,
              i.e. scores the RISE at each grid position)
                                  │ argmax → phase_measured
                                  ▼
      phase_pred = phase_smoothed + frames elapsed, wrapped at tau;
      delta = shortest signed arc to phase_measured;
      phase_smoothed = phase_pred + phase_correction_alpha · delta
                       (τ = dsp.phaseLock)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            write_beat_outputs          update_beat_pulses
            beatGrid  = [tau, phase,    beatPulses (4 saws, phase within
                         confidence]                1×/2×/4×/8× period cycle)
            beatState = [bpm, confidence, 0, 0]
```

- **Tempo bounds are lags, not BPM.** `tau_min = 30`, `tau_max = 100`, hardcoded in `BeatState::new` — there are no `BEAT_TRACKER_*_BPM` constants. At 48 kHz with hop 1024 (dt ≈ 21.3 ms) that window is roughly **28–94 BPM**, so faster music is tracked at a sub-harmonic.
- **`confidence` is a placeholder.** `confidence_smoothed` is assigned a literal `1.0` every frame (there's a TODO listing peak-ratio and ACF std-dev as candidates), so `beatGrid[2]` and `beatState[1]` are constant and carry no information yet.
- **No NaN sentinel.** Outputs are plain zeros before the first lock and during silence (`beatState[0]` is `0.0` whenever `tau_smoothed <= 0`). Renderers should treat 0, not NaN, as "no beat yet".
- **`parabolic_refine` requires a concave-down bracket.** It returns 0 unless `denom < 0`, so a monotone or concave-up triple contributes no sub-lag offset instead of a bogus one.

### Rendering path (`src/render/`, `src/store/`, `src/App.ts`)

- `Scene.ts` creates a `WebGPURenderer` (from `three/webgpu`) — **not** the default WebGL renderer. `await renderer.init()` is required before first render.
- `App.ts` is a thin orchestrator: scene + camera + `CameraRig` presets, keyboard/resize listeners, RAF loop, and worklet message routing. The `features` handler is a 3-line loop that copies `msg.buffers[name]` into the FeatureStore for every name. Renderers pick up the buffers on the next update tick via `source: () => store.get(name)` callbacks. App owns no per-feature state.
- **Latency instrumentation.** `LatencyProbe` (`src/audio/LatencyProbe.ts`) is fed from three points in `App`: the `features` handler, `onConsume(t4)` right after `components.update()`, and `onRendered()` in the `renderAsync` continuation. It feeds the `LAT` plot in the perf HUD continuously, and `L` fires a one-shot inaudible impulse probe that prints the software-path breakdown to the console. `docs/latency.md` has the full budget and the two out-of-process experiments (acoustic loopback for capture latency, slo-mo camera for photons).
- `DebugView` (`src/render/debug/DebugView.ts`) is the visualization layer — owns the line/bar renderers plus their position/scale wiring, and pulls in `DebugGrid`, `DebugLabels`, `BeatGridMarkers`, `StaticBeatGridMarkers`, and `BeatPulseSquares`. Driven by a static `LINE_COLORS` table specifying per-buffer color, render type (line vs bar), and x-scale. No `applyConfigured` — renderers self-init from their source on first non-empty buffer (via `TimeSeriesLineRenderer.update()`'s zero-length early return).
- `TimeSeriesRenderer` (abstract base) owns the per-frame loop: read source, hand each sample to `writeOne(i, n, x, y)` where `x = i / (n-1)` (or `log2(i+1) / log2(n)` if `scale: "logx"` was set). Subclasses `TimeSeriesLineRenderer` (Line strip, vertex colors) and `TimeSeriesBarRenderer` (instanced quads, per-instance color) render the data — each Object3D output is in `[0, 1] × [0, 1]` space, positioned/scaled by the consumer. Per-sample color is the configured flat color by default; opt in to value-modulated brightness with `colorByValue: true` (color = `configuredColor × clamp(|y|, 0, 1)`, signals fading to 0 fade to black).
- **Autogain** lives in `crates/dsp/src/autogain.rs` (Rust side) — scalar per-channel running peak with `exp(-dt/τ)` decay, applied **before** values are pushed into `rms`, `rmsLow`/`Mid`/`High`, and `onset` history buffers. τ is shared across the five channels via the `autoGain` param. There is no TS-side autogain anymore — renderers consume the already-normalized buffers directly.
- `CameraRig` supports named presets, eased tweens (`goTo`), and an optional procedural controller; `goTo` returns a promise that resolves on tween completion.
- `FeatureStore` is intentionally a thin `Map<string, Float32Array>` — buffers in, buffers out; no events. Missing keys return a shared empty `Float32Array` so renderers can no-op safely before the first features message arrives.

### Worklet ↔ main message protocol

Two main → worklet message types and one worklet → main:

- **`features`** (worklet → main, ~47 Hz): `{ type, buffers: { [name]: Float32Array }, t, hopMs }`. The buffer name set comes from `dsp.buffer_names()`, cached at boot/configure. Each frame the worklet builds the dict by calling `dsp.get_buffer(name)` for every cached name, posting them all in one transfer-list batch. `t` is the **AudioContext** time of the newest sample in this frame's window; `hopMs` is `performance.now()` in the worklet at publish (same time origin as the document). Both exist for latency measurement — see `docs/latency.md` and `src/audio/LatencyProbe.ts`.
- **`configure`** (main → worklet): `{ type, windowSize, rmsHistoryLen }`. Triggers `dsp.free()` + fresh `Dsp::new(...)`, re-applies cached params, refreshes `bufferNames`.
- **`param`** (main → worklet): `{ type, key: string, value: number }`. `hopSize` is intercepted (it controls the worklet's own dispatch cadence, not Dsp); everything else is forwarded to `dsp.set_param(key, value)` and cached worklet-side so `applyConfigure` can re-apply it across rebuilds.

There is **no `configured` event and no `sync` round-trip** — App reads sizes from `Float32Array.length` on the per-frame features messages.

### Wasm-bindgen surface (`Dsp`)

Five methods total:
- `new(window_size, sample_rate, hop_size, rms_history_len) -> Dsp`
- `process(input: &[f32])`
- `get_buffer(name: &str) -> Vec<f32>` — string-keyed buffer accessor (Float32Array on the JS side).
- `buffer_names() -> Vec<String>` — list all 14 buffer keys; called once per configure to populate the worklet's name cache.
- `set_param(key: &str, value: f32)` — recognized keys: `windowFall`, `smoothingTauSecs`, `onsetSmoothingTauSecs`, `teaTauSecs`, `acfSmoothingSigma`, `acfDecay`, `dbFloor`, `phaseLock`, `autoGain`. Unknown keys silently ignored.

wasm-bindgen with `--target web` exports method names verbatim (snake_case). JS calls are `dsp.get_buffer("...")`, `dsp.set_param("...", v)` — NOT `getBuffer`/`setParam`.

### Param store & WorkletBridge (`src/params/`)

- `ParamStore` holds `dsp.*` keys with continuous/discrete schemas (`schemas.ts`).
- `WorkletBridge` subscribes to changes and forwards them to the worklet. `windowSize`/`rmsHistoryLen` are **reconfig** (rebuild Dsp). Hot keys (in-place `dsp.set_param` calls): `hopSize`, `windowFall`, `smoothingTauSecs`, `onsetSmoothingTauSecs`, `dbFloor`, `teaTauSecs`, `acfSmoothingSigma`, `acfDecay`, `phaseLock`, `autoGain`. Add new hot keys to `HOT_KEYS` AND ensure the worklet handles them (cache the value so `applyConfigure` re-applies it across rebuilds). `tests/params/WorkletBridge.test.ts` pins the bootstrap message count — bump it when the list changes.
- `bridge.bootstrap()` is called once per page-lifetime in `main.ts`. On HMR the bridge is recreated but `bootstrap()` does **not** re-run — the worklet keeps its current Dsp + params; the App's new renderers come up via lazy init on the next features message.

### Build pipeline notes

- `vite-plugin-wasm` + `vite-plugin-top-level-await` handle `import "...?url"` of wasm and worklet sources. `vite.config.ts` also applies `wasm()` to the `worker` plugin chain (audio worklets count). The wasm URL and worklet URL are imported via `?url` / `?worker&url` and passed to `addModule` / `WebAssembly.compileStreaming`.
- **Three.js must resolve to exactly one copy.** `three` → `three.module.js` (core) while `three/webgpu` + `three/tsl` → `three.webgpu.js`, which is a *superset* that re-exports all of core. Importing from both loads two copies of every class, so a `DirectionalLight` from `three` is a different class than the one the WebGPU node renderer registers light nodes against → "Light node not found". `vite.config.ts` fixes this with an exact-match alias `/^three$/` → `three/webgpu` (plus `resolve.dedupe`), so `three/webgpu`, `three/tsl`, and `three/examples/*` pass through untouched. A "Multiple instances of Three.js" warning still appears once under vitest, which does not go through that alias; the app itself is clean.
- `src/wasm-pkg/` is generated; never edit by hand and never check it in.

## Conventions

- Comments explain **why** (non-obvious invariants, hidden constraints, pitfalls like the polyfill ordering or `mag_scale` derivation), not what. Existing code is the reference for tone — match it.
- TypeScript is strict (`noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`). Keep it that way.
- **Buffer keys are one canonical string each.** The same string is used as the Rust struct field (camelCase via `#[allow(non_snake_case)]`), the `Buffers::get` match arm, the worklet message field, and the FeatureStore key. Adding a new buffer is a 3-line edit in `buffers.rs` + the consumer.
- Specs and plans live in `docs/superpowers/specs/` and `docs/superpowers/plans/`. The roadmap (`ROADMAP.md`) tracks deferred work — check it before starting a new feature.

## Pitfalls / non-obvious invariants

- **`TextDecoder` polyfill is load-bearing.** It's only installed if `globalThis.TextDecoder` is missing in the AudioWorkletGlobalScope (Chrome before ~116). If installed, it must implement real UTF-8 decoding — a no-op stub returns `""` and silently corrupts every Rust→JS string (including all 14 buffer names from `buffer_names()`, which makes the visualizer go dark with no obvious error).
- **HMR teardown.** `App.dispose()` clears `port.onmessage` so in-flight features messages stop landing on a half-disposed app. The new App instance re-wires onmessage and lazy-inits renderers from the next features message — there is no `sync` round-trip, no `applyConfigured` to re-fire.
- **Tempo bounds are lag counts, not BPM.** `tau_min = 30` / `tau_max = 100`, hardcoded in `BeatState::new`. At 48 kHz with hop 1024 that is ≈ 28–94 BPM; faster music locks to a sub-harmonic. Changing the hop moves the BPM window.
- **Beat outputs are zero, not NaN, when there's no beat.** Before the first lock and during silence `tau_smoothed`/`phase_smoothed` are `0.0` and `beatState[0]` is `0.0`. Renderers must treat 0 as "no value" — an older design used NaN as the sentinel and that is no longer true.
- **`port.onmessage = null` then `= handler` does NOT replay queued messages.** Anything posted between dispose and the next `start()` is dropped. Counted on for HMR; don't try to recover it.
- **Worklet `process()` posts only `features`.** The old `configured` and `sync` events are gone. Don't reintroduce them — App reads sizes from the per-frame `Float32Array.length` and lazy-inits.
