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

- `AudioSource.ts` exposes three `AudioSourceBundle` factories: `createMicSource` (getUserMedia, no AGC/NS/EC), `createTabSource` (getDisplayMedia, drops video tracks immediately — picker requires both audio+video to surface tab-audio capture), `createTestSource` (in-process oscillator, no permissions — used for fast iteration; press `T` before any source is started).
- `dsp-worklet.ts` is the AudioWorklet processor. It receives a precompiled `WebAssembly.Module` via `processorOptions` (compiled on the main thread, passed in — workers cannot `compileStreaming` their own URL). It maintains a sliding window (`windowSize`, default 2048) and fires the FFT every `hopSize` samples (default 1024 = 50 % overlap → ~47 Hz update rate at 48 kHz). Output buffers go zero-copy via `postMessage` transfer list.
- `worklet-polyfills.ts` provides UTF-8 `TextDecoder` / `TextEncoder` polyfills for `AudioWorkletGlobalScope` in browsers that lack them. wasm-pack `--target web` glue uses both at module load AND for every Rust↔JS string round-trip (e.g. `Dsp::buffer_names()`), so a no-op stub silently corrupts strings to `""`. **Must be imported before** the wasm-pack glue — keep `import "./worklet-polyfills"` first in `dsp-worklet.ts`.

### DSP crate layout (`crates/dsp/src/`)

Five modules. Each pipeline stage owns its state struct; `lib.rs` is a thin orchestrator.

- **`lib.rs`** — `Dsp` struct (the wasm-bindgen surface), short `process()` that sequences the stages, plus integration tests.
- **`buffers.rs`** — `Buffers` struct: 15 named output `Vec<f32>` fields with **camelCase Rust field names** (`bufferAcf`, `rmsLow`, `onsetAcfEnhanced`, etc., via `#[allow(non_snake_case)]`) so the Rust field, the registry-lookup match arm, the worklet message field, and the FeatureStore key are all the same string. `Buffers::get(name) -> Option<&[f32]>` and `Buffers::descriptors() -> Vec<(&'static str, usize)>` are the only string-keyed entry points; stages use direct field access.
- **`spectrum.rs`** — `SpectrumState`: Hann-windowed real FFT (`realfft`), magnitude → dBFS → normalized [0,1] → temporally smoothed (α derived from `dsp.smoothingTauSecs`). `mag_scale = 2/sum(hann)` so a unit-amplitude sine peaks at ~1.0. Bin 0 (DC) dropped from output. Same FFT also produces low/mid/high-band RMS via Parseval-correct band-energy summation. Returns `(low_rms, mid_rms, high_rms, flux)` per call where `flux` is the spectral-flux onset signal.
- **`acf.rs`** — `AcfState`: generalized autocorrelation (Percival & Tzanetakis 2014 §II-B.2, `|X|^0.5` magnitude compression) on the onset history → smoothes along the lag axis with a Gaussian kernel (σ in lag bins, configurable via `dsp.acfSmoothingSigma`) → harmonic-enhanced ACF (sum of acf[τ] + acf[2τ] + acf[4τ]). Smoothing happens **before** harmonic enhancement so the enhanced output inherits the lag-axis broadening. Module also hosts the free functions `compute_gen_acf`, `compute_harmonic_enhanced`, `autocorrelate` (time-domain, used for `bufferAcf`), `bin_for_hz`.
- **`beat.rs`** — `BeatState`: tempo candidate picking → phase scoring → TEA accumulator → beat outputs. Diagram below.

### Beat detection pipeline (`beat.rs`)

```
onset_acf_enhanced ──pick_candidates──▶  candidates
                                        (top-N, [lag, mag, sharpness] stride 3,
                                         peak-spaced, parabolic-refined)
                                              │
                                              ▼
                onset ──score_candidates──▶ pulse_x, pulse_v, pulse_phi per candidate
                                            (Φ₁ + Φ₂ + Φ₁.₅ pulse trains, max + variance)
                                              │ pick winner: argmax(X_norm + V_norm)
                                              ▼
                                       period_inst, phase_inst, score_inst (per-frame)
                                              │
                                              ▼
                                       update_tea
                                       (Gaussian-smear vote at period_inst into TEA;
                                        EMA decay; argmax + parabolic refine
                                        → tau_smoothed, phase_smoothed)
                                              │
                                              ▼
                  write_beat_outputs    update_beat_pulses
                  ▼                     ▼
                  beat_grid             beat_pulses
                  beat_state            (4 saws, brightness = phase
                                         within 1×/2×/4×/8× period cycle)
```

- **`candidates`** is the public top-N tempo candidate list. `[lag, mag, sharpness]` triples, stride 3, length `3 * MAX_PEAKS` (capacity defined in `beat.rs`, currently being tuned). NaN means empty slot. `sharpness = -(y₀ - 2y₁ + y₂)` from parabolic interpolation — large for narrow real beats, small for broad smeared lobes. Consumed by both the scorer and `PeakMarkers`.
- **`score_phase_for_tau`** is the paper's pulse-train scoring — three combs (Φ₁ at k·τ weight 1.0, Φ₂ at k·2τ weight 0.5, Φ₁.₅ at (k+0.5)·τ weight 0.5) with N=4 pulses each. Returns max correlation (X) and variance across phases (V) for a given lag.
- **`tea`** is the Tempo Estimate Accumulator — per-lag EMA across frames with a Gaussian-smeared vote per frame (σ = `dsp.teaSigma` in lag bins, τ = `dsp.teaTauSecs` for the EMA). Smooths `period_inst` jitter.
- **Tempo bounds:** lags clamped to BPM range [`BEAT_TRACKER_MIN_BPM`, `BEAT_TRACKER_MAX_BPM`] = [40, 220]. At default sr/hop (48000/1024) that's tau_min ≈ 13, tau_max ≈ 70 lags. Music outside the range gets reported at the nearest octave inside it.
- **Beat phase NaN:** when no candidate scores positively (silence / no rhythmic content), `period_inst`/`phase_inst`/`tau_smoothed`/`phase_smoothed` are all NaN. `write_beat_outputs` and `update_beat_pulses` NaN-fill their outputs. Renderers expect NaN as the "no value" sentinel and should no-op in that case.

### Rendering path (`src/render/`, `src/store/`, `src/App.ts`)

- `Scene.ts` creates a `WebGPURenderer` (from `three/webgpu`) — **not** the default WebGL renderer. `await renderer.init()` is required before first render.
- `App.ts` is a thin orchestrator: scene + camera + `CameraRig` presets, keyboard/resize listeners, RAF loop, and worklet message routing. The `features` handler is a 3-line loop that copies `msg.buffers[name]` into the FeatureStore for every name. Renderers pick up the buffers on the next update tick via `source: () => store.get(name)` callbacks. App owns no per-feature state.
- `DebugView` (`src/render/DebugView.ts`) is the visualization layer — owns the line/bar renderers + `PeakMarkers`, plus their position/scale wiring. Driven by a static `LINE_COLORS` table specifying per-buffer color, render type (line vs bar), and autogain. No `applyConfigured` — renderers self-init from their source on first non-empty buffer (via `TimeSeriesLineRenderer.update()`'s zero-length early return).
- `BeatDebugView.ts` is currently a stub awaiting a lazy-init refactor — sub-renderers (grid, scrolling grid, pulse squares) are not yet rewired.
- `TimeSeriesRenderer` (abstract base) owns the per-frame loop: read source, optional autogain, hand each sample to `writeOne(i, n, x, y)` where `x = i / (n-1)` (or `log2(i+1) / log2(n)` if `scale: "logx"` was set). Subclasses `TimeSeriesLineRenderer` (Line strip) and `TimeSeriesBarRenderer` (instanced quads) render the data — each Object3D output is in `[0, 1] × [0, 1]` space, positioned/scaled by the consumer.
- **Autogain** lives in `TimeSeriesRenderer.applyAutoGain` — per-channel running peak with `exp(-dt/τ)` decay, normalizes raw → auto. Auto buffer advances **in lockstep with raw** (detected via reference change + `raw[N-2] === lastRaw[N-1]` check) so older auto entries keep their time-of-arrival denom — the line doesn't pump amplitude when peaks shift, and scroll rate matches an un-gained line. Tab-resume gaps trigger a full re-normalize.
- `PeakMarkers` renders markers on the autocorr lane from the `candidates` buffer.
- `CameraRig` supports named presets, eased tweens (`goTo`), and an optional procedural controller; `goTo` returns a promise that resolves on tween completion.
- `FeatureStore` is intentionally a thin `Map<string, Float32Array>` — buffers in, buffers out; no events. Missing keys return a shared empty `Float32Array` so renderers can no-op safely before the first features message arrives.

### Worklet ↔ main message protocol

Two main → worklet message types and one worklet → main:

- **`features`** (worklet → main, ~47 Hz): `{ type, buffers: { [name]: Float32Array } }`. The buffer name set comes from `dsp.buffer_names()`, cached at boot/configure. Each frame the worklet builds the dict by calling `dsp.get_buffer(name)` for every cached name, posting them all in one transfer-list batch.
- **`configure`** (main → worklet): `{ type, windowSize, rmsHistoryLen }`. Triggers `dsp.free()` + fresh `Dsp::new(...)`, re-applies cached params, refreshes `bufferNames`.
- **`param`** (main → worklet): `{ type, key: string, value: number }`. `hopSize` is intercepted (it controls the worklet's own dispatch cadence, not Dsp); everything else is forwarded to `dsp.set_param(key, value)` and cached worklet-side so `applyConfigure` can re-apply it across rebuilds.

There is **no `configured` event and no `sync` round-trip** — App reads sizes from `Float32Array.length` on the per-frame features messages.

### Wasm-bindgen surface (`Dsp`)

Five methods total:
- `new(window_size, sample_rate, hop_size, rms_history_len) -> Dsp`
- `process(input: &[f32])`
- `get_buffer(name: &str) -> Vec<f32>` — string-keyed buffer accessor (Float32Array on the JS side).
- `buffer_names() -> Vec<String>` — list all 15 buffer keys; called once per configure to populate the worklet's name cache.
- `set_param(key: &str, value: f32)` — recognized keys: `smoothingTauSecs`, `onsetSmoothingTauSecs`, `teaTauSecs`, `teaSigma`, `acfSmoothingSigma`, `dbFloor`. Unknown keys silently ignored.

wasm-bindgen with `--target web` exports method names verbatim (snake_case). JS calls are `dsp.get_buffer("...")`, `dsp.set_param("...", v)` — NOT `getBuffer`/`setParam`.

### Param store & WorkletBridge (`src/params/`)

- `ParamStore` holds `dsp.*` keys with continuous/discrete schemas (`schemas.ts`).
- `WorkletBridge` subscribes to changes and forwards them to the worklet. `windowSize`/`rmsHistoryLen` are **reconfig** (rebuild Dsp). Hot keys (in-place `dsp.set_param` calls): `hopSize`, `smoothingTauSecs`, `onsetSmoothingTauSecs`, `dbFloor`, `teaTauSecs`, `teaSigma`, `acfSmoothingSigma`. Add new hot keys to `HOT_KEYS` AND ensure the worklet handles them.
- `dsp.autoGain` is **TS-only** — controls the autogain τ in `TimeSeriesRenderer`, never reaches the worklet. Don't add it to `HOT_KEYS`.
- `bridge.bootstrap()` is called once per page-lifetime in `main.ts`. On HMR the bridge is recreated but `bootstrap()` does **not** re-run — the worklet keeps its current Dsp + params; the App's new renderers come up via lazy init on the next features message.

### Build pipeline notes

- `vite-plugin-wasm` + `vite-plugin-top-level-await` handle `import "...?url"` of wasm and worklet sources. `vite.config.ts` also applies `wasm()` to the `worker` plugin chain (audio worklets count). The wasm URL and worklet URL are imported via `?url` / `?worker&url` and passed to `addModule` / `WebAssembly.compileStreaming`.
- `resolve.dedupe: ["three"]` is intentional — without it Three.js double-loads (the `three/webgpu` subpath imports its own copy). A "Multiple instances of Three.js" warning is a known leftover and is on the roadmap.
- `src/wasm-pkg/` is generated; never edit by hand and never check it in.

## Conventions

- Comments explain **why** (non-obvious invariants, hidden constraints, pitfalls like the polyfill ordering or `mag_scale` derivation), not what. Existing code is the reference for tone — match it.
- TypeScript is strict (`noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`). Keep it that way.
- **Buffer keys are one canonical string each.** The same string is used as the Rust struct field (camelCase via `#[allow(non_snake_case)]`), the `Buffers::get` match arm, the worklet message field, and the FeatureStore key. Adding a new buffer is a 3-line edit in `buffers.rs` + the consumer.
- Specs and plans live in `docs/superpowers/specs/` and `docs/superpowers/plans/`. The roadmap (`ROADMAP.md`) tracks deferred work — check it before starting a new feature.

## Pitfalls / non-obvious invariants

- **`candidates` is stride 3.** Triples `[lag, mag, sharpness]`. Length `3 * MAX_PEAKS` (the constant lives in `beat.rs`). Anything iterating peaks must step in 3s. NaN means "empty slot."
- **`TextDecoder` polyfill is load-bearing.** It's only installed if `globalThis.TextDecoder` is missing in the AudioWorkletGlobalScope (Chrome before ~116). If installed, it must implement real UTF-8 decoding — a no-op stub returns `""` and silently corrupts every Rust→JS string (including all 15 buffer names from `buffer_names()`, which makes the visualizer go dark with no obvious error).
- **`MAX_PEAKS` is `pub(crate)` in `beat.rs`** and imported by `buffers.rs`. The two must agree because `Buffers` allocates `candidates` as `vec![f32::NAN; 3 * MAX_PEAKS]` and `BeatState` writes exactly `MAX_PEAKS` triples.
- **HMR teardown.** `App.dispose()` clears `port.onmessage` so in-flight features messages stop landing on a half-disposed app. The new App instance re-wires onmessage and lazy-inits renderers from the next features message — there is no `sync` round-trip, no `applyConfigured` to re-fire.
- **BPM bounds are hard.** `BEAT_TRACKER_MIN_BPM` / `MAX_BPM` = 40 / 220. Music outside this range is reported at the nearest octave inside it.
- **Beat NaN ≠ broken.** Silence / no rhythmic content → `period_inst`, `phase_inst`, `tau_smoothed`, `phase_smoothed` are all NaN; outputs are NaN-filled. Renderers must treat NaN as "no value" and no-op, not default to 0.
- **`port.onmessage = null` then `= handler` does NOT replay queued messages.** Anything posted between dispose and the next `start()` is dropped. Counted on for HMR; don't try to recover it.
- **Worklet `process()` posts only `features`.** The old `configured` and `sync` events are gone. Don't reintroduce them — App reads sizes from the per-frame `Float32Array.length` and lazy-inits.
