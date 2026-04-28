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
- `dsp-worklet.ts` is the AudioWorklet processor. It receives a precompiled `WebAssembly.Module` via `processorOptions` (compiled on the main thread, passed in — workers cannot `compileStreaming` their own URL). It maintains a sliding 2048-sample window and fires the FFT every 1024 samples (50% overlap → ~47 Hz update rate at 48 kHz). Output buffers are transferred (zero-copy) to the main thread via `postMessage`.
- `worklet-polyfills.ts` stubs `TextDecoder`/`TextEncoder` in `AudioWorkletGlobalScope` because wasm-pack `--target web` glue instantiates a `TextDecoder` at module load. **Must be imported before** the wasm-pack glue — keep the `import "./worklet-polyfills"` line first in `dsp-worklet.ts`.
- `crates/dsp/src/lib.rs` (`Dsp` struct) does Hann-windowed real FFT (`realfft`), magnitude → dBFS → normalized [0,1] → temporally smoothed (α=0.2). `mag_scale = 2/sum(hann)` so a unit-amplitude sine peaks at ~1.0. Bin 0 (DC) is dropped from the output spectrum. `rms_history` is a 512-sample ring of per-window RMS used by the RMS lane and (planned) beat detection.

### Rendering path (`src/render/`, `src/store/`, `src/App.ts`)

- `Scene.ts` creates a `WebGPURenderer` (from `three/webgpu`) — **not** the default WebGL renderer. `await renderer.init()` is required before first render.
- `App.ts` is the composition root: builds scene/camera/rig, registers four camera presets (`1`=front, `2`=side, `3`=spectrum, `4`=rms; space toggles front↔side), seeds `FeatureStore` with empty buffers, wires three `LineRenderer`s (waveform / spectrum / rms), starts the worklet, and runs the render loop. The worklet's `port.onmessage` writes new buffers into the store; `LineRenderer.update()` reads them by reference each frame.
- `LineRenderer` reads its `Float32Array` via a `source: () => Float32Array` callback (so the App can swap buffers in the store without re-wiring), maps each sample through a `LineLayoutFn` into a `BufferAttribute` with `DynamicDrawUsage`, and reallocates if buffer length changes.
- `LineLayouts.ts` provides `linearLayout` (used for waveform & RMS) and `logSpectrumLayout` (log2 x-axis for the spectrum).
- `CameraRig` supports named presets, eased tweens (`goTo`), and an optional procedural controller; `goTo` returns a promise that resolves on tween completion.
- `FeatureStore` is intentionally a thin `Map<string, Float32Array>` — buffers in, buffers out; no events.

### Build pipeline notes

- `vite-plugin-wasm` + `vite-plugin-top-level-await` handle `import "...?url"` of wasm and worklet sources. `vite.config.ts` also applies `wasm()` to the `worker` plugin chain (audio worklets count). The wasm URL and worklet URL are imported via `?url` / `?worker&url` and passed to `addModule` / `WebAssembly.compileStreaming`.
- `resolve.dedupe: ["three"]` is intentional — without it Three.js double-loads (the `three/webgpu` subpath imports its own copy). A "Multiple instances of Three.js" warning is a known leftover and is on the roadmap.
- `src/wasm-pkg/` is generated; never edit by hand and never check it in.

## Conventions

- Comments explain **why** (non-obvious invariants, hidden constraints, pitfalls like the polyfill ordering or `mag_scale` derivation), not what. Existing code is the reference for tone — match it.
- TypeScript is strict (`noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`). Keep it that way.
- Specs and plans live in `docs/superpowers/specs/` and `docs/superpowers/plans/`. The roadmap (`ROADMAP.md`) tracks deferred work — check it before starting a new feature; v3 (autocorrelation, beat detection from `rms_history`) is the current focus.
