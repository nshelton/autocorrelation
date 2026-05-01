# DSP Performance Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure runtime in microseconds for the major sections of `Dsp::process`, ship those numbers through the existing worklet `features` message, and display them in the debug labels without adding a separate message protocol.

**Architecture:** Add a fixed-size `dspPerfUs` output buffer to the DSP buffer registry. The buffer contains the latest per-hop timing sample in microseconds, ordered by a small Rust-side metric index table. `Dsp::process` records timestamps around each major stage using a wasm-safe timer. The existing AudioWorklet loop already sends every `buffer_names()` entry, so the new buffer reaches `FeatureStore` automatically. `DebugLabels` reads `store.get("dspPerfUs")` and formats a label.

**Tech Stack:** Rust + wasm-bindgen, TypeScript, AudioWorklet. No new worklet message type, no `configured`/`sync` event, no generated `src/wasm-pkg/` edits by hand.

---

## Timing Contract

Add one new public buffer key:

- `dspPerfUs`: latest DSP section runtimes, in microseconds, as `Float32Array` values.

Initial metric order:

1. `total` — full measured `Dsp::process` wall time.
2. `inputRms` — waveform copy, full-window RMS, and RMS history push.
3. `spectrum` — `SpectrumState::process` plus low/mid/high RMS and onset history pushes.
4. `onsetAcf` — generalized onset ACF, lag smoothing, and harmonic enhancement.
5. `beat` — candidate picking, phase scoring, TEA update, beat outputs, and beat pulses.
6. `bufferAcf` — time-domain waveform autocorrelation.

Notes:

- These timings are live diagnostics, not benchmark-grade measurements. The timer calls themselves perturb the result slightly.
- `total` measures Rust DSP work only. It does not include JS-side `dsp.get_buffer(...)` copies, transfer-list construction, `postMessage`, main-thread store updates, or rendering.
- Use microsecond differences derived from a high-resolution monotonic timer where available. Store only elapsed durations, never absolute timestamps.

---

## Task 1: Add a wasm-safe timing helper

**Files:**

- Create: `crates/dsp/src/perf.rs`
- Modify: `crates/dsp/src/lib.rs`
- Modify: `crates/dsp/Cargo.toml`

- [ ] Add a target-specific wasm dependency on `js-sys` in `Cargo.toml` so native `cargo test -p dsp` does not need to compile JS bindings for non-wasm targets.
- [ ] Create a `perf.rs` module that owns the instrumentation constants and timer helper.
- [ ] Define `PERF_METRIC_COUNT: usize = 6`.
- [ ] Define metric indexes for `total`, `inputRms`, `spectrum`, `onsetAcf`, `beat`, and `bufferAcf`.
- [ ] Optionally define `PERF_METRIC_NAMES: [&str; PERF_METRIC_COUNT]` for Rust tests/documentation, even if the names are mirrored in TypeScript for display.
- [ ] Implement `now_us() -> f64` behind `cfg` gates:
  - `wasm32`: call `globalThis.performance.now() * 1000.0`, with `Date.now() * 1000.0` as a low-resolution fallback.
  - non-`wasm32`: use `std::time::Instant` so `cargo test -p dsp` can compile and verify the instrumentation path.
- [ ] Add a tiny helper for elapsed durations that clamps negative clock deltas to `0.0` before casting to `f32`.
- [ ] Wire `mod perf;` into `lib.rs`.

---

## Task 2: Add `dspPerfUs` to the DSP buffer registry

**Files:**

- Modify: `crates/dsp/src/buffers.rs`
- Test: `crates/dsp/src/buffers.rs`, `crates/dsp/src/lib.rs`

- [ ] Add a camelCase field to `Buffers`: `pub dspPerfUs: Vec<f32>`.
- [ ] Initialize it as `vec![f32::NAN; PERF_METRIC_COUNT]` so uninitialized timing values have the same “no value” sentinel style as beat outputs.
- [ ] Add `("dspPerfUs", self.dspPerfUs.len())` to `descriptors()`.
- [ ] Add the `"dspPerfUs"` match arm to `Buffers::get`.
- [ ] Update buffer-registry tests so the canonical key list includes `dspPerfUs`.
- [ ] Add or update a `Dsp` test asserting `get_buffer("dspPerfUs")` returns `PERF_METRIC_COUNT` entries.

Expected result: the worklet’s existing `this.bufferNames = this.dsp.buffer_names()` path discovers the new buffer automatically.

---

## Task 3: Instrument `Dsp::process` around major sections

**Files:**

- Modify: `crates/dsp/src/lib.rs`
- Test: `crates/dsp/src/lib.rs`

- [ ] Take a timestamp at the start of `Dsp::process`.
- [ ] Wrap the current process body into these measured sections, preserving the current stage order and numerical behavior:
  - input waveform copy + full-window RMS + RMS history push.
  - spectrum processing + band RMS/onset pushes.
  - onset ACF processing.
  - beat processing.
  - waveform buffer ACF.
- [ ] Write each section duration into `self.buffers.dspPerfUs[...]` at the end of the stage.
- [ ] Write `total` after all sections complete.
- [ ] Keep instrumentation allocation-free inside `process`.
- [ ] Clamp negative durations to zero and store finite `f32` microsecond values.
- [ ] Add a unit test that calls `process` and verifies all `dspPerfUs` entries are finite and non-negative.
- [ ] Do not move any DSP work solely to make instrumentation easier; timings should reflect the current pipeline shape.

Implementation invariant: `Dsp::process` remains the only Rust orchestrator; stage modules still own their state and algorithms.

---

## Task 4: Send timings through the existing worklet feature path

**Files:**

- Inspect/usually no change: `src/audio/dsp-worklet.ts`
- Inspect/usually no change: `src/App.ts`
- Test manually in browser after `npm run wasm`

- [ ] Confirm no explicit TypeScript worklet changes are needed because `dspPerfUs` is included in `buffer_names()`.
- [ ] Confirm `src/audio/dsp-worklet.ts` posts `dspPerfUs` in the same `features.buffers` object as every other DSP buffer.
- [ ] Confirm `src/App.ts` continues to store it via the existing generic loop over `Object.entries(msg.buffers)`.
- [ ] Do not add a new worklet-to-main message type for this first pass.
- [ ] Do not add a `configured` or `sync` round-trip.

Expected result: after one feature frame, consumers can read `store.get("dspPerfUs")`.

---

## Task 5: Display the timings in debug labels

**Files:**

- Modify: `src/render/DebugLabels.ts`

- [ ] Add a new dynamic `TextLabel` for DSP performance, or extend `configSummary` if the line fits cleanly.
- [ ] Mirror the metric order in a small TypeScript constant near `DebugLabels`, for example: `total`, `inputRms`, `spectrum`, `onsetAcf`, `beat`, `bufferAcf`.
- [ ] Format values as microseconds, e.g. `dsp: total=420µs spec=120µs acf=80µs beat=60µs bufacf=90µs`.
- [ ] Use the existing 250 ms label update cadence.
- [ ] Display `--` when `dspPerfUs` is missing, too short, `NaN`, or non-finite.
- [ ] Keep `App` free of per-feature special cases; `DebugLabels` should read from `FeatureStore` like the existing beat/config summaries.
- [ ] If the raw label is too jumpy, add UI-only EMA smoothing inside `DebugLabels`; do not smooth the DSP buffer itself in the first pass.

---

## Task 6: Documentation updates

**Files:**

- Modify: `CLAUDE.md`
- Optionally modify related specs/plans if they enumerate the buffer set.

- [ ] Update the DSP buffer count from 15 to 16 anywhere it is stated as an invariant.
- [ ] Add `dspPerfUs` to the canonical buffer-key list where appropriate.
- [ ] Note that `dspPerfUs` is diagnostic data, not a visual time-series signal.
- [ ] Keep the generated `src/wasm-pkg/` directory out of manual edits and version control.

---

## Task 7: Validation

**Commands:**

- [ ] Run `cargo test -p dsp` from the repo root.
- [ ] Run `npm run wasm` because `crates/dsp/` changed.
- [ ] Run `npm test`.
- [ ] Run `npm run build`.
- [ ] Manual browser check:
  - Start the app.
  - Use the internal test source (`T`) or tab audio.
  - Confirm the DSP performance label appears and updates.
  - Confirm values are finite, non-negative, and plausibly below the hop budget.
  - At default 48 kHz / 1024 hop, compare `total` against the ~21.3 ms audio deadline.

---

## Deferred Follow-ups

- Add a second worklet-side `workletPerfUs` buffer or message field if we need to measure `dsp.get_buffer(...)`, transfer-list construction, and `postMessage` overhead separately.
- Add a `dsp.perfEnabled` ParamStore toggle only if the timer import overhead becomes a problem.
- Add min/mean/max rolling summaries in the frontend if the single latest frame is too noisy for practical debugging.
- Add development-only warnings when `total` approaches a configurable percentage of the hop deadline.
