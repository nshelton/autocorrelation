# DSP Crate Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `crates/dsp/src/lib.rs` into named pipeline-stage modules and replace the per-buffer wasm-bindgen surface with a single string-keyed buffer registry whose key vocabulary is identical across Rust and JS.

**Architecture:** Five Rust modules (`lib.rs` orchestrator + `buffers.rs` registry + `spectrum.rs` + `acf.rs` + `beat.rs`), each stage owning its own state struct. Public wasm-bindgen surface collapses from 21 typed methods to 5 generic ones (`new`, `process`, `getBuffer`, `bufferDescriptors`, `setParam`). The 15 buffer keys (e.g. `"onsetAcf"`, `"rmsLow"`, `"beatGrid"`) and 3 param keys are camelCase strings used identically as Rust struct field names, lookup keys, message field names, and FeatureStore keys.

**Tech Stack:** Rust + wasm-bindgen, realfft, TypeScript, AudioWorklet.

**Refactor strategy:** Two phases. Phase 1 (Tasks 1-6) is a pure Rust internal refactor — public wasm surface stays bit-identical, JS untouched, every existing test and visual still works after each task. Phase 2 (Tasks 7-13) swaps the wasm boundary and propagates the change to the JS side; old Rust API stays alive as wrappers until Task 11 deletes them.

**Spec:** `docs/superpowers/specs/2026-04-30-dsp-crate-modularization-design.md`.

---

## Task 1: Module skeletons + `push_history` helper

**Files:**
- Create: `crates/dsp/src/buffers.rs`
- Create: `crates/dsp/src/spectrum.rs`
- Create: `crates/dsp/src/acf.rs`
- Create: `crates/dsp/src/beat.rs`
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Create empty module files**

Create `crates/dsp/src/buffers.rs` with:
```rust
//! Output-buffer registry. Single source of truth for the JS-visible buffer
//! key vocabulary.
```

Create `crates/dsp/src/spectrum.rs` with:
```rust
//! FFT + windowing + spectrum smoothing + per-band RMS + spectral flux.
```

Create `crates/dsp/src/acf.rs` with:
```rust
//! Generalized autocorrelation (Percival & Tzanetakis) + harmonic
//! enhancement + time-domain autocorrelate.
```

Create `crates/dsp/src/beat.rs` with:
```rust
//! Candidate picking → phase scoring → TEA → beat outputs.
```

- [ ] **Step 2: Wire modules into lib.rs**

At the top of `crates/dsp/src/lib.rs`, after the existing `use` lines, add:
```rust
mod buffers;
mod spectrum;
mod acf;
mod beat;
```

- [ ] **Step 3: Add `push_history` helper to lib.rs**

Add this free function at the bottom of `crates/dsp/src/lib.rs` (above `#[cfg(test)]`):
```rust
/// Shift a history buffer left by one slot (oldest at index 0 falls off)
/// and write `value` at the end. No-op for empty buffers.
fn push_history(buf: &mut [f32], value: f32) {
    if buf.is_empty() {
        return;
    }
    buf.copy_within(1.., 0);
    let last = buf.len() - 1;
    buf[last] = value;
}
```

- [ ] **Step 4: Verify it builds**

Run: `cargo build -p dsp`
Expected: builds with no errors. (Warnings about unused `buffers`/`spectrum`/`acf`/`beat` modules are fine for now.)

- [ ] **Step 5: Verify tests still pass**

Run: `cargo test -p dsp`
Expected: all existing tests pass (~12 tests).

- [ ] **Step 6: Commit**

```bash
git add crates/dsp/src/buffers.rs crates/dsp/src/spectrum.rs crates/dsp/src/acf.rs crates/dsp/src/beat.rs crates/dsp/src/lib.rs
git commit -m "refactor(dsp): module skeleton + push_history helper"
```

---

## Task 2: Move pure free functions to `acf.rs` and `beat.rs`

**Files:**
- Modify: `crates/dsp/src/acf.rs`
- Modify: `crates/dsp/src/beat.rs`
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Move `compute_gen_acf`, `compute_harmonic_enhanced`, `autocorrelate`, `bin_for_hz` to `acf.rs`**

Replace contents of `crates/dsp/src/acf.rs` with:
```rust
//! Generalized autocorrelation (Percival & Tzanetakis) + harmonic
//! enhancement + time-domain autocorrelate.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};
use std::sync::Arc;

/// Generalized autocorrelation per Percival & Tzanetakis 2014 §II-B.2:
/// zero-pad input to 2N, forward FFT, magnitude compression `|X|^0.5`,
/// inverse FFT, take the first N/2 lags, normalize so output[0] == 1.0.
/// Allocation-free: caller passes scratch buffers (`time_buf` length 2N,
/// `freq_buf` length N+1) and pre-built `realfft` planners.
///
/// `c = 0.5` (the paper's empirically best choice) gives narrower ACF peaks
/// than `c = 2.0` (regular ACF) — this is what makes downstream peak picking
/// and pulse-train scoring more discriminative.
pub fn compute_gen_acf(
    input: &[f32],
    output: &mut [f32],
    fft_forward: &Arc<dyn RealToComplex<f32>>,
    fft_inverse: &Arc<dyn ComplexToReal<f32>>,
    time_buf: &mut [f32],
    freq_buf: &mut [Complex<f32>],
) {
    let n = input.len();
    debug_assert_eq!(time_buf.len(), 2 * n);
    debug_assert_eq!(freq_buf.len(), n + 1);

    time_buf[..n].copy_from_slice(input);
    time_buf[n..].fill(0.0);

    let _ = fft_forward.process(time_buf, freq_buf);

    for x in freq_buf.iter_mut() {
        let compressed = (x.re * x.re + x.im * x.im).powf(0.25);
        *x = Complex::new(compressed, 0.0);
    }

    let _ = fft_inverse.process(freq_buf, time_buf);

    let zero = time_buf[0].max(1e-12);
    for i in 0..output.len() {
        output[i] = time_buf[i] / zero;
    }
}

/// Per Percival & Tzanetakis 2014 §II-B.3: boost peaks corresponding to
/// integer multiples of the underlying tempo by adding time-stretched
/// versions of the ACF.
pub fn compute_harmonic_enhanced(acf: &[f32], enhanced: &mut [f32]) {
    const HARMONIC_MULTIPLES: [usize; 2] = [2, 4];
    let n = acf.len();
    for tau in 0..n {
        let mut sum = acf[tau];
        for &mult in &HARMONIC_MULTIPLES {
            let idx = tau * mult;
            if idx < n {
                sum += acf[idx];
            }
        }
        enhanced[tau] = sum;
    }
}

/// Direct time-domain autocorrelation, normalized so output[0] == 1.0
/// for any nonzero input. For all-zero input the output is filled with
/// zeros. Caller chooses how many lags to compute via `output.len()`.
pub fn autocorrelate(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    for k in 0..output.len() {
        let mut sum = 0.0f32;
        if k < n {
            for i in 0..(n - k) {
                sum += input[i] * input[i + k];
            }
        }
        output[k] = sum;
    }
    let zero = output[0];
    if zero > 0.0 {
        for v in output.iter_mut() {
            *v /= zero;
        }
    } else {
        output.fill(0.0);
    }
}

/// Snap a frequency in Hz to the nearest one-sided real-FFT bin index,
/// clamped to [1, N/2 - 1] (DC and Nyquist excluded by design).
pub fn bin_for_hz(hz: f32, sample_rate: f32, n: usize) -> usize {
    let bin = (hz * n as f32 / sample_rate).round() as usize;
    bin.clamp(1, n / 2 - 1)
}
```

- [ ] **Step 2: Move `score_phase_for_tau` to `beat.rs`**

Replace contents of `crates/dsp/src/beat.rs` with:
```rust
//! Candidate picking → phase scoring → TEA → beat outputs.

/// Number of pulses per train in `score_phase_for_tau`.
pub const PULSE_N: usize = 4;

/// Score one tempo lag `tau` against the OSS by sweeping integer phases
/// `phi ∈ [0, ceil(tau))`. Returns `(best_phi, best_corr, sum_corr,
/// sum_corr_sq, n_phases)`. Pulse-train is the paper's combined
/// `Φ₁ (w=1.0) + Φ₂ (w=0.5) + Φ₁.₅ (w=0.5)` with N=4 pulses each.
pub fn score_phase_for_tau(onset: &[f32], tau: f32) -> (usize, f32, f32, f32, usize) {
    let n = onset.len();
    if n == 0 || tau < 1.0 {
        return (0, 0.0, 0.0, 0.0, 0);
    }
    let last = (n - 1) as i32;
    let phi_max = (tau.ceil() as usize).max(1);

    let mut best_phi = 0usize;
    let mut best_corr = -1.0f32;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;

    for phi in 0..phi_max {
        let mut corr = 0.0f32;
        for k in 0..PULSE_N {
            let off = (k as f32 * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                corr += onset[pos as usize];
            }
        }
        for k in 0..PULSE_N {
            let off = (k as f32 * 2.0 * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                corr += 0.5 * onset[pos as usize];
            }
        }
        for k in 0..PULSE_N {
            let off = ((k as f32 + 0.5) * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                corr += 0.5 * onset[pos as usize];
            }
        }

        sum += corr;
        sum_sq += corr * corr;
        if corr > best_corr {
            best_corr = corr;
            best_phi = phi;
        }
    }

    (best_phi, best_corr.max(0.0), sum, sum_sq, phi_max)
}
```

- [ ] **Step 3: Update `lib.rs` to call these from the new modules**

In `crates/dsp/src/lib.rs`:

a) Delete the bodies of `fn compute_gen_acf`, `fn compute_harmonic_enhanced`, `fn autocorrelate`, `fn bin_for_hz`, `fn score_phase_for_tau` — they are gone, only references remain.

b) Delete the standalone `const HARMONIC_MULTIPLES` and `const PULSE_N` constants in `lib.rs` (they live in their respective modules now).

c) Update existing call sites in `lib.rs::Dsp::process` and helper methods:
- `compute_gen_acf(...)` → `crate::acf::compute_gen_acf(...)`
- `compute_harmonic_enhanced(...)` → `crate::acf::compute_harmonic_enhanced(...)`
- `autocorrelate(...)` → `crate::acf::autocorrelate(...)`
- `bin_for_hz(...)` → `crate::acf::bin_for_hz(...)`
- `score_phase_for_tau(...)` → `crate::beat::score_phase_for_tau(...)`

d) Update test references in `lib.rs::tests::autocorrelate_helper_correctness` to call `crate::acf::autocorrelate(...)`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p dsp`
Expected: all ~12 tests pass.

- [ ] **Step 5: Run JS tests + build wasm**

Run: `npm run wasm && npm test`
Expected: wasm rebuilds successfully, all JS tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/dsp/src/acf.rs crates/dsp/src/beat.rs crates/dsp/src/lib.rs
git commit -m "refactor(dsp): move pure free functions into acf.rs/beat.rs"
```

---

## Task 3: Introduce `Buffers` struct, migrate output fields

**Files:**
- Modify: `crates/dsp/src/buffers.rs`
- Modify: `crates/dsp/src/lib.rs`

This task moves all 15 output `Vec<f32>` fields from `Dsp` into a `Buffers` struct using camelCase names, and rewires `Dsp::process` and the typed getters to read/write through `self.buffers.*`. The wasm-bindgen surface stays unchanged — getters still return `Vec<f32>` clones from the same data.

- [ ] **Step 1: Define `Buffers` in `buffers.rs`**

Replace contents of `crates/dsp/src/buffers.rs` with:
```rust
//! Output-buffer registry. Single source of truth for the JS-visible buffer
//! key vocabulary.
//!
//! Field names use camelCase to match the JS-side keys exactly — the same
//! string is used as Rust struct field, registry-lookup match arm, worklet
//! message field, and FeatureStore key. `#[allow(non_snake_case)]` accepts
//! these names.

const MAX_PEAKS: usize = 10;
const BEAT_GRID_LEN: usize = 3;
const BEAT_PULSES_LEN: usize = 4;
const BEAT_STATE_LEN: usize = 4;

#[allow(non_snake_case)]
pub struct Buffers {
    pub waveform: Vec<f32>,
    pub spectrum: Vec<f32>,
    pub bufferAcf: Vec<f32>,
    pub rms: Vec<f32>,
    pub rmsLow: Vec<f32>,
    pub rmsMid: Vec<f32>,
    pub rmsHigh: Vec<f32>,
    pub onset: Vec<f32>,
    pub onsetAcf: Vec<f32>,
    pub onsetAcfEnhanced: Vec<f32>,
    pub tea: Vec<f32>,
    pub candidates: Vec<f32>,
    pub beatGrid: Vec<f32>,
    pub beatPulses: Vec<f32>,
    pub beatState: Vec<f32>,
}

impl Buffers {
    pub fn new(window_size: usize, rms_history_len: usize) -> Self {
        let onset_acf_len = rms_history_len / 2;
        Self {
            waveform: vec![0.0; window_size],
            spectrum: vec![0.0; window_size / 2],
            bufferAcf: vec![0.0; window_size / 2],
            rms: vec![0.0; rms_history_len],
            rmsLow: vec![0.0; rms_history_len],
            rmsMid: vec![0.0; rms_history_len],
            rmsHigh: vec![0.0; rms_history_len],
            onset: vec![0.0; rms_history_len],
            onsetAcf: vec![0.0; onset_acf_len],
            onsetAcfEnhanced: vec![0.0; onset_acf_len],
            tea: vec![0.0; onset_acf_len],
            candidates: vec![f32::NAN; 3 * MAX_PEAKS],
            beatGrid: vec![f32::NAN; BEAT_GRID_LEN],
            beatPulses: vec![f32::NAN; BEAT_PULSES_LEN],
            beatState: vec![f32::NAN; BEAT_STATE_LEN],
        }
    }

    /// Look up a buffer's current contents by string key. Returns `None`
    /// for unknown names. The 15 keys here ARE the JS contract — keep this
    /// match in sync with `descriptors()` and the `Buffers` field list.
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        match name {
            "waveform" => Some(&self.waveform),
            "spectrum" => Some(&self.spectrum),
            "bufferAcf" => Some(&self.bufferAcf),
            "rms" => Some(&self.rms),
            "rmsLow" => Some(&self.rmsLow),
            "rmsMid" => Some(&self.rmsMid),
            "rmsHigh" => Some(&self.rmsHigh),
            "onset" => Some(&self.onset),
            "onsetAcf" => Some(&self.onsetAcf),
            "onsetAcfEnhanced" => Some(&self.onsetAcfEnhanced),
            "tea" => Some(&self.tea),
            "candidates" => Some(&self.candidates),
            "beatGrid" => Some(&self.beatGrid),
            "beatPulses" => Some(&self.beatPulses),
            "beatState" => Some(&self.beatState),
            _ => None,
        }
    }

    /// `[(name, length), ...]` for `Dsp::buffer_descriptors`. Order is stable
    /// (compile-time literal); the worklet caches it once per `configured`.
    pub fn descriptors(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("waveform", self.waveform.len()),
            ("spectrum", self.spectrum.len()),
            ("bufferAcf", self.bufferAcf.len()),
            ("rms", self.rms.len()),
            ("rmsLow", self.rmsLow.len()),
            ("rmsMid", self.rmsMid.len()),
            ("rmsHigh", self.rmsHigh.len()),
            ("onset", self.onset.len()),
            ("onsetAcf", self.onsetAcf.len()),
            ("onsetAcfEnhanced", self.onsetAcfEnhanced.len()),
            ("tea", self.tea.len()),
            ("candidates", self.candidates.len()),
            ("beatGrid", self.beatGrid.len()),
            ("beatPulses", self.beatPulses.len()),
            ("beatState", self.beatState.len()),
        ]
    }
}
```

- [ ] **Step 2: Replace 15 output fields on `Dsp` with `buffers: Buffers`**

In `crates/dsp/src/lib.rs`:

a) At the top, after `mod` declarations, add:
```rust
use crate::buffers::Buffers;
```

b) In the `Dsp` struct definition, **delete** the following 15 fields:
- `waveform`, `spectrum`, `buffer_acf`, `rms_history`, `low_rms_history`, `mid_rms_history`, `high_rms_history`, `onset_history`, `onset_acf`, `onset_acf_enhanced`, `tea`, `candidates`, `beat_grid`, `beat_pulses`, `beat_state`

c) Add a single new field in their place:
```rust
buffers: Buffers,
```

d) In `Dsp::new`, replace the per-field initializations (all 15 `vec![0.0; ...]` / `vec![f32::NAN; ...]` lines for the deleted fields) with:
```rust
buffers: Buffers::new(window_size, rms_history_len),
```

The other fields on `Dsp` (FFT planners, hann, scratch, scalars) keep their existing initialization.

- [ ] **Step 3: Update `Dsp::process` to use `self.buffers.*`**

In `Dsp::process`, replace every reference to the deleted flat fields with the new `Buffers` field name:

| Old | New |
|---|---|
| `self.waveform` | `self.buffers.waveform` |
| `self.spectrum` | `self.buffers.spectrum` |
| `self.buffer_acf` | `self.buffers.bufferAcf` |
| `self.rms_history` | `self.buffers.rms` |
| `self.low_rms_history` | `self.buffers.rmsLow` |
| `self.mid_rms_history` | `self.buffers.rmsMid` |
| `self.high_rms_history` | `self.buffers.rmsHigh` |
| `self.onset_history` | `self.buffers.onset` |
| `self.onset_acf` | `self.buffers.onsetAcf` |
| `self.onset_acf_enhanced` | `self.buffers.onsetAcfEnhanced` |
| `self.tea` | `self.buffers.tea` |
| `self.candidates` | `self.buffers.candidates` |
| `self.beat_grid` | `self.buffers.beatGrid` |
| `self.beat_pulses` | `self.buffers.beatPulses` |
| `self.beat_state` | `self.buffers.beatState` |

Apply the same renames inside `pick_candidates`, `score_candidates`, `update_tea`, `write_beat_outputs`, `update_beat_pulses_v2`.

Replace the inline rms-history shifting with `push_history`:
```rust
push_history(&mut self.buffers.rms, rms);
```
(Same idiom for the band RMS shifts at the end of the FFT loop, and for the onset_history shift after the spectral-flux block.)

- [ ] **Step 4: Update typed getters to wrap `self.buffers.*`**

The 15 typed getter methods in the `#[wasm_bindgen]` impl block stay. Each one's body becomes a one-liner:
```rust
pub fn waveform(&self) -> Vec<f32> { self.buffers.waveform.clone() }
pub fn spectrum(&self) -> Vec<f32> { self.buffers.spectrum.clone() }
pub fn buffer_acf(&self) -> Vec<f32> { self.buffers.bufferAcf.clone() }
pub fn rms_history(&self) -> Vec<f32> { self.buffers.rms.clone() }
pub fn onset_history(&self) -> Vec<f32> { self.buffers.onset.clone() }
pub fn onset_acf(&self) -> Vec<f32> { self.buffers.onsetAcf.clone() }
pub fn onset_acf_enhanced(&self) -> Vec<f32> { self.buffers.onsetAcfEnhanced.clone() }
pub fn candidates(&self) -> Vec<f32> { self.buffers.candidates.clone() }
pub fn tea(&self) -> Vec<f32> { self.buffers.tea.clone() }
pub fn low_rms_history(&self) -> Vec<f32> { self.buffers.rmsLow.clone() }
pub fn mid_rms_history(&self) -> Vec<f32> { self.buffers.rmsMid.clone() }
pub fn high_rms_history(&self) -> Vec<f32> { self.buffers.rmsHigh.clone() }
pub fn beat_grid(&self) -> Vec<f32> { self.buffers.beatGrid.clone() }
pub fn beat_pulses(&self) -> Vec<f32> { self.buffers.beatPulses.clone() }
pub fn beat_state(&self) -> Vec<f32> { self.buffers.beatState.clone() }
```

- [ ] **Step 5: Update `#[cfg(test)] impl Dsp` poke methods**

In `crates/dsp/src/lib.rs`, the `#[cfg(test)] impl Dsp` block has methods that touch `self.onset_acf_enhanced` and `self.onset_history` directly. Update those:
- `self.onset_acf_enhanced` → `self.buffers.onsetAcfEnhanced` (every reference inside `onset_acf_enhanced_len`, `test_set_onset_acf_enhanced`, `test_run_pick_and_score`)
- `self.onset_history` → `self.buffers.onset` (every reference inside `onset_history_len`, `test_set_onset_history`, `test_run_pick_and_score`)
- `self.tea` → `self.buffers.tea` (inside `tea_len`, `test_set_tea`)
- `self.tau_smoothed` stays as-is (it's a scalar, not a buffer)

- [ ] **Step 6: Run tests**

Run: `cargo test -p dsp`
Expected: all ~12 tests pass.

- [ ] **Step 7: Verify wasm + JS tests**

Run: `npm run wasm && npm test`
Expected: wasm rebuilds, JS tests pass. (No JS code changed — we're verifying we didn't break the `Vec<f32>`-returning getters.)

- [ ] **Step 8: Commit**

```bash
git add crates/dsp/src/buffers.rs crates/dsp/src/lib.rs
git commit -m "refactor(dsp): introduce Buffers struct, migrate output fields"
```

---

## Task 4: Extract `SpectrumState`

**Files:**
- Modify: `crates/dsp/src/spectrum.rs`
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Define `SpectrumState` in `spectrum.rs`**

Replace contents of `crates/dsp/src/spectrum.rs` with:
```rust
//! FFT + windowing + spectrum smoothing + per-band RMS + spectral flux.

use crate::acf::bin_for_hz;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Spectrum smoothing time constant in seconds (default).
const SMOOTHING_TAU_SECS_DEFAULT: f32 = 0.0956;
const LOW_BAND_HZ_MAX: f32 = 150.0;
const MID_BAND_HZ_MAX: f32 = 1500.0;

pub struct SpectrumState {
    fft: Arc<dyn RealToComplex<f32>>,
    fft_buffer: Vec<f32>,
    freq_buffer: Vec<Complex<f32>>,
    hann: Vec<f32>,
    /// 2/sum(hann). FFT bin magnitude → amplitude-equivalent units.
    mag_scale: f32,
    /// Previous frame's per-bin |X|, scaled by `mag_scale`. Used for spectral flux.
    prev_mag: Vec<f32>,
    /// EMA coefficient: `1 - exp(-dt / tau)`. Recomputed by `set_smoothing_tau`.
    smoothing_alpha: f32,
    low_band_bin_end: usize,
    mid_band_bin_end: usize,
    /// Parseval scale: 2 / (N · Σ hann²). Maps Σ|X[k]|² over a band → band RMS².
    parseval_band_scale: f32,
}

impl SpectrumState {
    pub fn new(window_size: usize, sample_rate: f32, dt: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let freq_buffer = fft.make_output_vec();
        let spectrum_len = freq_buffer.len() - 1;
        let hann: Vec<f32> = (0..window_size)
            .map(|i| {
                0.5 - 0.5
                    * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos()
            })
            .collect();
        let mag_scale = 2.0 / hann.iter().sum::<f32>();
        let smoothing_alpha = 1.0 - (-dt / SMOOTHING_TAU_SECS_DEFAULT).exp();
        let low_band_bin_end = bin_for_hz(LOW_BAND_HZ_MAX, sample_rate, window_size);
        let mid_band_bin_end = bin_for_hz(MID_BAND_HZ_MAX, sample_rate, window_size);
        let hann_energy: f32 = hann.iter().map(|h| h * h).sum();
        let parseval_band_scale = 2.0 / (window_size as f32 * hann_energy);
        Self {
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            hann,
            mag_scale,
            prev_mag: vec![0.0; spectrum_len],
            smoothing_alpha,
            low_band_bin_end,
            mid_band_bin_end,
            parseval_band_scale,
        }
    }

    pub fn set_smoothing_tau(&mut self, tau_secs: f32, dt: f32) {
        let tau = tau_secs.clamp(0.001, 10.0);
        self.smoothing_alpha = 1.0 - (-dt / tau).exp();
    }

    /// Run one FFT hop. Writes the smoothed normalized [0,1] `spectrum`.
    /// Returns `(low_rms, mid_rms, high_rms, flux)` — three Parseval-correct
    /// band-RMS scalars (caller pushes into history buffers via push_history)
    /// and the spectral-flux onset value (Σ max(0, |X[k]| - prev_mag[k])).
    pub fn process(
        &mut self,
        input: &[f32],
        spectrum: &mut [f32],
        db_floor: f32,
    ) -> (f32, f32, f32, f32) {
        let window_size = self.fft_buffer.len();
        let n = input.len().min(window_size);

        for i in 0..n {
            self.fft_buffer[i] = input[i] * self.hann[i];
        }
        for i in n..window_size {
            self.fft_buffer[i] = 0.0;
        }

        let _ = self.fft.process(&mut self.fft_buffer, &mut self.freq_buffer);

        // Spectral flux + spectrum smoothing in one pass over bins 1..=N/2.
        let mut flux = 0.0f32;
        for (out_i, bin) in self.freq_buffer[1..=spectrum.len()].iter().enumerate() {
            let mag = (bin.re * bin.re + bin.im * bin.im).sqrt() * self.mag_scale;
            flux += (mag - self.prev_mag[out_i]).max(0.0);
            self.prev_mag[out_i] = mag;

            let db = if mag > 0.0 {
                20.0 * mag.log10()
            } else {
                db_floor
            };
            let clipped = db.clamp(db_floor, 0.0);
            let normalized = (clipped - db_floor) / (-db_floor);
            spectrum[out_i] =
                self.smoothing_alpha * normalized + (1.0 - self.smoothing_alpha) * spectrum[out_i];
        }

        // Per-band RMS via Parseval-correct FFT-bin energy summation.
        let nyquist_bin = self.freq_buffer.len() - 1;
        let mut low_e = 0.0f32;
        for k in 1..=self.low_band_bin_end {
            let c = self.freq_buffer[k];
            low_e += c.re * c.re + c.im * c.im;
        }
        let mut mid_e = 0.0f32;
        for k in (self.low_band_bin_end + 1)..=self.mid_band_bin_end {
            let c = self.freq_buffer[k];
            mid_e += c.re * c.re + c.im * c.im;
        }
        let mut high_e = 0.0f32;
        for k in (self.mid_band_bin_end + 1)..nyquist_bin {
            let c = self.freq_buffer[k];
            high_e += c.re * c.re + c.im * c.im;
        }
        let low_rms = (low_e * self.parseval_band_scale).sqrt();
        let mid_rms = (mid_e * self.parseval_band_scale).sqrt();
        let high_rms = (high_e * self.parseval_band_scale).sqrt();

        (low_rms, mid_rms, high_rms, flux)
    }
}
```

- [ ] **Step 2: Update `Dsp` to hold `SpectrumState`**

In `crates/dsp/src/lib.rs`:

a) Add to imports:
```rust
use crate::spectrum::SpectrumState;
```

b) In the `Dsp` struct, **delete** these fields:
- `fft`, `fft_buffer`, `freq_buffer`, `hann`, `mag_scale`, `prev_mag`, `smoothing_alpha`, `low_band_bin_end`, `mid_band_bin_end`, `parseval_band_scale`

Add in their place:
```rust
spectrum: SpectrumState,
```

c) In `Dsp::new`:
- Delete the local-variable computations of `hann`, `mag_scale`, `low_band_bin_end`, `mid_band_bin_end`, `hann_energy`, `parseval_band_scale`, `smoothing_alpha`, and the `fft`/`freq_buffer` planner calls.
- The `dt` computation stays: `let dt = hop_size as f32 / sample_rate;`
- Replace the deleted field initializations with:
  ```rust
  spectrum: SpectrumState::new(window_size, sample_rate, dt),
  ```

d) Rename the `set_smoothing_tau` method to forward:
```rust
pub fn set_smoothing_tau(&mut self, tau_secs: f32) {
    self.spectrum.set_smoothing_tau(tau_secs, self.dt);
}
```

- [ ] **Step 3: Replace the inline FFT/spectrum/bands block in `Dsp::process` with `self.spectrum.process(...)`**

Inside `Dsp::process`, **delete** the entire inline block that does:
- the `for i in 0..n { self.fft_buffer[i] = input[i] * self.hann[i]; }` loop
- the FFT call
- the spectral-flux + spectrum smoothing loop
- the band-RMS Parseval block
- the per-band history shift loop

Replace it with:
```rust
let (low_rms, mid_rms, high_rms, flux) = self.spectrum.process(
    input,
    &mut self.buffers.spectrum,
    self.db_floor,
);
push_history(&mut self.buffers.rmsLow, low_rms);
push_history(&mut self.buffers.rmsMid, mid_rms);
push_history(&mut self.buffers.rmsHigh, high_rms);
push_history(&mut self.buffers.onset, flux);
```

(The pre-FFT `waveform[..n].copy_from_slice(...)` and full-band RMS computation + `push_history(&mut self.buffers.rms, rms)` stay where they are — they don't depend on the FFT.)

- [ ] **Step 4: Run tests**

Run: `cargo test -p dsp`
Expected: all tests pass. Specifically: `loud_sine_produces_a_peak`, `silent_input_yields_low_spectrum`, `spectrum_has_window_size_div_2_bins`, `smoothing_alpha_matches_time_constant_formula` — the spectrum behavior is unchanged.

- [ ] **Step 5: Verify wasm + JS tests + visual smoke**

Run: `npm run wasm && npm test`
Expected: wasm rebuilds, JS tests pass.

(Optional but encouraged: `npm run dev` and confirm spectrum + RMS bands render normally before committing.)

- [ ] **Step 6: Commit**

```bash
git add crates/dsp/src/spectrum.rs crates/dsp/src/lib.rs
git commit -m "refactor(dsp): extract SpectrumState"
```

---

## Task 5: Extract `AcfState`

**Files:**
- Modify: `crates/dsp/src/acf.rs`
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Add `AcfState` to `acf.rs`**

Append to `crates/dsp/src/acf.rs`:
```rust
use realfft::ComplexToReal;

pub struct AcfState {
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    time_buf: Vec<f32>,
    freq_buf: Vec<Complex<f32>>,
}

impl AcfState {
    pub fn new(rms_history_len: usize) -> Self {
        let n = rms_history_len;
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(2 * n);
        let fft_inverse = planner.plan_fft_inverse(2 * n);
        Self {
            fft_forward,
            fft_inverse,
            time_buf: vec![0.0; 2 * n],
            freq_buf: vec![Complex::new(0.0, 0.0); n + 1],
        }
    }

    /// Run gen-ACF on `onset` → write `onset_acf`, then harmonic-enhance →
    /// write `onset_acf_enhanced`.
    pub fn process(
        &mut self,
        onset: &[f32],
        onset_acf: &mut [f32],
        onset_acf_enhanced: &mut [f32],
    ) {
        compute_gen_acf(
            onset,
            onset_acf,
            &self.fft_forward,
            &self.fft_inverse,
            &mut self.time_buf,
            &mut self.freq_buf,
        );
        compute_harmonic_enhanced(onset_acf, onset_acf_enhanced);
    }
}
```

- [ ] **Step 2: Update `Dsp` to hold `AcfState`**

In `crates/dsp/src/lib.rs`:

a) Add to imports:
```rust
use crate::acf::AcfState;
```

b) Delete these fields from `Dsp`:
- `gen_acf_fft_forward`, `gen_acf_fft_inverse`, `gen_acf_time_buf`, `gen_acf_freq_buf`

Add in their place:
```rust
acf: AcfState,
```

c) In `Dsp::new`:
- Delete the local computations of `gen_acf_n`, `gen_acf_fft_forward`, `gen_acf_fft_inverse`, `gen_acf_time_buf`, `gen_acf_freq_buf`.
- Replace the deleted field initializations with:
  ```rust
  acf: AcfState::new(rms_history_len),
  ```

- [ ] **Step 3: Replace the gen-ACF + harmonic-enhanced calls in `Dsp::process`**

In `Dsp::process`, find these two consecutive calls:
```rust
crate::acf::compute_gen_acf(
    &self.buffers.onset,
    &mut self.buffers.onsetAcf,
    &self.gen_acf_fft_forward,
    &self.gen_acf_fft_inverse,
    &mut self.gen_acf_time_buf,
    &mut self.gen_acf_freq_buf,
);
crate::acf::compute_harmonic_enhanced(&self.buffers.onsetAcf, &mut self.buffers.onsetAcfEnhanced);
```

Replace with:
```rust
self.acf.process(
    &self.buffers.onset,
    &mut self.buffers.onsetAcf,
    &mut self.buffers.onsetAcfEnhanced,
);
```

- [ ] **Step 4: Update `#[cfg(test)] impl Dsp::test_run_pick_and_score`**

The test helper currently calls `compute_gen_acf` directly with the old field names. Update it:
```rust
pub fn test_run_pick_and_score(&mut self) {
    self.acf.process(
        &self.buffers.onset,
        &mut self.buffers.onsetAcf,
        &mut self.buffers.onsetAcfEnhanced,
    );
    self.pick_candidates();
    self.score_candidates();
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p dsp`
Expected: all tests pass.

- [ ] **Step 6: Verify wasm + JS tests**

Run: `npm run wasm && npm test`
Expected: wasm rebuilds, JS tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/dsp/src/acf.rs crates/dsp/src/lib.rs
git commit -m "refactor(dsp): extract AcfState"
```

---

## Task 6: Extract `BeatState`

**Files:**
- Modify: `crates/dsp/src/beat.rs`
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Add `BeatState` to `beat.rs`**

Append to `crates/dsp/src/beat.rs`:
```rust
const MAX_PEAKS: usize = 10;
const MIN_PEAK_SPACING: usize = 3;
const BEAT_TRACKER_MIN_BPM: f32 = 40.0;
const BEAT_TRACKER_MAX_BPM: f32 = 220.0;
const BEAT_PULSE_CYCLES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const TEA_GAUSSIAN_SIGMA: f32 = 5.0;
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;

pub struct BeatState {
    tau_min: usize,
    tau_max: usize,
    cand_scratch: Vec<(usize, f32)>,
    pulse_x: [f32; MAX_PEAKS],
    pulse_v: [f32; MAX_PEAKS],
    pulse_phi: [f32; MAX_PEAKS],
    pulse_score: [f32; MAX_PEAKS],
    period_inst: f32,
    phase_inst: f32,
    score_inst: f32,
    tea_alpha: f32,
    tau_smoothed: f32,
    phase_smoothed: f32,
    beat_position: f32,
}

impl BeatState {
    pub fn new(rms_history_len: usize, dt: f32) -> Self {
        let onset_acf_len = rms_history_len / 2;
        let tau_min = ((60.0 / BEAT_TRACKER_MAX_BPM) / dt).floor().max(1.0) as usize;
        let tau_max_unbounded = ((60.0 / BEAT_TRACKER_MIN_BPM) / dt).ceil() as usize;
        let tau_max = tau_max_unbounded.min(onset_acf_len.saturating_sub(2));
        let tea_alpha = 1.0 - (-dt / TEA_TAU_DEFAULT_SECS).exp();
        Self {
            tau_min,
            tau_max,
            cand_scratch: Vec::with_capacity(onset_acf_len / 2 + 1),
            pulse_x: [0.0; MAX_PEAKS],
            pulse_v: [0.0; MAX_PEAKS],
            pulse_phi: [0.0; MAX_PEAKS],
            pulse_score: [0.0; MAX_PEAKS],
            period_inst: f32::NAN,
            phase_inst: f32::NAN,
            score_inst: 0.0,
            tea_alpha,
            tau_smoothed: f32::NAN,
            phase_smoothed: f32::NAN,
            beat_position: 0.0,
        }
    }

    pub fn set_tea_tau(&mut self, tau_secs: f32, dt: f32) {
        let tau = tau_secs.clamp(0.05, 60.0);
        self.tea_alpha = 1.0 - (-dt / tau).exp();
    }

    /// Run one beat-tracker frame: pick candidates, score phases, update TEA,
    /// write public outputs (`candidates`, `tea`, `beatGrid`, `beatState`,
    /// `beatPulses`).
    pub fn process(
        &mut self,
        onset: &[f32],
        onset_acf_enhanced: &[f32],
        candidates: &mut [f32],
        tea: &mut [f32],
        beat_grid: &mut [f32],
        beat_state: &mut [f32],
        beat_pulses: &mut [f32],
        dt: f32,
    ) {
        self.pick_candidates(onset_acf_enhanced, candidates);
        self.score_candidates(onset, candidates);
        self.update_tea(onset, tea);
        self.write_beat_outputs(beat_grid, beat_state, dt);
        self.update_beat_pulses(beat_pulses);
    }

    fn pick_candidates(&mut self, onset_acf_enhanced: &[f32], candidates: &mut [f32]) {
        for slot in candidates.iter_mut() {
            *slot = f32::NAN;
        }
        if self.tau_max < self.tau_min + 1 {
            return;
        }

        self.cand_scratch.clear();
        let upper = self.tau_max.min(onset_acf_enhanced.len() - 1);
        for k in self.tau_min..upper {
            let y = onset_acf_enhanced[k];
            if y > 0.0 && y > onset_acf_enhanced[k - 1] && y > onset_acf_enhanced[k + 1] {
                self.cand_scratch.push((k, y));
            }
        }

        self.cand_scratch
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut accepted: [u32; MAX_PEAKS] = [0; MAX_PEAKS];
        let mut count: usize = 0;
        for &(k, _) in &self.cand_scratch {
            if count == MAX_PEAKS {
                break;
            }
            let too_close = accepted[..count]
                .iter()
                .any(|&j| ((k as i32 - j as i32).unsigned_abs() as usize) < MIN_PEAK_SPACING);
            if !too_close {
                accepted[count] = k as u32;
                count += 1;
            }
        }

        for i in 0..count {
            let k = accepted[i] as usize;
            let y0 = onset_acf_enhanced[k - 1];
            let y1 = onset_acf_enhanced[k];
            let y2 = onset_acf_enhanced[k + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let (lag_frac, mag) = if denom.abs() < 1e-12 {
                (k as f32, y1)
            } else {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                (k as f32 + delta, y1 - 0.25 * (y0 - y2) * delta)
            };
            candidates[3 * i] = lag_frac;
            candidates[3 * i + 1] = mag;
            candidates[3 * i + 2] = -denom;
        }
    }

    fn score_candidates(&mut self, onset: &[f32], candidates: &[f32]) {
        let mut count = 0usize;
        for i in 0..MAX_PEAKS {
            let lag = candidates[3 * i];
            if lag.is_nan() {
                break;
            }
            let (phi, x, sum, sum_sq, n_phi) = score_phase_for_tau(onset, lag);
            let n = n_phi as f32;
            let mean = if n > 0.0 { sum / n } else { 0.0 };
            let var = if n > 0.0 {
                (sum_sq / n - mean * mean).max(0.0)
            } else {
                0.0
            };
            self.pulse_x[i] = x;
            self.pulse_v[i] = var;
            self.pulse_phi[i] = phi as f32;
            count += 1;
        }
        if count == 0 {
            self.period_inst = f32::NAN;
            self.phase_inst = f32::NAN;
            self.score_inst = 0.0;
            return;
        }
        let sum_x: f32 = self.pulse_x[..count].iter().sum();
        let sum_v: f32 = self.pulse_v[..count].iter().sum();
        let mut best_i = 0usize;
        let mut best_score = -1.0f32;
        for i in 0..count {
            let xn = if sum_x > 0.0 { self.pulse_x[i] / sum_x } else { 0.0 };
            let vn = if sum_v > 0.0 { self.pulse_v[i] / sum_v } else { 0.0 };
            let s = xn + vn;
            self.pulse_score[i] = s;
            if s > best_score {
                best_score = s;
                best_i = i;
            }
        }
        if best_score <= 0.0 {
            self.period_inst = f32::NAN;
            self.phase_inst = f32::NAN;
            self.score_inst = 0.0;
        } else {
            self.period_inst = candidates[3 * best_i];
            self.phase_inst = self.pulse_phi[best_i];
            self.score_inst = best_score;
        }
    }

    fn update_tea(&mut self, onset: &[f32], tea: &mut [f32]) {
        let alpha = self.tea_alpha;
        if self.score_inst > 0.0 && self.period_inst.is_finite() {
            let inv_2sig2 = 1.0 / (2.0 * TEA_GAUSSIAN_SIGMA * TEA_GAUSSIAN_SIGMA);
            for tau in 0..tea.len() {
                let delta = tau as f32 - self.period_inst;
                let g = (-delta * delta * inv_2sig2).exp();
                tea[tau] = (1.0 - alpha) * tea[tau] + alpha * g;
            }
        } else {
            for v in tea.iter_mut() {
                *v *= 1.0 - alpha;
            }
        }

        let upper = self.tau_max.min(tea.len() - 1);
        let mut best_i = self.tau_min;
        let mut best_v = -1.0f32;
        for i in self.tau_min..=upper {
            if tea[i] > best_v {
                best_v = tea[i];
                best_i = i;
            }
        }
        if best_v <= 0.0 {
            self.tau_smoothed = f32::NAN;
            self.phase_smoothed = f32::NAN;
            return;
        }
        let mut tau = best_i as f32;
        if best_i > self.tau_min && best_i < upper {
            let y0 = tea[best_i - 1];
            let y1 = tea[best_i];
            let y2 = tea[best_i + 1];
            let denom = y0 - 2.0 * y1 + y2;
            if denom.abs() > 1e-12 {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                tau = best_i as f32 + delta;
            }
        }
        self.tau_smoothed = tau;
        let (phi, _, _, _, _) = score_phase_for_tau(onset, tau);
        self.phase_smoothed = phi as f32;
    }

    fn write_beat_outputs(&self, beat_grid: &mut [f32], beat_state: &mut [f32], dt: f32) {
        let p = self.tau_smoothed;
        let phi = self.phase_smoothed;
        let s = self.score_inst;
        if p.is_nan() || s <= 0.0 {
            beat_grid[0] = f32::NAN;
            beat_grid[1] = f32::NAN;
            beat_grid[2] = 0.0;
            beat_state[0] = f32::NAN;
            beat_state[1] = 0.0;
            beat_state[2] = f32::NAN;
            beat_state[3] = f32::NAN;
        } else {
            beat_grid[0] = p;
            beat_grid[1] = phi;
            beat_grid[2] = s;
            beat_state[0] = if p > 0.0 { 60.0 / (p * dt) } else { f32::NAN };
            beat_state[1] = s;
            beat_state[2] = f32::NAN;
            beat_state[3] = f32::NAN;
        }
    }

    fn update_beat_pulses(&mut self, beat_pulses: &mut [f32]) {
        let period = self.tau_smoothed;
        let phase = self.phase_smoothed;
        let score = self.score_inst;
        if period.is_nan() || period <= 0.0 || score <= 0.0 || phase.is_nan() {
            for slot in beat_pulses.iter_mut() {
                *slot = f32::NAN;
            }
            return;
        }
        let phase_frac = (phase / period).clamp(0.0, 0.999_999);
        let prev = self.beat_position;
        let mut bp = prev.floor() + phase_frac;
        if bp < prev - 0.5 {
            bp += 1.0;
        }
        self.beat_position = bp.rem_euclid(16.0);
        for (i, &m) in BEAT_PULSE_CYCLES.iter().enumerate() {
            let frac = (self.beat_position / m).fract();
            beat_pulses[i] = 1.0 - frac;
        }
    }
}

#[cfg(test)]
impl BeatState {
    pub fn test_per_frame_estimate(&self) -> (f32, f32, f32) {
        (self.period_inst, self.phase_inst, self.score_inst)
    }
    pub fn tau_smoothed(&self) -> f32 {
        self.tau_smoothed
    }
    pub fn tea_alpha(&self) -> f32 {
        self.tea_alpha
    }
}
```

- [ ] **Step 2: Update `Dsp` to hold `BeatState` and remove the old beat fields**

In `crates/dsp/src/lib.rs`:

a) Add to imports:
```rust
use crate::beat::BeatState;
```

b) Delete these fields from `Dsp`:
- `tau_min`, `tau_max`, `cand_scratch`, `pulse_x`, `pulse_v`, `pulse_phi`, `pulse_score`, `period_inst`, `phase_inst`, `score_inst`, `tea_alpha`, `tau_smoothed`, `phase_smoothed`, `beat_position`

Add in their place:
```rust
beat: BeatState,
```

c) Delete these constants from `lib.rs` (they're in `beat.rs` now): `MAX_PEAKS`, `MIN_PEAK_SPACING`, `BEAT_TRACKER_MIN_BPM`, `BEAT_TRACKER_MAX_BPM`, `BEAT_STATE_LEN`, `BEAT_GRID_LEN`, `BEAT_PULSE_CYCLES`, `BEAT_PULSES_LEN`, `TEA_GAUSSIAN_SIGMA`, `TEA_TAU_DEFAULT_SECS`. (These are also defined in `buffers.rs` and `beat.rs` where needed; lib.rs should have none left.)

d) In `Dsp::new`:
- Delete the local computations of `tau_min`, `tau_max_unbounded`, `tau_max`, `tea_alpha`.
- Replace the deleted field initializations with:
  ```rust
  beat: BeatState::new(rms_history_len, dt),
  ```

e) Replace the existing `set_tea_tau_secs` body:
```rust
pub fn set_tea_tau_secs(&mut self, tau_secs: f32) {
    self.beat.set_tea_tau(tau_secs, self.dt);
}
```

- [ ] **Step 3: Delete inline beat methods on `Dsp`, call `BeatState::process` instead**

In `crates/dsp/src/lib.rs`:

a) **Delete** the entire `impl Dsp { fn pick_candidates ... fn score_candidates ... fn update_tea ... fn write_beat_outputs ... fn update_beat_pulses_v2 ... }` non-wasm-bindgen block. These all live on `BeatState` now.

b) In `Dsp::process`, **delete** the calls:
```rust
self.pick_candidates();
self.score_candidates();
self.update_tea();
self.write_beat_outputs();
self.update_beat_pulses_v2();
```

Replace with:
```rust
self.beat.process(
    &self.buffers.onset,
    &self.buffers.onsetAcfEnhanced,
    &mut self.buffers.candidates,
    &mut self.buffers.tea,
    &mut self.buffers.beatGrid,
    &mut self.buffers.beatState,
    &mut self.buffers.beatPulses,
    self.dt,
);
```

- [ ] **Step 4: Update `#[cfg(test)] impl Dsp` poke methods**

The remaining test helpers must route through `self.beat`:

```rust
pub fn test_run_pick_and_score(&mut self) {
    self.acf.process(
        &self.buffers.onset,
        &mut self.buffers.onsetAcf,
        &mut self.buffers.onsetAcfEnhanced,
    );
    self.beat.process(
        &self.buffers.onset,
        &self.buffers.onsetAcfEnhanced,
        &mut self.buffers.candidates,
        &mut self.buffers.tea,
        &mut self.buffers.beatGrid,
        &mut self.buffers.beatState,
        &mut self.buffers.beatPulses,
        self.dt,
    );
}

pub fn test_per_frame_estimate(&self) -> (f32, f32, f32) {
    self.beat.test_per_frame_estimate()
}

pub fn tea_argmax(&self) -> f32 {
    self.beat.tau_smoothed()
}

pub fn tea_alpha(&self) -> f32 {
    self.beat.tea_alpha()
}
```

(Delete `test_run_pick_candidates` if it exists — its only caller would have been a test that called pick without score, and the new model always couples them. If a test currently calls it, update that test to use `test_run_pick_and_score`.)

- [ ] **Step 5: Run tests**

Run: `cargo test -p dsp`
Expected: all tests pass.

- [ ] **Step 6: Verify wasm + JS tests + visual smoke**

Run: `npm run wasm && npm test`
Expected: wasm rebuilds, JS tests pass.

(Strongly recommended: `npm run dev` and confirm beat tracker is working — pulse squares respond to a beat, BPM converges to a sensible value.)

- [ ] **Step 7: Confirm `Dsp` size**

Run: `wc -l crates/dsp/src/lib.rs`
Expected: well under 500 lines (was 1799). The bulk of `lib.rs` is now just the `Dsp` struct, its `new`, its short `process`, the wasm-bindgen typed-getter wrappers, and the test module.

- [ ] **Step 8: Commit**

```bash
git add crates/dsp/src/beat.rs crates/dsp/src/lib.rs
git commit -m "refactor(dsp): extract BeatState"
```

---

## Task 7: Add new wasm-bindgen API (`get_buffer`, `buffer_descriptors`, `set_param`)

**Files:**
- Modify: `crates/dsp/src/lib.rs`

The old typed getters/setters stay during this task; the new API runs alongside them so the JS side can switch over in the next tasks.

- [ ] **Step 1: Add `get_buffer` and `buffer_descriptors`**

In `crates/dsp/src/lib.rs`, inside the `#[wasm_bindgen] impl Dsp { ... }` block, **before** the existing typed getters, add:

```rust
/// String-keyed buffer accessor — the new JS contract. Returns the named
/// output buffer's current contents as a fresh Vec. Unknown names return
/// an empty Vec; callers should rely on `buffer_descriptors` for the
/// authoritative list.
pub fn get_buffer(&self, name: &str) -> Vec<f32> {
    self.buffers.get(name).map(|s| s.to_vec()).unwrap_or_default()
}

/// `[{name, length}, ...]` for every public output buffer. Order matches
/// `Buffers::descriptors`. Called once per `configured` from the worklet.
pub fn buffer_descriptors(&self) -> Vec<JsValue> {
    let descriptors = self.buffers.descriptors();
    let mut out = Vec::with_capacity(descriptors.len());
    for (name, length) in descriptors {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&obj, &"name".into(), &name.into());
        let _ = js_sys::Reflect::set(&obj, &"length".into(), &(length as u32).into());
        out.push(obj.into());
    }
    out
}
```

- [ ] **Step 2: Add `js-sys` dependency**

In `crates/dsp/Cargo.toml`, under `[dependencies]`, add:
```toml
js-sys = "0.3"
```

- [ ] **Step 3: Add `set_param`**

In the same `#[wasm_bindgen] impl Dsp { ... }` block, add:

```rust
/// Set a tunable param. Unknown keys are silently ignored.
/// Recognized keys: "smoothingTauSecs", "teaTauSecs", "dbFloor".
pub fn set_param(&mut self, key: &str, value: f32) {
    match key {
        "smoothingTauSecs" => self.spectrum.set_smoothing_tau(value, self.dt),
        "teaTauSecs" => self.beat.set_tea_tau(value, self.dt),
        "dbFloor" => self.db_floor = value.clamp(-200.0, 0.0),
        _ => {}
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p dsp`
Expected: all tests pass.

Run: `npm run wasm && npm test`
Expected: wasm rebuilds, JS tests pass. (No JS code yet uses the new methods.)

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/Cargo.toml crates/dsp/src/lib.rs
git commit -m "feat(dsp): add string-keyed wasm API (getBuffer/bufferDescriptors/setParam)"
```

---

## Task 8: Switch worklet to new API

**Files:**
- Modify: `src/audio/dsp-worklet.ts`

- [ ] **Step 1: Update message protocol types**

In `src/audio/dsp-worklet.ts`, replace the existing `ConfiguredOutbound` type with:

```ts
type ConfiguredOutbound = {
  type: "configured";
  buffers: { name: string; length: number }[];
};
```

Also loosen the `WorkletInbound` `param` variant so the bridge can forward any param key the wasm side recognizes — the worklet no longer needs to know the recognized set:
```ts
type WorkletInbound =
  | { type: "configure"; windowSize: number; rmsHistoryLen: number }
  | { type: "param"; key: string; value: number }
  | { type: "sync" };
```

(The `*Len` per-field form on `ConfiguredOutbound` is gone.)

- [ ] **Step 2: Use `dsp.bufferDescriptors()` in `applyConfigure`**

Replace the existing `payload: ConfiguredOutbound = { type: "configured", waveformLen: ..., spectrumLen: ..., ... }` literal in `applyConfigure` with:

```ts
const descriptors = this.dsp!.bufferDescriptors() as { name: string; length: number }[];
const payload: ConfiguredOutbound = {
  type: "configured",
  buffers: descriptors,
};
this.lastConfigured = payload;
this.port.postMessage(payload);
```

- [ ] **Step 3: Cache buffer names + post features as a dict**

Add a class field on `DSPProcessor`:
```ts
private bufferNames: string[] = [];
```

In `applyConfigure`, after computing `descriptors`, set:
```ts
this.bufferNames = descriptors.map((d) => d.name);
```

In `process()`, replace the entire block that currently does the 15 `dsp.X()` calls + builds the `{ type: "features", waveform: ..., spectrum: ..., ... }` literal + the 15-entry transfer list with:

```ts
const buffers: { [name: string]: Float32Array } = {};
const transferList: ArrayBuffer[] = [];
for (const name of this.bufferNames) {
  const data = new Float32Array(this.dsp.getBuffer(name));
  buffers[name] = data;
  transferList.push(data.buffer);
}
this.port.postMessage(
  { type: "features", buffers },
  transferList,
);
```

- [ ] **Step 4: Forward `set_param` in the param message handler**

Replace the existing `if (msg.type === "param") { ... }` body in `onMessage` with:

```ts
if (msg.type === "param") {
  if (msg.key === "hopSize") {
    this.hopSize = Math.min(msg.value, this.windowSize);
    return;
  }
  if (this.ready && this.dsp) {
    this.dsp.setParam(msg.key, msg.value);
  }
  // Cache the value so a later `applyConfigure` re-applies it.
  if (msg.key === "smoothingTauSecs") this.smoothingTauSecs = msg.value;
  else if (msg.key === "teaTauSecs") this.teaTauSecs = msg.value;
  else if (msg.key === "dbFloor") this.dbFloor = msg.value;
  return;
}
```

In `applyConfigure`, replace the three lines:
```ts
this.dsp.set_smoothing_tau(this.smoothingTauSecs);
this.dsp.set_db_floor(this.dbFloor);
this.dsp.set_tea_tau_secs(this.teaTauSecs);
```
with:
```ts
this.dsp.setParam("smoothingTauSecs", this.smoothingTauSecs);
this.dsp.setParam("teaTauSecs", this.teaTauSecs);
this.dsp.setParam("dbFloor", this.dbFloor);
```

(The typed setters still exist on the wasm side until Task 11; we're switching the call site early so the worklet uses one path.)

- [ ] **Step 5: Build wasm + verify TypeScript compiles**

Run: `npm run wasm`
Expected: wasm rebuilds.

Run: `npx tsc --noEmit`
Expected: type-checks clean.

- [ ] **Step 6: Commit**

```bash
git add src/audio/dsp-worklet.ts
git commit -m "refactor(worklet): use string-keyed dsp API for buffers and params"
```

---

## Task 9: Switch DebugView + BeatDebugView to loop pattern

**Files:**
- Modify: `src/render/DebugView.ts`
- Modify: `src/render/BeatDebugView.ts`

- [ ] **Step 1: Update message types in DebugView**

In `src/render/DebugView.ts`, find the `DebugFeatures` and `DebugSizes` interfaces (or type imports — depending on where they're declared). Replace with:

```ts
export interface DebugFeatures {
  buffers: { [name: string]: Float32Array };
}

export interface DebugSizes {
  buffers: { name: string; length: number }[];
}
```

- [ ] **Step 2: Rewrite `applyFeatures`**

Replace `DebugView.applyFeatures` body with a generic loop:

```ts
applyFeatures(msg: DebugFeatures): void {
  const { store } = this.deps;
  for (const [name, buf] of Object.entries(msg.buffers)) {
    store.set(name, buf);
  }
  // Apply autogain to the four RMS history channels (TS-side).
  for (const key of ["rms", "rmsLow", "rmsMid", "rmsHigh"] as const) {
    const buf = msg.buffers[key];
    if (buf) this.applyAutoGain(key, buf);
  }
}
```

- [ ] **Step 3: Rewrite `applyConfigured`**

Replace `DebugView.applyConfigured`'s buffer-allocation block (the chain of `store.set("waveform", new Float32Array(sizes.waveformLen))`, etc.) with:

```ts
applyConfigured(sizes: DebugSizes): void {
  this.disposeLines();
  this.peakMarkers?.dispose();

  const { store, scene } = this.deps;

  // Allocate every Float32Array the worklet will produce, NaN-filled where
  // the data semantically represents "no value yet" (candidates, beat*).
  const NAN_KEYS = new Set(["candidates", "beatGrid", "beatPulses", "beatState"]);
  for (const { name, length } of sizes.buffers) {
    const arr = new Float32Array(length);
    if (NAN_KEYS.has(name)) arr.fill(NaN);
    store.set(name, arr);
  }

  // Allocate parallel autogain buffers (TS-only, never appears in
  // sizes.buffers because the DSP doesn't produce them).
  const rmsLen = sizes.buffers.find((b) => b.name === "rms")?.length ?? 0;
  for (const k of ["rms", "rmsLow", "rmsMid", "rmsHigh"] as const) {
    store.set(`${k}Auto`, new Float32Array(rmsLen));
  }
  // ... rest of the renderer construction stays unchanged, but read sizes
  // off the descriptors via lookups (e.g. find rmsLen, spectrumLen, etc.).
}
```

For each downstream renderer construction inside `applyConfigured` that reads `sizes.waveformLen` / `sizes.spectrumLen` / etc., replace with a one-time lookup at the top of the method:
```ts
const sizeOf = (name: string) => sizes.buffers.find((b) => b.name === name)?.length ?? 0;
const waveformLen = sizeOf("waveform");
const spectrumLen = sizeOf("spectrum");
const rmsLen = sizeOf("rms");
const bufferAcfLen = sizeOf("bufferAcf");
const onsetAcfLen = sizeOf("onsetAcf");
// ... etc, only the ones actually used downstream
```

- [ ] **Step 4: Update `BeatDebugView` similarly**

In `src/render/BeatDebugView.ts`:

a) Replace the `BeatDebugFeatures` interface with:
```ts
export interface BeatDebugFeatures {
  buffers: { [name: string]: Float32Array };
}
```

b) Replace `applyFeatures` body with:
```ts
applyFeatures(msg: BeatDebugFeatures): void {
  for (const key of ["beatGrid", "beatPulses", "beatState"] as const) {
    const buf = msg.buffers[key];
    if (buf) this.store.set(key, buf);
  }
}
```

c) Replace `BeatDebugSizes` interface with:
```ts
export interface BeatDebugSizes {
  buffers: { name: string; length: number }[];
}
```

d) In `applyConfigured`, swap `sizes.rmsLen` / `sizes.onsetAcfLen` / etc. references for the same `sizeOf` lookup pattern.

- [ ] **Step 5: Update App.ts (the message routing)**

In `src/App.ts`, find where the worklet's `features` and `configured` messages are forwarded to `DebugView`. The forwarded payloads now contain `buffers` instead of named fields, but the type of the message is the same — the existing `if (msg.type === "features") debug.applyFeatures(msg)` line keeps working as long as the type is consistent.

If App.ts has any code that destructures specific buffer fields off the message (e.g. `const { waveform, spectrum } = msg`), update it to `const { buffers } = msg; const waveform = buffers.waveform`.

- [ ] **Step 6: Type-check and run JS tests**

Run: `npx tsc --noEmit`
Expected: clean.

Run: `npm test`
Expected: all tests pass. Tests that mock `DebugFeatures` / `DebugSizes` will need to switch to the new shape — update them as failures surface.

- [ ] **Step 7: Commit**

```bash
git add src/render/DebugView.ts src/render/BeatDebugView.ts src/App.ts
git commit -m "refactor(render): consume worklet buffers as string-keyed dict"
```

---

## Task 10: Switch `WorkletBridge` to forward `set_param` directly

**Files:**
- Modify: `src/params/WorkletBridge.ts`

- [ ] **Step 1: Simplify the param forwarding**

In `src/params/WorkletBridge.ts`, the existing `HOT_KEYS` array and `resolveHotValue` mapping already use the param-suffix form (`smoothingTauSecs`, etc.). The bridge already posts `{ type: "param", key: hotKey, value }`. **No changes are strictly required here** — the worklet now forwards the `key` string directly to `dsp.setParam(key, value)`, which is happy to receive any of `smoothingTauSecs`, `teaTauSecs`, `dbFloor`. `hopSize` is intercepted by the worklet itself before reaching `dsp.setParam`.

- [ ] **Step 2: Verify by reading and confirming**

Read `src/params/WorkletBridge.ts` end-to-end and confirm:
- `HOT_KEYS = ["hopSize", "smoothingTauSecs", "dbFloor", "teaTauSecs"]` — covers all the keys the new worklet handler accepts.
- `bootstrap()` posts one `param` message per hot key after the `configure` message.
- `handleChange()` posts a `param` message for any hot-key store change.

If any HOT_KEYS entry doesn't match a key recognized by `dsp.setParam` (or by the worklet's `hopSize` interceptor), update the array. Currently they all match.

- [ ] **Step 3: Run JS tests**

Run: `npm test`
Expected: all tests pass.

- [ ] **Step 4: No commit unless code changed**

If you made any actual code changes in this task, commit them:
```bash
git add src/params/WorkletBridge.ts
git commit -m "refactor(params): align WorkletBridge with new dsp.setParam API"
```

If no code changed, skip the commit and proceed.

---

## Task 11: Delete old wasm-bindgen typed methods + Rust test poke methods

**Files:**
- Modify: `crates/dsp/src/lib.rs`

After Tasks 8-9 nothing on the JS side calls the old methods. Now we delete them.

- [ ] **Step 1: Delete the 15 typed buffer getters**

In `crates/dsp/src/lib.rs`, inside the `#[wasm_bindgen] impl Dsp { ... }` block, delete:
- `pub fn waveform`
- `pub fn spectrum`
- `pub fn buffer_acf`
- `pub fn rms_history`
- `pub fn onset_history`
- `pub fn onset_acf`
- `pub fn onset_acf_enhanced`
- `pub fn candidates`
- `pub fn tea`
- `pub fn low_rms_history`
- `pub fn mid_rms_history`
- `pub fn high_rms_history`
- `pub fn beat_grid`
- `pub fn beat_pulses`
- `pub fn beat_state`

- [ ] **Step 2: Delete the 3 typed param setters**

Delete:
- `pub fn set_smoothing_tau`
- `pub fn set_tea_tau_secs`
- `pub fn set_db_floor`

(`Dsp::set_param` is the sole entry point now.)

- [ ] **Step 3: Run cargo tests**

Run: `cargo test -p dsp`
Expected: all tests pass — the test module still uses the new API for its assertions (set up in earlier tasks).

If any tests still call the deleted methods, this is the moment to update them — they should call `dsp.get_buffer("...")` and `dsp.set_param("...", v)` instead.

- [ ] **Step 4: Build wasm + run JS tests**

Run: `npm run wasm && npm test`
Expected: wasm rebuilds (the wasm-pkg shrinks — fewer exports), JS tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "refactor(dsp): delete old typed wasm-bindgen getters and setters"
```

---

## Task 12: Switch Rust integration tests to new API

**Files:**
- Modify: `crates/dsp/src/lib.rs`

The integration tests in the `#[cfg(test)] mod tests { ... }` block currently call the deleted typed methods (`dsp.spectrum()`, `dsp.beat_grid()`, etc.) — they would have started failing in Task 11 if any did. Update them now (or earlier as Task 11 surfaces them).

- [ ] **Step 1: Update test assertions**

Walk every test in `#[cfg(test)] mod tests` and replace:
| Old | New |
|---|---|
| `dsp.waveform()` | `dsp.get_buffer("waveform")` |
| `dsp.spectrum()` | `dsp.get_buffer("spectrum")` |
| `dsp.buffer_acf()` | `dsp.get_buffer("bufferAcf")` |
| `dsp.rms_history()` | `dsp.get_buffer("rms")` |
| `dsp.onset_history()` | `dsp.get_buffer("onset")` |
| `dsp.onset_acf()` | `dsp.get_buffer("onsetAcf")` |
| `dsp.onset_acf_enhanced()` | `dsp.get_buffer("onsetAcfEnhanced")` |
| `dsp.candidates()` | `dsp.get_buffer("candidates")` |
| `dsp.tea()` | `dsp.get_buffer("tea")` |
| `dsp.low_rms_history()` | `dsp.get_buffer("rmsLow")` |
| `dsp.mid_rms_history()` | `dsp.get_buffer("rmsMid")` |
| `dsp.high_rms_history()` | `dsp.get_buffer("rmsHigh")` |
| `dsp.beat_grid()` | `dsp.get_buffer("beatGrid")` |
| `dsp.beat_pulses()` | `dsp.get_buffer("beatPulses")` |
| `dsp.beat_state()` | `dsp.get_buffer("beatState")` |

(`autocorrelate(...)` direct calls in `tests::autocorrelate_helper_correctness` already point to `crate::acf::autocorrelate` from Task 2 — no change there.)

- [ ] **Step 2: Add a `Buffers` registry test**

`Dsp::buffer_descriptors()` returns `Vec<JsValue>` and is only callable under wasm — testing it from `cargo test -p dsp` is friction. Instead, test the underlying `Buffers::descriptors` and `Buffers::get` directly. Add to `crates/dsp/src/buffers.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_lists_15_named_buffers() {
        let b = Buffers::new(2048, 512);
        let d = b.descriptors();
        assert_eq!(d.len(), 15);
        assert_eq!(d[0], ("waveform", 2048));
        assert_eq!(d[1], ("spectrum", 1024));
        assert!(d.iter().any(|&(n, _)| n == "beatPulses"));
    }

    #[test]
    fn get_returns_some_for_known_keys_none_for_unknown() {
        let b = Buffers::new(2048, 512);
        assert!(b.get("waveform").is_some());
        assert!(b.get("rmsHigh").is_some());
        assert!(b.get("notARealKey").is_none());
    }
}
```

Use the simpler alternative.

- [ ] **Step 3: Run cargo tests**

Run: `cargo test -p dsp`
Expected: all tests pass, including the two new `buffers::tests`.

- [ ] **Step 4: Run JS tests + wasm rebuild**

Run: `npm run wasm && npm test`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs crates/dsp/src/buffers.rs
git commit -m "test(dsp): switch integration tests to string-keyed API + add Buffers tests"
```

---

## Task 13: Final smoke + acceptance checklist

**Files:** none modified (verification only).

- [ ] **Step 1: File sizes**

Run: `wc -l crates/dsp/src/*.rs`
Expected:
- `lib.rs` < 250 lines
- `buffers.rs` < 200 lines
- `spectrum.rs` < 200 lines
- `acf.rs` < 200 lines
- `beat.rs` < 500 lines

If `lib.rs` is over 250 lines, look for residual logic that should have moved into a stage module (constants, helper methods).

- [ ] **Step 2: All tests pass**

Run: `cargo test -p dsp && npm test`
Expected: green.

- [ ] **Step 3: Visual smoke test**

Run: `npm run dev`

In the browser at `localhost:5173`:
- Press `T` (test source) — visualizer starts.
- Verify spectrum lane animates and looks identical to before.
- Verify the four RMS-history lanes (full + low/mid/high) animate with sensible amplitudes.
- Verify the onset-history line moves with onsets.
- Verify onset-ACF and onset-ACF-enhanced lanes show peaks.
- Verify the TEA accumulator lane converges to a clear peak.
- Verify peak markers render on the autocorr lane.
- Verify the four beat-pulse squares pulse at sensible rates (cycle 1 fastest).
- Switch camera presets `1`-`6`; verify each layout renders.
- Toggle space (front↔side); verify tween works.
- Open ParamPanel; tweak `dsp.smoothingTauSecs` — confirm the spectrum visibly responds (no worklet rebuild).
- Tweak `dsp.teaTauSecs` — confirm beat tracker behavior changes (no worklet rebuild).
- Tweak `dsp.dbFloor` — confirm spectrum noise floor changes.

- [ ] **Step 4: HMR check**

While `npm run dev` is running, edit `src/render/DebugView.ts` (e.g. nudge a Y offset) and save. Verify:
- The visualizer rebuilds without flicker / errors.
- DSP state is preserved (the running RMS history doesn't reset to zero, the beat tracker doesn't lose lock).

- [ ] **Step 5: Adding-a-new-buffer audit**

Walk through what it would take to add a new buffer (mental exercise — don't actually add one):
- New `pub` field on `Buffers`. ✓ in `buffers.rs`
- New match arm in `Buffers::get`. ✓ in `buffers.rs`
- New entry in `Buffers::descriptors`. ✓ in `buffers.rs`
- Stage code writes to `&mut self.buffers.theNewKey`. ✓ in the relevant stage module
- JS-side: nothing — `DebugView.applyConfigured` allocates it via the loop, `applyFeatures` sets it in the store.
- Renderers that want to consume it: `store.get("theNewKey")` in their `source` callback.

If this audit reveals anything that would still need touching outside `buffers.rs` + the stage that writes the new buffer + the new renderer, fix it.

- [ ] **Step 6: Open the PR or merge**

Run: `git log --oneline main..HEAD`
Expected: ~10 focused refactor commits, each in a logical step.

If using the finishing-a-development-branch skill, invoke it now to choose between merge / PR / cleanup.

---

## Notes for the implementer

- **One thing breaks at a time.** The phase split (Tasks 1-6 keep the public API stable, Tasks 7-12 swap it) is deliberate. After every task, every test should pass — if it doesn't, you've gone outside the bite-sized step.
- **Don't shortcut Task 3.** The temptation is to also extract `SpectrumState` / `BeatState` while you're moving fields into `Buffers`. Resist — Task 3 has the largest blast radius (touches every reference in `Dsp::process`), and combining it with state extraction makes the diff illegible. Pure rename in Task 3, structural moves in Tasks 4-6.
- **JS field names are camelCase to match Rust struct fields exactly.** The Rust struct has `pub bufferAcf: Vec<f32>` (with `#[allow(non_snake_case)]`), the worklet posts `msg.buffers["bufferAcf"]`, the FeatureStore stores under `"bufferAcf"`. One key per buffer, used identically across the stack.
- **`#[allow(non_snake_case)]` only goes on `Buffers`.** Don't sprinkle it elsewhere. Other Rust types stay idiomatic snake_case; only the registered key vocabulary needs the camelCase carve-out.
- **Rust method names stay snake_case; wasm-bindgen converts them to JS camelCase automatically.** `dsp.get_buffer` → `dsp.getBuffer`, `dsp.buffer_descriptors` → `dsp.bufferDescriptors`, `dsp.set_param` → `dsp.setParam`. Don't add `#[wasm_bindgen(js_name = ...)]` overrides unless a name doesn't translate cleanly.
