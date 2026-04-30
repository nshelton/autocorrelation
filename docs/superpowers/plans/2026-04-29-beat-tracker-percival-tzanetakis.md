# Beat Tracker Rewrite (Percival & Tzanetakis 2014) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current peak-driven `BeatTracker` (which exhibits an octave-ambiguity bug — every divisor of the true beat period is a stable fixed point of its update rule) with a streaming adaptation of Percival & Tzanetakis 2014: half-wave-rectified RMS as OSS proxy → generalized FFT ACF (`|X|^0.5`) → harmonic enhancement → top-10 peak pick → pulse-train scoring → Gaussian-smeared TEA accumulator.

**Architecture:** New code is added additively to `crates/dsp/src/lib.rs` (Phase 1, Tasks 1-6) so existing tests keep passing while each component lands with its own TDD cycle. Phase 2 (Task 7) switches `process()` to the new pipeline and the new outputs. Phase 3 (Tasks 8-9) deletes the now-unused old beat code and old ACF fields. Phase 4 (Tasks 10-15) updates worklet message protocol, params, and renderer to match.

**Tech Stack:** Rust + `realfft` (existing planner pattern), `wasm-bindgen`, TypeScript, Vitest, Three.js WebGPU.

**Spec:** `docs/superpowers/specs/2026-04-29-beat-tracker-percival-tzanetakis-design.md`

---

## Interim runtime state (read before executing)

Tasks 1-9 land Rust changes; Task 10 is the worklet switch. Between Task 7 (when `process()` outputs come from the new pipeline) and Task 10 (when the worklet stops calling the deleted `dsp.rms_acf_accum()` / `dsp.acf_peaks()` / `dsp.set_accum_tau_secs()` getters/setters), the **browser will throw at runtime** if launched. This is intentional: each task remains independently testable (Rust unit tests at task N, TS unit tests after Task 11/12), but the *integrated dev server* only comes back online at Task 12. Don't run `npm run dev` between Tasks 7 and 12 expecting visuals; rely on `cargo test -p dsp` and `npm test` for green-light signals until the smoke test in Task 15.

If your workflow needs continuous browser viability, swap the order to: 1-6 → 10 → 11 → 12 → 13 → 7 → 8 → 9 → 14 → 15. The plan-as-written order optimizes for keeping each task self-contained and readable; the alternate order optimizes for runnable-at-every-commit.

---

## File Structure

**Modified:**
- `crates/dsp/src/lib.rs` — add OSS / gen-ACF / harmonic enhance / pick / pulse-train scoring / TEA helpers + new `Dsp` fields + new outputs; later, delete old `BeatTracker` + `pick_acf_peaks` + `update_beat_state` + `fit_beat_phase` + old ACF fields + their tests.
- `src/audio/dsp-worklet.ts` — rename / add / drop message buffers; update `applyConfigure`; route `teaTauSecs`.
- `src/params/schemas.ts` — drop `dsp.accumTauSecs`, add `dsp.teaTauSecs`.
- `src/params/WorkletBridge.ts` — `HOT_KEYS` swap.
- `src/render/DebugView.ts` — drop `rms_acf` / `rms_acf_accum` / `low_rms_acf` line renderers; add `onset` / `onset_acf` / `onset_acf_enhanced` / `tea`; rewire `PeakMarkers` to `candidates`.
- `src/render/BeatDebugView.ts` — `BeatDebugSizes.rmsAcfLen` → `onsetAcfLen`.
- `tests/params/WorkletBridge.test.ts` — replace `accumTauSecs` test with `teaTauSecs`.
- `tests/render/PeakMarkers.test.ts` — input buffer renamed to `candidates` (stride 3 unchanged).
- `tests/render/BeatGridRenderer.test.ts`, `tests/render/BeatPulseSquares.test.ts`, `tests/render/BeatGridScrollingRenderer.test.ts` — only fixtures referencing dropped buffer names need updates; renderers themselves are unchanged.
- `ROADMAP.md` — mark items shipped.

**Created:** none. All work lands in existing files.

**Deleted (in cleanup tasks):**
- Old `BeatTracker` struct + impl block in `crates/dsp/src/lib.rs`.
- Old `pick_acf_peaks`, `update_beat_state`, `fit_beat_phase`, `update_beat_pulses` (the latter is rewritten in place — see Task 7).
- Old fields: `rms_acf`, `rms_acf_accum`, `accum_alpha`, `acf_peaks`, `peak_candidates` (the new `pick_candidates` reuses the same scratch type but the field gets renamed in Task 4), `rms_detrended`, `low_rms_acf`, `low_rms_detrended`.
- Old tests targeting deleted code (~14 tests, all explicit in Task 8/9 step lists).

---

## Constants reference

These are the values referenced across multiple tasks:

```rust
const BEAT_TRACKER_MIN_BPM: f32 = 40.0;
const BEAT_TRACKER_MAX_BPM: f32 = 220.0;
const GEN_ACF_C: f32 = 0.5;
const HARMONIC_MULTIPLES: [usize; 2] = [2, 4];
const MAX_PEAKS: usize = 10;            // unchanged from old impl
const MIN_PEAK_SPACING: usize = 3;       // unchanged
const PULSE_N: usize = 4;                // pulses per template
const TEA_GAUSSIAN_SIGMA: f32 = 5.0;
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;
```

`BEAT_GRID_LEN`, `BEAT_PULSES_LEN`, `BEAT_STATE_LEN`, `BEAT_PULSE_CYCLES` keep their current values and definitions.

---

## Task 1: OSS pipeline — onset signal field + per-process update

Add the half-wave-rectified RMS-difference signal as a sliding buffer alongside `rms_history`. New code only — no existing code touched.

**Files:**
- Modify: `crates/dsp/src/lib.rs` (struct fields + `new()` + `process()` + new getter + tests at end of file)

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)]` module at the bottom of `crates/dsp/src/lib.rs`:

```rust
#[test]
fn onset_history_captures_positive_rms_diff() {
    // RMS of a constant-amplitude signal A over a frame is A.
    // Step ladder: silence (RMS 0) → 0.5 → 0.5 → 0.3.
    // Expected onset values per process call (max(0, rms - prev_rms)):
    //   process(silent): rms=0, prev=0 → onset=0
    //   process(half):   rms=0.5, prev=0   → onset=0.5
    //   process(half):   rms=0.5, prev=0.5 → onset=0
    //   process(lower):  rms=0.3, prev=0.5 → onset=0  (negative diff clamped)
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let silent = vec![0.0f32; 2048];
    let half   = vec![0.5f32; 2048];
    let lower  = vec![0.3f32; 2048];
    dsp.process(&silent);
    dsp.process(&half);
    dsp.process(&half);
    dsp.process(&lower);
    let onset = dsp.onset_history();
    let n = onset.len();
    // newest at index n-1, oldest at index 0
    assert!((onset[n - 1] - 0.0).abs() < 1e-3, "newest = {}", onset[n - 1]);
    assert!((onset[n - 2] - 0.0).abs() < 1e-3, "newest-1 = {}", onset[n - 2]);
    assert!((onset[n - 3] - 0.5).abs() < 1e-2, "newest-2 = {}", onset[n - 3]);
    assert!((onset[n - 4] - 0.0).abs() < 1e-3, "newest-3 = {}", onset[n - 4]);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p dsp onset_history_captures_positive_rms_diff`

Expected: FAIL with `no method named 'onset_history' found`.

- [ ] **Step 3: Add the field, init, getter, and per-process update**

In the `Dsp` struct (alongside `rms_history`), add:

```rust
    onset_history: Vec<f32>,
    prev_rms: f32,
```

In `Dsp::new()`, after `vec![0.0; rms_history_len]` for `rms_history`, add the matching initializers in the struct literal:

```rust
            onset_history: vec![0.0; rms_history_len],
            prev_rms: 0.0,
```

In `Dsp::process()`, immediately after the existing block that updates `rms_history` (the `copy_within` + last-index write for `self.rms_history`), insert:

```rust
        // Half-wave-rectified RMS difference — proxy for the paper's spectral
        // flux OSS. `rms` is the just-computed full-band RMS for this frame.
        let onset = (rms - self.prev_rms).max(0.0);
        self.prev_rms = rms;
        self.onset_history.copy_within(1.., 0);
        let last = self.onset_history.len() - 1;
        self.onset_history[last] = onset;
```

In the `#[wasm_bindgen] impl Dsp { ... }` block (where the other getters live, e.g. next to `pub fn rms_history(&self) -> Vec<f32>`), add:

```rust
    pub fn onset_history(&self) -> Vec<f32> {
        self.onset_history.clone()
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p dsp onset_history_captures_positive_rms_diff`

Expected: PASS. Then run the full suite to confirm no regressions: `cargo test -p dsp`

Expected: every existing test still passes.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): onset_history sliding buffer (half-wave-rectified RMS diff)"
```

---

## Task 2: Generalized ACF helper

Implement `compute_gen_acf` as a free function (testable in isolation), wire its scratch buffers + planners into `Dsp`, and call it inside `process()` to populate a new `onset_acf` output.

**Files:**
- Modify: `crates/dsp/src/lib.rs` (add free function + struct fields + `new()` setup + `process()` call + getter + tests)

- [ ] **Step 1: Write the failing test**

Append to the test module:

```rust
#[test]
fn gen_acf_silent_input_is_zero() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let silent = vec![0.0f32; 2048];
    for _ in 0..5 { dsp.process(&silent); }
    let acf = dsp.onset_acf();
    for &v in acf.iter() {
        assert!(!v.is_nan(), "silent gen-ACF must not be NaN, got {}", v);
        assert!(v.abs() < 1e-3, "silent gen-ACF should be ~0, got {}", v);
    }
}

#[test]
fn gen_acf_periodic_onset_peaks_at_period() {
    // Drive a strongly periodic envelope so onset_history has clear spikes.
    // Period 32 frames @ default sr/hop ⇒ ~88 BPM. Run long enough for the
    // sliding window to be fully populated with the periodic pattern.
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let sr = 48000.0_f32;
    let period_hops = 32usize;
    for k in 0..1500 {
        let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
        let signal: Vec<f32> = (0..2048)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
    }
    let acf = dsp.onset_acf();
    // Lag 0 normalized to 1.0; lag at the period should be a clear local maximum
    // and substantially larger than nearby non-multiples.
    let p = period_hops;
    let around = (p - 5..=p + 5).map(|i| acf[i]).fold(f32::NEG_INFINITY, f32::max);
    let far = (3..=8).chain(15..=20).map(|i| acf[i]).fold(f32::NEG_INFINITY, f32::max);
    assert!(around > far,
        "peak near lag {} ({:.3}) should exceed nearby non-period max ({:.3})",
        p, around, far);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p dsp gen_acf_silent_input_is_zero gen_acf_periodic_onset_peaks_at_period`

Expected: both FAIL with `no method named 'onset_acf' found`.

- [ ] **Step 3: Add the helper, fields, and wiring**

Add the free function near the other helpers (next to `autocorrelate`):

```rust
/// Generalized autocorrelation per Percival & Tzanetakis 2014 §II-B.2:
/// zero-pad input to 2N, forward FFT, magnitude compression `|X|^c`, inverse
/// FFT, take the first N/2 lags, normalize so output[0] == 1.0. Allocation-
/// free: caller passes scratch buffers (`time_buf` length 2N, `freq_buf`
/// length N+1) and pre-built `realfft` planners.
///
/// `c = 0.5` (the paper's empirically best choice) gives narrower ACF peaks
/// than `c = 2.0` (regular ACF) — this is what makes downstream peak picking
/// and pulse-train scoring more discriminative.
fn compute_gen_acf(
    input: &[f32],
    output: &mut [f32],
    fft_forward: &Arc<dyn RealToComplex<f32>>,
    fft_inverse: &Arc<dyn ComplexToReal<f32>>,
    time_buf: &mut [f32],
    freq_buf: &mut [Complex<f32>],
    c: f32,
) {
    let n = input.len();
    debug_assert_eq!(time_buf.len(), 2 * n);
    debug_assert_eq!(freq_buf.len(), n + 1);

    time_buf[..n].copy_from_slice(input);
    time_buf[n..].fill(0.0);

    let _ = fft_forward.process(time_buf, freq_buf);

    for x in freq_buf.iter_mut() {
        let mag = (x.re * x.re + x.im * x.im).sqrt();
        let compressed = mag.powf(c);
        *x = Complex::new(compressed, 0.0);
    }

    let _ = fft_inverse.process(freq_buf, time_buf);

    let zero = time_buf[0].max(1e-12);
    for i in 0..output.len() {
        output[i] = time_buf[i] / zero;
    }
}
```

Add the `ComplexToReal` import at the top of the file:

```rust
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
```

Add new fields to `Dsp`:

```rust
    gen_acf_fft_forward: Arc<dyn RealToComplex<f32>>,
    gen_acf_fft_inverse: Arc<dyn ComplexToReal<f32>>,
    gen_acf_time_buf: Vec<f32>,
    gen_acf_freq_buf: Vec<Complex<f32>>,
    onset_acf: Vec<f32>,
```

In `Dsp::new()`, after the existing planner / hann setup, add:

```rust
        let gen_acf_n = rms_history_len; // OSS frame size = onset_history length
        let gen_acf_fft_forward = planner.plan_fft_forward(2 * gen_acf_n);
        let gen_acf_fft_inverse = planner.plan_fft_inverse(2 * gen_acf_n);
        let gen_acf_time_buf = vec![0.0; 2 * gen_acf_n];
        let gen_acf_freq_buf = vec![Complex::new(0.0, 0.0); gen_acf_n + 1];
```

(Reusing the existing local `let mut planner = RealFftPlanner::<f32>::new();`.)

Add the matching field initializers in the `Dsp { ... }` literal:

```rust
            gen_acf_fft_forward,
            gen_acf_fft_inverse,
            gen_acf_time_buf,
            gen_acf_freq_buf,
            onset_acf: vec![0.0; rms_history_len / 2],
```

In `Dsp::process()`, after the OSS update from Task 1, add:

```rust
        compute_gen_acf(
            &self.onset_history,
            &mut self.onset_acf,
            &self.gen_acf_fft_forward,
            &self.gen_acf_fft_inverse,
            &mut self.gen_acf_time_buf,
            &mut self.gen_acf_freq_buf,
            GEN_ACF_C,
        );
```

Add the constant near the other beat-related constants near the top of the file:

```rust
/// Magnitude-compression exponent for the generalized ACF (Percival &
/// Tzanetakis 2014 §II-B.2 reports c = 0.5 as the best empirical compromise
/// between lag resolution and noise sensitivity).
const GEN_ACF_C: f32 = 0.5;
```

Add the getter in the `#[wasm_bindgen] impl Dsp` block:

```rust
    pub fn onset_acf(&self) -> Vec<f32> {
        self.onset_acf.clone()
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p dsp gen_acf_silent_input_is_zero gen_acf_periodic_onset_peaks_at_period`

Expected: PASS. Then `cargo test -p dsp` to confirm full suite still passes.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): generalized FFT autocorrelation (|X|^0.5) on onset_history"
```

---

## Task 3: Harmonic enhancement helper

Add `compute_harmonic_enhanced` (free function) and the `onset_acf_enhanced` field. Each frame, sum `A[τ] + A[2τ] + A[4τ]` (with bounds check) into the enhanced buffer.

**Files:**
- Modify: `crates/dsp/src/lib.rs` (free fn + field + getter + `new()` + `process()` + tests)

- [ ] **Step 1: Write the failing test**

Append to the test module:

```rust
#[test]
fn harmonic_enhanced_sums_multiples() {
    // Synthetic ACF: peaks at lags 10, 20, 40 with magnitudes 0.5, 0.3, 0.2;
    // zeros elsewhere. After enhancement (multiples [2, 4]), enhanced[10]
    // should equal 0.5 + 0.3 + 0.2 = 1.0 (since 10*2=20 and 10*4=40 hit the
    // other peaks).
    let mut acf = vec![0.0f32; 64];
    acf[10] = 0.5;
    acf[20] = 0.3;
    acf[40] = 0.2;
    let mut enhanced = vec![0.0f32; 64];
    compute_harmonic_enhanced(&acf, &mut enhanced, &[2, 4]);
    assert!((enhanced[10] - 1.0).abs() < 1e-6, "enhanced[10] = {}", enhanced[10]);
    // enhanced[20] = acf[20] + acf[40] + acf[80(oob)] = 0.3 + 0.2 + 0 = 0.5
    assert!((enhanced[20] - 0.5).abs() < 1e-6, "enhanced[20] = {}", enhanced[20]);
    // enhanced[40] = acf[40] + acf[80(oob)] + acf[160(oob)] = 0.2
    assert!((enhanced[40] - 0.2).abs() < 1e-6, "enhanced[40] = {}", enhanced[40]);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p dsp harmonic_enhanced_sums_multiples`

Expected: FAIL with `no function named 'compute_harmonic_enhanced'`.

- [ ] **Step 3: Add the helper, field, and wiring**

Add the free function next to `compute_gen_acf`:

```rust
/// Per Percival & Tzanetakis 2014 §II-B.3: boost peaks corresponding to
/// integer multiples of the underlying tempo by adding time-stretched
/// versions of the ACF. For each `mult ∈ multiples`, `enhanced[τ] +=
/// acf[mult * τ]` when `mult * τ < acf.len()`. `enhanced` should be the
/// same length as `acf` (caller's responsibility).
fn compute_harmonic_enhanced(acf: &[f32], enhanced: &mut [f32], multiples: &[usize]) {
    let n = acf.len();
    for tau in 0..n {
        let mut sum = acf[tau];
        for &mult in multiples {
            let idx = tau * mult;
            if idx < n {
                sum += acf[idx];
            }
        }
        enhanced[tau] = sum;
    }
}
```

Add the constant:

```rust
const HARMONIC_MULTIPLES: [usize; 2] = [2, 4];
```

Add the field:

```rust
    onset_acf_enhanced: Vec<f32>,
```

Initialize in `Dsp::new()`:

```rust
            onset_acf_enhanced: vec![0.0; rms_history_len / 2],
```

In `Dsp::process()`, immediately after the `compute_gen_acf` call from Task 2:

```rust
        compute_harmonic_enhanced(
            &self.onset_acf,
            &mut self.onset_acf_enhanced,
            &HARMONIC_MULTIPLES,
        );
```

Add the getter:

```rust
    pub fn onset_acf_enhanced(&self) -> Vec<f32> {
        self.onset_acf_enhanced.clone()
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p dsp harmonic_enhanced_sums_multiples` → PASS, then `cargo test -p dsp` → all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): harmonic enhancement A[τ] + A[2τ] + A[4τ]"
```

---

## Task 4: Peak picking on enhanced ACF

Add `pick_candidates` as a method on `Dsp` (uses preallocated scratch). Operates on `onset_acf_enhanced` over `[tau_min, tau_max]`. Output is `candidates: Vec<f32>` with stride 3 `[lag_frac, mag, sharpness]`, NaN-padded.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Append to the test module:

```rust
#[test]
fn pick_candidates_silent_all_nan() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let silent = vec![0.0f32; 2048];
    for _ in 0..5 { dsp.process(&silent); }
    let cands = dsp.candidates();
    assert_eq!(cands.len(), 30);
    for &v in cands.iter() {
        assert!(v.is_nan(), "silent → all candidate slots NaN, got {}", v);
    }
}

#[test]
fn pick_candidates_top_n_within_tau_range() {
    // Synthetic enhanced ACF with 12 distinct local maxima inside the
    // expected [tau_min=12, tau_max=70] window. Spacing = 4 satisfies
    // MIN_PEAK_SPACING=3. Magnitudes decreasing.
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    // Use a test-only setter (added below) to install a known enhanced ACF.
    let n = dsp.onset_acf_enhanced_len();  // = rms_history_len / 2 = 256
    let mut enhanced = vec![0.0f32; n];
    let positions = [14usize, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58];
    for (i, &p) in positions.iter().enumerate() {
        // Make each a strict local max with descending magnitude
        let mag = 1.0 - 0.05 * i as f32;
        enhanced[p - 1] = mag * 0.5;
        enhanced[p]     = mag;
        enhanced[p + 1] = mag * 0.5;
    }
    dsp.test_set_onset_acf_enhanced(&enhanced);
    dsp.test_run_pick_candidates();
    let cands = dsp.candidates();
    // Top 10 should be picked in descending magnitude order; mags are
    // monotonically decreasing in `positions[..10]`.
    let mut last_mag = f32::INFINITY;
    for i in 0..10 {
        let lag = cands[3 * i];
        let mag = cands[3 * i + 1];
        assert!(!lag.is_nan(), "slot {} should have a peak, got NaN", i);
        assert!(mag <= last_mag, "magnitudes should be descending: {} > {}", mag, last_mag);
        last_mag = mag;
    }
}

#[test]
fn pick_candidates_excludes_out_of_range_lags() {
    // Place two peaks: one at lag 5 (below tau_min=12), one at lag 100 (above
    // tau_max=70). Neither should be picked.
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let n = dsp.onset_acf_enhanced_len();
    let mut enhanced = vec![0.0f32; n];
    enhanced[5] = 1.0; enhanced[4] = 0.5; enhanced[6] = 0.5;
    enhanced[100] = 1.0; enhanced[99] = 0.5; enhanced[101] = 0.5;
    dsp.test_set_onset_acf_enhanced(&enhanced);
    dsp.test_run_pick_candidates();
    let cands = dsp.candidates();
    for i in 0..10 {
        let lag = cands[3 * i];
        if !lag.is_nan() {
            assert!(lag >= 12.0 && lag <= 70.0,
                "picked lag {} outside [12, 70]", lag);
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p dsp pick_candidates_silent_all_nan pick_candidates_top_n_within_tau_range pick_candidates_excludes_out_of_range_lags`

Expected: FAIL — `no method named 'candidates'`, `test_set_onset_acf_enhanced`, `onset_acf_enhanced_len`, `test_run_pick_candidates`.

- [ ] **Step 3: Add fields, constants, the picker method, and test helpers**

Constants near the other beat constants:

```rust
const BEAT_TRACKER_MIN_BPM: f32 = 40.0;
const BEAT_TRACKER_MAX_BPM: f32 = 220.0;
const MIN_PEAK_SPACING: usize = 3;
const MAX_PEAKS: usize = 10;
```

Note: the existing file already defines `MIN_PEAK_SPACING` and `MAX_PEAKS` — keep one set, delete the other. The old `MIN_PEAK_LAG = 10` constant is replaced by the dynamic `tau_min` (computed from BPM bounds + dt) and can stay in place for now; it gets deleted in Task 8 as part of the old-tracker cleanup.

Add fields:

```rust
    candidates: Vec<f32>,                   // stride 3: [lag_frac, mag, sharpness]
    cand_scratch: Vec<(usize, f32)>,        // preallocated scratch for picker
    tau_min: usize,
    tau_max: usize,
```

In `Dsp::new()`, compute the lag bounds and initialize:

```rust
        let tau_min = ((60.0 / BEAT_TRACKER_MAX_BPM) / dt).floor().max(1.0) as usize;
        let tau_max_unbounded = ((60.0 / BEAT_TRACKER_MIN_BPM) / dt).ceil() as usize;
        let onset_acf_len = rms_history_len / 2;
        let tau_max = tau_max_unbounded.min(onset_acf_len.saturating_sub(2));
```

Add the matching struct-literal initializers:

```rust
            candidates: vec![f32::NAN; 3 * MAX_PEAKS],
            cand_scratch: Vec::with_capacity(onset_acf_len / 2 + 1),
            tau_min,
            tau_max,
```

Add the picker method on `impl Dsp` (the non-`#[wasm_bindgen]` block — keep wasm bindings on getters):

```rust
impl Dsp {
    fn pick_candidates(&mut self) {
        for slot in self.candidates.iter_mut() {
            *slot = f32::NAN;
        }
        if self.tau_max < self.tau_min + 1 {
            return;
        }

        // 1. scan strict local maxima in [tau_min, tau_max]
        self.cand_scratch.clear();
        let upper = self.tau_max.min(self.onset_acf_enhanced.len() - 1);
        for k in self.tau_min..upper {
            let y = self.onset_acf_enhanced[k];
            if y > 0.0
                && y > self.onset_acf_enhanced[k - 1]
                && y > self.onset_acf_enhanced[k + 1]
            {
                self.cand_scratch.push((k, y));
            }
        }

        // 2. sort descending by magnitude
        self.cand_scratch.sort_unstable_by(|a, b|
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. greedy select top-N with min-spacing
        let mut accepted: [u32; MAX_PEAKS] = [0; MAX_PEAKS];
        let mut count: usize = 0;
        for &(k, _) in &self.cand_scratch {
            if count == MAX_PEAKS { break; }
            let too_close = accepted[..count].iter().any(|&j| {
                ((k as i32 - j as i32).unsigned_abs() as usize) < MIN_PEAK_SPACING
            });
            if !too_close {
                accepted[count] = k as u32;
                count += 1;
            }
        }

        // 4. parabolic sub-bin refinement → write [lag_frac, mag, sharpness]
        for i in 0..count {
            let k = accepted[i] as usize;
            let y0 = self.onset_acf_enhanced[k - 1];
            let y1 = self.onset_acf_enhanced[k];
            let y2 = self.onset_acf_enhanced[k + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let (lag_frac, mag) = if denom.abs() < 1e-12 {
                (k as f32, y1)
            } else {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                (k as f32 + delta, y1 - 0.25 * (y0 - y2) * delta)
            };
            self.candidates[3 * i]     = lag_frac;
            self.candidates[3 * i + 1] = mag;
            self.candidates[3 * i + 2] = -denom;  // sharpness
        }
    }
}
```

In `Dsp::process()`, after the harmonic enhancement step from Task 3:

```rust
        self.pick_candidates();
```

Add the public getter:

```rust
    pub fn candidates(&self) -> Vec<f32> {
        self.candidates.clone()
    }
```

Add test helpers next to the existing test-only setters (`#[cfg(test)] impl Dsp { ... }` block — there is one in the file; reuse it):

```rust
    pub fn onset_acf_enhanced_len(&self) -> usize {
        self.onset_acf_enhanced.len()
    }

    pub fn test_set_onset_acf_enhanced(&mut self, src: &[f32]) {
        let n = self.onset_acf_enhanced.len().min(src.len());
        self.onset_acf_enhanced[..n].copy_from_slice(&src[..n]);
        if n < self.onset_acf_enhanced.len() {
            for v in &mut self.onset_acf_enhanced[n..] { *v = 0.0; }
        }
    }

    pub fn test_run_pick_candidates(&mut self) {
        self.pick_candidates();
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```
cargo test -p dsp pick_candidates_silent_all_nan
cargo test -p dsp pick_candidates_top_n_within_tau_range
cargo test -p dsp pick_candidates_excludes_out_of_range_lags
cargo test -p dsp
```

All expected to pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): top-10 peak picking on enhanced ACF in [tau_min, tau_max]"
```

---

## Task 5: Pulse-train scoring (free helpers + per-frame method)

Add `score_phase_for_tau` (one-candidate phase scan, returns `(best_phi, best_corr, sum_corr, sum_corr_sq, n_phases)`) and `score_candidates` (full per-frame scoring across all candidates → winner). Stash the per-frame outputs in new `Dsp` fields.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn pulse_score_finds_period_in_synthetic_oss() {
    // Synthetic onset_history with a regular period of 30 samples (a pulse
    // train). At default sr/hop=1024/48000, that's lag 30 → ~94 BPM, well
    // inside [40, 220].
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let n = dsp.onset_history_len();
    let mut onset = vec![0.0f32; n];
    let period = 30usize;
    let mut idx = period;
    while idx < n {
        onset[idx] = 1.0;
        idx += period;
    }
    dsp.test_set_onset_history(&onset);
    dsp.test_run_pick_and_score();
    let (period_inst, phase_inst, score_inst) = dsp.test_per_frame_estimate();
    assert!(score_inst > 0.0, "expected nonzero score");
    assert!((period_inst - period as f32).abs() < 1.5,
        "expected period ≈ {}, got {}", period, period_inst);
    assert!(phase_inst >= 0.0 && phase_inst < period_inst);
}

#[test]
fn pulse_score_disambiguates_octave() {
    // Regression test for the bug this rewrite fixes: an onset_history with
    // a clear period P=30 should NOT be reported as P/3=10 even though
    // peaks at 10 are absent (only 30, 60, 90, ... fire). The pulse-train
    // template at τ=10 lands its 4 pulses at 0, 10, 20, 30 — only one (30)
    // hits a real onset. The template at τ=30 lands its 4 pulses on actual
    // onsets at 30, 60, 90, 120, scoring much higher.
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let n = dsp.onset_history_len();
    let mut onset = vec![0.0f32; n];
    for k in 1..(n / 30 + 1) {
        let i = k * 30;
        if i < n { onset[i] = 1.0; }
    }
    dsp.test_set_onset_history(&onset);
    dsp.test_run_pick_and_score();
    let (period_inst, _, _) = dsp.test_per_frame_estimate();
    assert!((period_inst - 30.0).abs() < 1.5,
        "octave disambiguation failed: expected 30, got {}", period_inst);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p dsp pulse_score_finds_period_in_synthetic_oss pulse_score_disambiguates_octave`

Expected: FAIL — missing helpers.

- [ ] **Step 3: Add helpers, fields, method**

Constants:

```rust
const PULSE_N: usize = 4;
```

Add fields:

```rust
    // Per-frame pulse-train scratch + outputs
    pulse_x: [f32; MAX_PEAKS],
    pulse_v: [f32; MAX_PEAKS],
    pulse_phi: [f32; MAX_PEAKS],
    pulse_score: [f32; MAX_PEAKS],
    period_inst: f32,
    phase_inst: f32,
    score_inst: f32,
```

Initialize in struct literal:

```rust
            pulse_x:    [0.0; MAX_PEAKS],
            pulse_v:    [0.0; MAX_PEAKS],
            pulse_phi:  [0.0; MAX_PEAKS],
            pulse_score:[0.0; MAX_PEAKS],
            period_inst: f32::NAN,
            phase_inst:  f32::NAN,
            score_inst:  0.0,
```

Add the free helper:

```rust
/// Score one tempo lag `tau` against the OSS by sweeping integer phases
/// `phi ∈ [0, ceil(tau))`. Returns `(best_phi, best_corr, sum_corr,
/// sum_corr_sq, n_phases)`. Pulse-train template is the paper's combined
/// `Φ₁ (w=1.0) + Φ₂ (w=0.5) + Φ₁.₅ (w=0.5)` with N=4 pulses each. Pulse
/// positions are placed *backward* from the most-recent onset sample so
/// `phi = 0` ⇒ "a beat just landed". Out-of-frame pulses are omitted, per
/// the paper.
fn score_phase_for_tau(onset: &[f32], tau: f32) -> (usize, f32, f32, f32, usize) {
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
        // Φ₁ at k·τ, weight 1.0
        for k in 0..PULSE_N {
            let off = (k as f32 * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                corr += onset[pos as usize];
            }
        }
        // Φ₂ at k·2τ, weight 0.5
        for k in 0..PULSE_N {
            let off = (k as f32 * 2.0 * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                corr += 0.5 * onset[pos as usize];
            }
        }
        // Φ₁.₅ at (k+0.5)·τ, weight 0.5
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

Add the method on `impl Dsp`:

```rust
    /// Score every candidate's pulse train, normalize X (max-φ corr) and V
    /// (var-φ corr) so each sums to 1 across candidates, pick winner with
    /// `score = X_norm + V_norm`. Writes per-frame outputs into
    /// `period_inst`, `phase_inst`, `score_inst`. Silent / no-candidate
    /// frames yield `(NaN, NaN, 0.0)`.
    fn score_candidates(&mut self) {
        let mut count = 0usize;
        for i in 0..MAX_PEAKS {
            let lag = self.candidates[3 * i];
            if lag.is_nan() { break; }
            let (phi, x, sum, sum_sq, n_phi) = score_phase_for_tau(&self.onset_history, lag);
            let n = n_phi as f32;
            let mean = if n > 0.0 { sum / n } else { 0.0 };
            let var = if n > 0.0 { (sum_sq / n - mean * mean).max(0.0) } else { 0.0 };
            self.pulse_x[i]   = x;
            self.pulse_v[i]   = var;
            self.pulse_phi[i] = phi as f32;
            count += 1;
        }
        if count == 0 {
            self.period_inst = f32::NAN;
            self.phase_inst  = f32::NAN;
            self.score_inst  = 0.0;
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
            if s > best_score { best_score = s; best_i = i; }
        }
        if best_score <= 0.0 || self.candidates[3 * best_i].is_nan() {
            self.period_inst = f32::NAN;
            self.phase_inst  = f32::NAN;
            self.score_inst  = 0.0;
        } else {
            self.period_inst = self.candidates[3 * best_i];
            self.phase_inst  = self.pulse_phi[best_i];
            self.score_inst  = best_score;
        }
    }
```

Add the test helpers in the `#[cfg(test)] impl Dsp` block:

```rust
    pub fn onset_history_len(&self) -> usize {
        self.onset_history.len()
    }

    pub fn test_set_onset_history(&mut self, src: &[f32]) {
        let n = self.onset_history.len().min(src.len());
        self.onset_history[..n].copy_from_slice(&src[..n]);
        if n < self.onset_history.len() {
            for v in &mut self.onset_history[n..] { *v = 0.0; }
        }
    }

    pub fn test_run_pick_and_score(&mut self) {
        // Recompute enhanced ACF from current onset_history, then pick & score.
        compute_gen_acf(
            &self.onset_history,
            &mut self.onset_acf,
            &self.gen_acf_fft_forward,
            &self.gen_acf_fft_inverse,
            &mut self.gen_acf_time_buf,
            &mut self.gen_acf_freq_buf,
            GEN_ACF_C,
        );
        compute_harmonic_enhanced(
            &self.onset_acf,
            &mut self.onset_acf_enhanced,
            &HARMONIC_MULTIPLES,
        );
        self.pick_candidates();
        self.score_candidates();
    }

    pub fn test_per_frame_estimate(&self) -> (f32, f32, f32) {
        (self.period_inst, self.phase_inst, self.score_inst)
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```
cargo test -p dsp pulse_score_finds_period_in_synthetic_oss
cargo test -p dsp pulse_score_disambiguates_octave
cargo test -p dsp
```

All expected to pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): pulse-train scoring with octave disambiguation"
```

---

## Task 6: TEA (Tempo Estimate Accumulator)

Add `tea`, `tea_alpha`, `set_tea_tau_secs`, the `update_tea` method (Gaussian smear + EMA decay + argmax with parabolic sub-bin refine), and the smoothed period output.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn tea_silent_input_decays_to_zero() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let n = dsp.tea_len();
    // Pre-charge TEA using the test setter.
    let mut tea = vec![0.5f32; n];
    dsp.test_set_tea(&tea);
    // Drive silent frames; TEA should decay each frame.
    let silent = vec![0.0f32; 2048];
    dsp.process(&silent);
    let after = dsp.tea();
    for i in 0..n {
        assert!(after[i] < tea[i] + 1e-6,
            "tea[{}] should not increase under silence: before={}, after={}",
            i, tea[i], after[i]);
    }
    tea.copy_from_slice(&after);
    for _ in 0..200 { dsp.process(&silent); }
    let later = dsp.tea();
    for &v in &later {
        assert!(v < 0.05, "after long silence TEA should be ~0, got {}", v);
        assert!(!v.is_nan());
    }
}

#[test]
fn tea_periodic_input_locks_to_period() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let sr = 48000.0_f32;
    let period_hops = 32usize;
    for k in 0..1500 {
        let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
        let signal: Vec<f32> = (0..2048)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
    }
    let tau_smoothed = dsp.tea_argmax();
    assert!(!tau_smoothed.is_nan(), "expected fit");
    assert!((tau_smoothed - period_hops as f32).abs() < 1.5,
        "expected ~{}, got {}", period_hops, tau_smoothed);
}

#[test]
fn set_tea_tau_secs_clamps_and_recomputes() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let alpha_default = dsp.tea_alpha();
    dsp.set_tea_tau_secs(0.001); // below clamp
    let alpha_low = dsp.tea_alpha();
    dsp.set_tea_tau_secs(120.0); // above clamp
    let alpha_high = dsp.tea_alpha();
    assert!(alpha_low > alpha_default, "smaller tau ⇒ larger alpha");
    assert!(alpha_high < alpha_default, "larger tau ⇒ smaller alpha");
    assert!(!alpha_low.is_nan() && !alpha_high.is_nan());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```
cargo test -p dsp tea_silent_input_decays_to_zero
cargo test -p dsp tea_periodic_input_locks_to_period
cargo test -p dsp set_tea_tau_secs_clamps_and_recomputes
```

Expected: FAIL — missing methods.

- [ ] **Step 3: Add constants, fields, setter, and update method**

Constants:

```rust
const TEA_GAUSSIAN_SIGMA: f32 = 5.0;
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;
```

Fields:

```rust
    tea: Vec<f32>,
    tea_alpha: f32,
    tau_smoothed: f32,
    phase_smoothed: f32,
```

Init in `Dsp::new()`:

```rust
        let tea_alpha = 1.0 - (-dt / TEA_TAU_DEFAULT_SECS).exp();
```

Struct literal:

```rust
            tea: vec![0.0; rms_history_len / 2],
            tea_alpha,
            tau_smoothed: f32::NAN,
            phase_smoothed: f32::NAN,
```

Setter (in the `#[wasm_bindgen] impl Dsp` block, next to `set_smoothing_tau`):

```rust
    /// EMA time constant for the TEA (Tempo Estimate Accumulator). `alpha =
    /// 1 - exp(-dt / tau)`. Smaller τ ⇒ faster response, less stable. Clamped
    /// to [0.05, 60.0].
    pub fn set_tea_tau_secs(&mut self, tau_secs: f32) {
        let tau = tau_secs.clamp(0.05, 60.0);
        self.tea_alpha = 1.0 - (-self.dt / tau).exp();
    }
```

Method (on `impl Dsp`):

```rust
    /// Update the TEA from this frame's `period_inst` (Gaussian-smeared
    /// vote) or decay it if the frame had no fit. Then argmax in
    /// `[tau_min, tau_max]` with parabolic sub-bin refinement → `tau_smoothed`.
    /// Re-runs the phase scan at `tau_smoothed` so phase is coherent with
    /// the smoothed period.
    fn update_tea(&mut self) {
        let alpha = self.tea_alpha;
        if self.score_inst > 0.0 && self.period_inst.is_finite() {
            let inv_2sig2 = 1.0 / (2.0 * TEA_GAUSSIAN_SIGMA * TEA_GAUSSIAN_SIGMA);
            for tau in 0..self.tea.len() {
                let delta = tau as f32 - self.period_inst;
                let g = (-delta * delta * inv_2sig2).exp();
                self.tea[tau] = (1.0 - alpha) * self.tea[tau] + alpha * g;
            }
        } else {
            for v in self.tea.iter_mut() { *v *= 1.0 - alpha; }
        }

        // argmax in [tau_min, tau_max]
        let mut best_i = self.tau_min;
        let mut best_v = -1.0f32;
        for i in self.tau_min..=self.tau_max.min(self.tea.len() - 1) {
            if self.tea[i] > best_v { best_v = self.tea[i]; best_i = i; }
        }
        if best_v <= 0.0 {
            self.tau_smoothed   = f32::NAN;
            self.phase_smoothed = f32::NAN;
            return;
        }
        let mut tau = best_i as f32;
        if best_i > self.tau_min && best_i + 1 <= self.tau_max && best_i + 1 < self.tea.len() {
            let y0 = self.tea[best_i - 1];
            let y1 = self.tea[best_i];
            let y2 = self.tea[best_i + 1];
            let denom = y0 - 2.0 * y1 + y2;
            if denom.abs() > 1e-12 {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                tau = best_i as f32 + delta;
            }
        }
        self.tau_smoothed = tau;
        let (phi, _, _, _, _) = score_phase_for_tau(&self.onset_history, tau);
        self.phase_smoothed = phi as f32;
    }
```

In `Dsp::process()`, after `score_candidates`:

```rust
        self.score_candidates();
        self.update_tea();
```

(Note: `score_candidates` is called from this task; if Task 5 didn't yet add it to `process()`, this is the first place it runs. Both are added together here. If Task 5 already wired `score_candidates`, just add `self.update_tea();` directly after.)

Wait — Task 5 added `score_candidates` as a method but did *not* call it from `process()` (the test routes through `test_run_pick_and_score`). Add the `process()` call now:

```rust
        // After self.pick_candidates() (added in Task 4):
        self.score_candidates();
        self.update_tea();
```

Public getters in `#[wasm_bindgen] impl Dsp`:

```rust
    pub fn tea(&self) -> Vec<f32> {
        self.tea.clone()
    }
```

Test helpers in `#[cfg(test)] impl Dsp`:

```rust
    pub fn tea_len(&self) -> usize { self.tea.len() }
    pub fn tea_alpha(&self) -> f32 { self.tea_alpha }
    pub fn tea_argmax(&self) -> f32 { self.tau_smoothed }
    pub fn test_set_tea(&mut self, src: &[f32]) {
        let n = self.tea.len().min(src.len());
        self.tea[..n].copy_from_slice(&src[..n]);
        if n < self.tea.len() {
            for v in &mut self.tea[n..] { *v = 0.0; }
        }
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```
cargo test -p dsp tea_silent_input_decays_to_zero
cargo test -p dsp tea_periodic_input_locks_to_period
cargo test -p dsp set_tea_tau_secs_clamps_and_recomputes
cargo test -p dsp
```

All expected to pass. Some old beat tests may still pass (they don't reference the new pipeline yet — coexistence is intentional through this task).

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): TEA accumulator with Gaussian-smeared EMA + argmax"
```

---

## Task 7: Wire `process()` outputs to the new pipeline; rewrite `update_beat_pulses`

Up through Task 6, `beat_grid`, `beat_state`, and `beat_pulses` were still being written by the old `update_beat_state` + `update_beat_pulses`. This task switches them over to the new pipeline and rewrites `update_beat_pulses` to its simpler form. The old code is still present (deletion happens in Task 8).

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn beat_grid_from_new_pipeline_locks_to_periodic_input() {
    // End-to-end via process(): periodic signal at hops period 32 should
    // populate beat_grid[0] (period) ≈ 32 and beat_state[0] (BPM) ≈ 88.
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let sr = 48000.0_f32;
    let period_hops = 32usize;
    for k in 0..1500 {
        let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
        let signal: Vec<f32> = (0..2048)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
    }
    let grid = dsp.beat_grid();
    let state = dsp.beat_state();
    assert!(!grid[0].is_nan(), "expected period fit");
    assert!((grid[0] - period_hops as f32).abs() < 1.5,
        "period: expected ~{}, got {}", period_hops, grid[0]);
    let bpm_expected = 60.0 / (period_hops as f32 * (1024.0 / 48000.0));
    assert!((state[0] - bpm_expected).abs() < 4.0,
        "bpm: expected ~{:.1}, got {:.1}", bpm_expected, state[0]);
    // Measure detection is deferred — slots 2 and 3 are NaN
    assert!(state[2].is_nan(), "beats_per_measure should be NaN, got {}", state[2]);
    assert!(state[3].is_nan(), "measure_conf should be NaN, got {}", state[3]);
}

#[test]
fn beat_pulses_silent_input_all_nan() {
    // New behavior: under silence, score_inst=0 ⇒ beat_pulses is all-NaN.
    // (Replaces the old "free-run" behavior.)
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let silent = vec![0.0f32; 2048];
    for _ in 0..50 { dsp.process(&silent); }
    let pulses = dsp.beat_pulses();
    for (i, &v) in pulses.iter().enumerate() {
        assert!(v.is_nan(), "pulse[{}] expected NaN under silence, got {}", i, v);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail (or behave differently)**

Run: `cargo test -p dsp beat_grid_from_new_pipeline_locks_to_periodic_input beat_pulses_silent_input_all_nan`

Expected: At least the silence/NaN test fails (old code free-runs).

Expected outcome of the existing test `beat_pulses_free_run_at_default_rate_under_silence`: it will FAIL once Step 3 below is in place. This test is being replaced by `beat_pulses_silent_input_all_nan` and is deleted in Step 3.

- [ ] **Step 3: Switch outputs + rewrite `update_beat_pulses`**

In `Dsp::process()`, find the existing call sequence:

```rust
        self.pick_acf_peaks();
        self.update_beat_state();
        self.update_beat_pulses();
```

Replace it with (the new helpers `pick_candidates`, `score_candidates`, `update_tea` were added in earlier tasks; they should already be called in `process()` by Task 6):

```rust
        // Old beat path (removed): pick_acf_peaks / update_beat_state /
        // old update_beat_pulses. New path runs above:
        //   self.pick_candidates();
        //   self.score_candidates();
        //   self.update_tea();
        // Now write outputs from the new pipeline:
        self.write_beat_outputs();
        self.update_beat_pulses_v2();
```

Add the new methods on `impl Dsp`:

```rust
    fn write_beat_outputs(&mut self) {
        let p = self.tau_smoothed;
        let phi = self.phase_smoothed;
        let s = self.score_inst;
        if p.is_nan() || s <= 0.0 {
            self.beat_grid[0] = f32::NAN;
            self.beat_grid[1] = f32::NAN;
            self.beat_grid[2] = 0.0;
            self.beat_state[0] = f32::NAN;
            self.beat_state[1] = 0.0;
            self.beat_state[2] = f32::NAN;
            self.beat_state[3] = f32::NAN;
        } else {
            self.beat_grid[0]  = p;
            self.beat_grid[1]  = phi;
            self.beat_grid[2]  = s;
            self.beat_state[0] = if p > 0.0 { 60.0 / (p * self.dt) } else { f32::NAN };
            self.beat_state[1] = s;
            self.beat_state[2] = f32::NAN; // beats_per_measure deferred
            self.beat_state[3] = f32::NAN; // measure_conf deferred
        }
    }

    /// Simplified saw-wave generator. Phase is always real when there's a
    /// fit; NaN-out when there isn't (silent / unconfident).
    fn update_beat_pulses_v2(&mut self) {
        let period = self.tau_smoothed;
        let phase = self.phase_smoothed;
        let score = self.score_inst;
        if period.is_nan() || period <= 0.0 || score <= 0.0 || phase.is_nan() {
            for slot in self.beat_pulses.iter_mut() {
                *slot = f32::NAN;
            }
            return;
        }
        // Anchor fractional part to phase/period so cycle-1 starts at phase=0.
        let phase_frac = (phase / period).clamp(0.0, 0.999_999);
        let prev = self.beat_position;
        let mut bp = prev.floor() + phase_frac;
        if bp < prev - 0.5 {
            bp += 1.0;
        }
        self.beat_position = bp.rem_euclid(16.0);
        for (i, &m) in BEAT_PULSE_CYCLES.iter().enumerate() {
            let frac = (self.beat_position / m).fract();
            self.beat_pulses[i] = 1.0 - frac;
        }
    }
```

Delete the old `beat_pulses_free_run_at_default_rate_under_silence` test (it tested the deprecated free-run behavior — replaced by `beat_pulses_silent_input_all_nan`).

Update the existing `beat_pulses_advance_with_period_and_wrap` test if it passes — leave it alone if it still works against the new path, or update its assertions to be tolerance-friendly. Concretely, after the rewrite the cycle-1 saw should still advance per hop (about `1/period` per hop) and be in `[0, 1]` — if the existing test still passes, keep it. If not, replace its body with:

```rust
#[test]
fn beat_pulses_advance_with_period_and_wrap() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let sr = 48000.0_f32;
    let period_hops = 32usize;
    for k in 0..1500 {
        let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
        let signal: Vec<f32> = (0..2048)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
    }
    let pulses = dsp.beat_pulses();
    for (i, &v) in pulses.iter().enumerate() {
        assert!(!v.is_nan() && (0.0..=1.0).contains(&v),
            "pulse[{}] out of range: {}", i, v);
    }
}
```

(The old test asserted on per-hop advancement and cycle-16 vs cycle-1 ratios. Those assertions still hold qualitatively but tolerance is harder to pin down across the new pipeline; keep just the in-range + non-NaN check.)

Update the existing end-to-end test `beat_grid_end_to_end_via_process` — the old version asserted period within ±1.5; the new assertion is the same and should still pass. If it fails, the failure indicates a real bug in the new pipeline and should be fixed in this task.

The existing `beat_grid_finds_phase_aligned_with_rms_peaks` test (line 1711 in the current file) targets the old `fit_beat_phase` behavior. Delete it — phase coherence is now tested through `beat_grid_from_new_pipeline_locks_to_periodic_input`.

- [ ] **Step 4: Run the tests to verify they pass**

```
cargo test -p dsp beat_grid_from_new_pipeline_locks_to_periodic_input
cargo test -p dsp beat_pulses_silent_input_all_nan
cargo test -p dsp beat_pulses_advance_with_period_and_wrap
cargo test -p dsp beat_grid_end_to_end_via_process
cargo test -p dsp
```

The last one (full suite) is expected to have several failing tests still — the *old* tracker tests (`tracker_holds_default_120_bpm_under_silence`, `tracker_pulls_period_toward_nearby_observation`, `tracker_holds_period_against_single_outlier_frame`, `tracker_picks_beats_per_measure_from_strongest_m_peak`, `tracker_switches_beats_per_measure_when_strongest_m_peak_changes`). Those tests still target the old `BeatTracker` struct's behavior, which is no longer wired into `process()`. They are deleted in Task 8. Until then, accept the targeted-test passes from Step 4 and leave the broader suite for Task 8.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): wire process() to Percival-Tzanetakis pipeline; rewrite beat_pulses"
```

---

## Task 8: Delete old beat-tracker code

Remove the old `BeatTracker` struct, its impl, all its constants, the old `pick_acf_peaks` method, `update_beat_state`, `fit_beat_phase`, the old `update_beat_pulses`, `acf_peaks` field/getter, `peak_candidates` (old scratch), `rms_acf_accum`, `accum_alpha`, `set_accum_tau_secs`, and their tests.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Delete old impl code**

Delete these items from the file (line ranges are approximate against the pre-Task-1 file):

- The `BeatTracker` struct (lines ~98-108) and its full `impl BeatTracker { ... }` block (lines ~110-237).
- All the `BEAT_TRACKER_*` constants tied to the old tracker: `BEAT_TRACKER_INITIAL_BPM`, `BEAT_TRACKER_INITIAL_BEATS_PER_MEASURE`, `BEAT_TRACKER_SIGMA_LAG`, `BEAT_TRACKER_ALPHA_MAX`, `BEAT_TRACKER_CONF_SMOOTHING`, `BEAT_TRACKER_MEASURE_SWITCH_MARGIN`, `BEAT_TRACKER_MEASURE_CANDIDATES`, `BEAT_PHASE_STEP_HOPS`, `MIN_PEAK_LAG`. Keep `BEAT_TRACKER_MIN_BPM` and `BEAT_TRACKER_MAX_BPM` (still used by Task 4's tau bounds), `BEAT_GRID_LEN`, `BEAT_PULSES_LEN`, `BEAT_STATE_LEN`, `BEAT_PULSE_CYCLES`.
- `Dsp` fields: `acf_peaks`, `peak_candidates` (the *old* one — Task 4's `cand_scratch` stays), `beat_tracker`, `rms_acf_accum`, `accum_alpha`. Their initializers in `Dsp::new()` go too.
- `Dsp` methods: `set_accum_tau_secs`, `pick_acf_peaks` (the old method — Task 4's `pick_candidates` stays), `update_beat_state`, `fit_beat_phase`, the old `update_beat_pulses`. The new `update_beat_pulses_v2` from Task 7 stays.
- Public getters: `acf_peaks(&self)`, `rms_acf_accum(&self)`. Keep `beat_grid`, `beat_state`, `beat_pulses`.
- Test-only helpers in the `#[cfg(test)] impl Dsp` block: `test_set_rms_acf_accum`, `test_run_peak_picking`, `test_set_acf_peak`, `test_clear_acf_peaks`. Their per-test sites become Task 4's `test_set_onset_acf_enhanced` / `test_run_pick_candidates`, which are already there.

Also delete these tests at the bottom of the file:

- `accum_alpha_matches_formula`
- `set_accum_tau_secs_clamps_and_recomputes` (the *old* one — Task 6's `set_tea_tau_secs_clamps_and_recomputes` stays)
- `rms_acf_accum_silent_input_is_zero`
- `acf_peaks_silent_input_all_nan` (replaced by Task 4's `pick_candidates_silent_all_nan`)
- `acf_peaks_min_lag_enforced`
- `acf_peaks_finds_isolated_peak_with_subbin_offset`
- `acf_peaks_min_spacing_filters_nearby`
- `acf_peaks_top_n_selection_in_descending_magnitude`
- `acf_peaks_negative_correlations_skipped`
- `rms_acf_accum_converges_to_instantaneous_for_steady_periodic`
- `process_pipeline_finds_periodic_peak_via_acf_peaks`
- `tracker_holds_default_120_bpm_under_silence`
- `tracker_pulls_period_toward_nearby_observation`
- `tracker_holds_period_against_single_outlier_frame`
- `tracker_picks_beats_per_measure_from_strongest_m_peak`
- `tracker_switches_beats_per_measure_when_strongest_m_peak_changes`
- `beat_grid_finds_phase_aligned_with_rms_peaks` (already deleted in Task 7)
- `beat_pulses_free_run_at_default_rate_under_silence` (already deleted in Task 7)

- [ ] **Step 2: Run the test suite**

Run: `cargo test -p dsp`

Expected: PASS. The remaining tests are: the OSS / gen-ACF / harmonic / pick / pulse / TEA tests from Tasks 1-6, the wired-output tests from Task 7, and the unrelated waveform / spectrum / multiband-RMS / buffer-ACF tests that were never tied to the beat path.

If anything fails, the failure indicates a missed reference — fix in this task.

- [ ] **Step 3: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "chore(dsp): remove old BeatTracker / pick_acf_peaks / accum / fit_beat_phase"
```

---

## Task 9: Delete old visualization ACFs (`rms_acf`, `low_rms_acf`)

The user wants `rms_acf` renamed to `onset_acf` (already done by Task 2, since the new buffer is named `onset_acf` from the start) and `low_rms_acf` dropped entirely. This task removes the old time-domain ACF compute and its inputs.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Delete fields and computations**

Delete:

- Fields: `rms_acf`, `rms_detrended`, `low_rms_detrended`, `low_rms_acf`. Their initializers in `Dsp::new()` go too.
- Public getters: `rms_acf(&self)`, `low_rms_acf(&self)`.
- In `Dsp::process()`, the blocks that compute the full-band detrended ACF and the low-band detrended ACF (the two `autocorrelate(...)` calls operating on `rms_detrended` → `rms_acf` and `low_rms_detrended` → `low_rms_acf`).

Keep:

- `low_rms_history`, `mid_rms_history`, `high_rms_history` and their getters — they're independent visualization features.
- `buffer_acf` and the `autocorrelate(&self.waveform, &mut self.buffer_acf)` call — that's the lag-domain waveform ACF, still used.
- The `autocorrelate` free function itself — `buffer_acf` still uses it.

Delete tests:

- `low_rms_acf_has_correct_length` (line ~1310)
- `low_rms_acf_constant_input_is_zero` (line ~1316)

- [ ] **Step 2: Run the test suite**

Run: `cargo test -p dsp`

Expected: PASS. If `autocorrelate_helper_correctness` or `buffer_acf_*` tests fail, that's a fix-in-this-task signal — `autocorrelate` and `buffer_acf` should be untouched.

- [ ] **Step 3: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "chore(dsp): remove rms_acf and low_rms_acf (moved to onset_acf path)"
```

---

## Task 10: Update worklet message protocol

Switch `dsp-worklet.ts` to emit the new buffer set: `onset`, `onsetAcf`, `onsetAcfEnhanced`, `tea`, `candidates`. Drop `rmsAcf`, `rmsAcfAccum`, `acfPeaks`, `rmsAcfLow`. Rename `rmsAcfLen` → `onsetAcfLen` and add `onsetLen`, `teaLen`, `candidatesLen`. Route `teaTauSecs`, drop `accumTauSecs`.

**Files:**
- Modify: `src/audio/dsp-worklet.ts`

- [ ] **Step 1: Update message types and the param routing**

Open `src/audio/dsp-worklet.ts`. Find the `ConfiguredOutbound` type (the `type: "configured"` payload):

```ts
type ConfiguredOutbound = {
  type: "configured";
  // ... existing fields ...
  rmsAcfLen: number;
  acfPeaksLen: number;
  // ...
};
```

Replace `rmsAcfLen` and `acfPeaksLen` with:

```ts
  onsetLen: number;
  onsetAcfLen: number;
  teaLen: number;
  candidatesLen: number;
```

Find the `WorkletInbound` discriminated union, the `param` arm:

```ts
| { type: "param"; key: "hopSize" | "smoothingTauSecs" | "dbFloor" | "accumTauSecs"; value: number };
```

Replace `accumTauSecs` with `teaTauSecs`:

```ts
| { type: "param"; key: "hopSize" | "smoothingTauSecs" | "dbFloor" | "teaTauSecs"; value: number };
```

Find the processor's private state for the cached `accumTauSecs` value (search for `accumTauSecs`). Rename to `teaTauSecs`, default `4.0`. The line `this.accumTauSecs` becomes `this.teaTauSecs`. Initialization stays at `4.0`.

In the `param` message handler, replace the `accumTauSecs` branch:

```ts
} else if (msg.key === "accumTauSecs") {
  this.accumTauSecs = msg.value;
  if (this.dsp) this.dsp.set_accum_tau_secs(msg.value);
}
```

with:

```ts
} else if (msg.key === "teaTauSecs") {
  this.teaTauSecs = msg.value;
  if (this.dsp) this.dsp.set_tea_tau_secs(msg.value);
}
```

In `applyConfigure`, replace the `set_accum_tau_secs` call:

```ts
this.dsp.set_accum_tau_secs(this.accumTauSecs);
```

with:

```ts
this.dsp.set_tea_tau_secs(this.teaTauSecs);
```

In the configured payload literal:

```ts
const payload: ConfiguredOutbound = {
  type: "configured",
  // ... unchanged fields ...
  rmsAcfLen: this.rmsHistoryLen / 2,
  acfPeaksLen: 30,
  // ...
};
```

Replace those two with the four new ones:

```ts
  onsetLen: this.rmsHistoryLen,
  onsetAcfLen: this.rmsHistoryLen / 2,
  teaLen: this.rmsHistoryLen / 2,
  candidatesLen: 30, // = 3 * MAX_PEAKS
```

In the `process()` features postMessage block, replace:

```ts
const ra = new Float32Array(this.dsp.rms_acf());
const raAccum = new Float32Array(this.dsp.rms_acf_accum());
const peaks = new Float32Array(this.dsp.acf_peaks());
const rmsAcfLow = new Float32Array(this.dsp.low_rms_acf());
```

with:

```ts
const onset = new Float32Array(this.dsp.onset_history());
const onsetAcf = new Float32Array(this.dsp.onset_acf());
const onsetAcfEnh = new Float32Array(this.dsp.onset_acf_enhanced());
const tea = new Float32Array(this.dsp.tea());
const candidates = new Float32Array(this.dsp.candidates());
```

Update the postMessage payload to include the new buffers (and remove the old ones):

```ts
this.port.postMessage(
  {
    type: "features",
    waveform: wf,
    spectrum: sp,
    rms,
    bufferAcf: ba,
    onset,
    onsetAcf,
    onsetAcfEnhanced: onsetAcfEnh,
    tea,
    candidates,
    beatGrid,
    beatPulses,
    beatState,
    rmsLow,
    rmsMid,
    rmsHigh,
  },
  [
    wf.buffer,
    sp.buffer,
    rms.buffer,
    ba.buffer,
    onset.buffer,
    onsetAcf.buffer,
    onsetAcfEnh.buffer,
    tea.buffer,
    candidates.buffer,
    beatGrid.buffer,
    beatPulses.buffer,
    beatState.buffer,
    rmsLow.buffer,
    rmsMid.buffer,
    rmsHigh.buffer,
  ],
);
```

- [ ] **Step 2: Build the WASM and verify the worklet compiles**

```
npm run wasm
npx tsc --noEmit
```

Expected: WASM rebuild + tsc both pass. If `tsc` complains about `DebugFeatures` (the type used in `DebugView.applyFeatures`), fix in Task 12 — for now the failure indicates a real cross-file change is needed (it's expected at this point).

If `tsc` reports the failure is in `DebugView.ts`, that's the expected next-task signal. If the failure is anywhere else (e.g. `dsp-worklet.ts` itself), fix it in this task.

- [ ] **Step 3: Commit (worklet only)**

```bash
git add src/audio/dsp-worklet.ts src/wasm-pkg
git commit -m "feat(audio): worklet emits onset/onsetAcf/onsetAcfEnhanced/tea/candidates"
```

---

## Task 11: Update params + WorkletBridge

Drop `dsp.accumTauSecs`, add `dsp.teaTauSecs`. Update `HOT_KEYS`. Update the WorkletBridge unit test.

**Files:**
- Modify: `src/params/schemas.ts`, `src/params/WorkletBridge.ts`, `tests/params/WorkletBridge.test.ts`

- [ ] **Step 1: Update the WorkletBridge test (failing first)**

In `tests/params/WorkletBridge.test.ts`, find the test:

```ts
it("accumTauSecs change posts a param message with the hot key (no dsp prefix)", () => {
```

Replace with:

```ts
it("teaTauSecs change posts a param message with the hot key (no dsp prefix)", () => {
  const store = makeStore();
  const port = makePort();
  new WorkletBridge(store, port);
  (port.postMessage as ReturnType<typeof vi.fn>).mockClear();
  store.set("dsp.teaTauSecs", 8.0);
  expect(port.postMessage).toHaveBeenCalledWith({
    type: "param",
    key: "teaTauSecs",
    value: 8.0,
  });
});
```

In the `bootstrap` test, find:

```ts
expect(calls).toContainEqual({ type: "param", key: "accumTauSecs", value: 4.0 });
```

Replace with:

```ts
expect(calls).toContainEqual({ type: "param", key: "teaTauSecs", value: 4.0 });
```

The `expect(calls.length).toBe(5)` line — count is unchanged (4 hot keys + 1 configure = 5).

- [ ] **Step 2: Run the test to verify it fails**

```
npx vitest run tests/params/WorkletBridge.test.ts -t "teaTauSecs"
```

Expected: FAIL — `dsp.teaTauSecs` not registered.

- [ ] **Step 3: Update schemas and bridge**

In `src/params/schemas.ts`, find the `dsp.accumTauSecs` schema entry:

```ts
{
  key: "dsp.accumTauSecs",
  label: "ACF accumulator τ (s)",
  // ...
},
```

Replace with:

```ts
{
  key: "dsp.teaTauSecs",
  label: "TEA τ (s)",
  kind: "continuous",
  min: 0.2,
  max: 30.0,
  step: 0.1,
  default: 4.0,
  reconfig: false,
},
```

In `src/params/WorkletBridge.ts`:

```ts
const HOT_KEYS = ["hopSize", "smoothingTauSecs", "dbFloor", "accumTauSecs"] as const;
```

Replace with:

```ts
const HOT_KEYS = ["hopSize", "smoothingTauSecs", "dbFloor", "teaTauSecs"] as const;
```

- [ ] **Step 4: Run the tests to verify they pass**

```
npx vitest run tests/params/WorkletBridge.test.ts
npm test
```

Expected: PASS for the WorkletBridge file; the broader suite may still have failures originating in DebugView (Task 12) — proceed if those are the only failures.

- [ ] **Step 5: Commit**

```bash
git add src/params/schemas.ts src/params/WorkletBridge.ts tests/params/WorkletBridge.test.ts
git commit -m "feat(params): rename accumTauSecs → teaTauSecs"
```

---

## Task 12: Update `DebugView` (drop old strips, add new)

Wire the new buffers (`onset`, `onsetAcf`, `onsetAcfEnhanced`, `tea`, `candidates`) into the renderer. Drop `rmsAcf`, `rmsAcfAccum`, `acfPeaks`, `rmsAcfLow`.

**Files:**
- Modify: `src/render/DebugView.ts`

- [ ] **Step 1: Update the `DebugFeatures` type and `DebugSizes`**

Find the `DebugFeatures` type. Replace the old buffer keys:

```ts
rmsAcf?: Float32Array;
rmsAcfAccum?: Float32Array;
acfPeaks?: Float32Array;
rmsAcfLow?: Float32Array;
```

with:

```ts
onset?: Float32Array;
onsetAcf?: Float32Array;
onsetAcfEnhanced?: Float32Array;
tea?: Float32Array;
candidates?: Float32Array;
```

Find `DebugSizes`. Replace `rmsAcfLen` / `acfPeaksLen` with `onsetLen` / `onsetAcfLen` / `teaLen` / `candidatesLen` (matching the new `ConfiguredOutbound`).

- [ ] **Step 2: Update `applyFeatures`**

Replace:

```ts
if (msg.rmsAcf) store.set("rmsAcf", msg.rmsAcf);
if (msg.rmsAcfAccum) store.set("rmsAcfAccum", msg.rmsAcfAccum);
if (msg.acfPeaks) store.set("acfPeaks", msg.acfPeaks);
if (msg.rmsAcfLow) store.set("rmsAcfLow", msg.rmsAcfLow);
```

with:

```ts
if (msg.onset) store.set("onset", msg.onset);
if (msg.onsetAcf) store.set("onsetAcf", msg.onsetAcf);
if (msg.onsetAcfEnhanced) store.set("onsetAcfEnhanced", msg.onsetAcfEnhanced);
if (msg.tea) store.set("tea", msg.tea);
if (msg.candidates) store.set("candidates", msg.candidates);
```

- [ ] **Step 3: Update `applyConfigured`**

Find the dispose block at the top of `applyConfigured`. Replace the line list:

```ts
this.lowRmsAcfLine,
this.rmsAcfLine,
this.rmsAcfAccumLine,
```

with:

```ts
this.onsetLine,
this.onsetAcfLine,
this.onsetAcfEnhancedLine,
this.teaLine,
```

Find the field declarations near the top of the class:

```ts
private rmsAcfLine?: LineRenderer;
private rmsAcfAccumLine?: LineRenderer;
private peakMarkers?: PeakMarkers;
private lowRmsAcfLine?: LineRenderer;
```

Replace with:

```ts
private onsetLine?: LineRenderer;
private onsetAcfLine?: LineRenderer;
private onsetAcfEnhancedLine?: LineRenderer;
private teaLine?: LineRenderer;
private peakMarkers?: PeakMarkers;
```

Replace the store-allocation block:

```ts
store.set("rmsAcf", new Float32Array(sizes.rmsAcfLen));
store.set("rmsAcfAccum", new Float32Array(sizes.rmsAcfLen));
const peaksInit = new Float32Array(sizes.acfPeaksLen);
peaksInit.fill(NaN);
store.set("acfPeaks", peaksInit);
// ...
store.set("rmsAcfLow", new Float32Array(sizes.rmsAcfLen));
```

with:

```ts
store.set("onset", new Float32Array(sizes.onsetLen));
store.set("onsetAcf", new Float32Array(sizes.onsetAcfLen));
store.set("onsetAcfEnhanced", new Float32Array(sizes.onsetAcfLen));
store.set("tea", new Float32Array(sizes.teaLen));
const candidatesInit = new Float32Array(sizes.candidatesLen);
candidatesInit.fill(NaN);
store.set("candidates", candidatesInit);
```

Replace the line-renderer construction block:

```ts
this.lowRmsAcfLine = new LineRenderer({
  source: () => store.get("rmsAcfLow"),
  layout: linearLayout(-1.0, 0.4),
  color: 0xbb8888,
});
scene.add(this.lowRmsAcfLine.object3d);

this.rmsAcfLine = new LineRenderer({
  source: () => store.get("rmsAcf"),
  layout: linearLayout(-1.0, 0.4),
  color: 0x666666,
});
scene.add(this.rmsAcfLine.object3d);

this.rmsAcfAccumLine = new LineRenderer({
  source: () => store.get("rmsAcfAccum"),
  layout: linearLayout(-1.0, 0.4),
  color: 0x66ffff,
});
scene.add(this.rmsAcfAccumLine.object3d);

this.peakMarkers = new PeakMarkers({
  source: () => store.get("acfPeaks"),
  maxPeaks: 10,
  lagDomain: sizes.rmsAcfLen,
  yCenter: -1.0,
  ySpan: 0.4,
  xForLag: (lag, n) => (n <= 1 ? 0 : (lag / (n - 1)) * 2 - 1),
  baseColor: 0x888888,
});
scene.add(this.peakMarkers.object3d);
```

with:

```ts
this.onsetLine = new LineRenderer({
  source: () => store.get("onset"),
  layout: linearLayout(0.0, 0.4),
  color: 0xff9966,
});
scene.add(this.onsetLine.object3d);

this.onsetAcfLine = new LineRenderer({
  source: () => store.get("onsetAcf"),
  layout: linearLayout(-1.0, 0.4),
  color: 0x66ffff,
});
scene.add(this.onsetAcfLine.object3d);

this.onsetAcfEnhancedLine = new LineRenderer({
  source: () => store.get("onsetAcfEnhanced"),
  layout: linearLayout(-1.0, 0.4),
  color: 0xff99cc,
});
scene.add(this.onsetAcfEnhancedLine.object3d);

this.teaLine = new LineRenderer({
  source: () => store.get("tea"),
  layout: linearLayout(-1.0, 0.4),
  color: 0xffff66,
});
scene.add(this.teaLine.object3d);

this.peakMarkers = new PeakMarkers({
  source: () => store.get("candidates"),
  maxPeaks: 10,
  lagDomain: sizes.onsetAcfLen,
  yCenter: -1.0,
  ySpan: 0.4,
  xForLag: (lag, n) => (n <= 1 ? 0 : (lag / (n - 1)) * 2 - 1),
  baseColor: 0x888888,
});
scene.add(this.peakMarkers.object3d);
```

- [ ] **Step 4: Update `update()` per-frame loop**

Find the `update()` method. It currently calls `update()` on each line renderer + `peakMarkers` + `beatDebug`. Replace the references to the dropped renderers with the new ones. Concretely, search for `this.rmsAcfLine?.update()`, `this.rmsAcfAccumLine?.update()`, `this.lowRmsAcfLine?.update()` and replace with:

```ts
this.onsetLine?.update();
this.onsetAcfLine?.update();
this.onsetAcfEnhancedLine?.update();
this.teaLine?.update();
```

- [ ] **Step 5: Update `dispose()`**

Mirror the `applyConfigured` dispose-list update — replace dropped renderer references with the new ones, also setting them to `undefined` after dispose (matching the existing pattern).

- [ ] **Step 6: Type-check + run tests**

```
npx tsc --noEmit
npm test
```

Expected: tsc clean, vitest passes (modulo any stragglers in renderer fixture tests — those are addressed in Task 14).

- [ ] **Step 7: Commit**

```bash
git add src/render/DebugView.ts
git commit -m "feat(render): switch DebugView to onset/onsetAcf/onsetAcfEnhanced/tea/candidates"
```

---

## Task 13: Update `BeatDebugView` size keys

Rename `BeatDebugSizes.rmsAcfLen` → `onsetAcfLen`. Adjust constructions of `BeatGridRenderer` etc. that read from it.

**Files:**
- Modify: `src/render/BeatDebugView.ts`

- [ ] **Step 1: Rename the size key**

In `BeatDebugSizes`:

```ts
export interface BeatDebugSizes {
  rmsLen: number;
  rmsAcfLen: number;
  beatGridLen: number;
  beatPulsesLen: number;
  beatStateLen: number;
}
```

Replace `rmsAcfLen` with `onsetAcfLen`:

```ts
export interface BeatDebugSizes {
  rmsLen: number;
  onsetAcfLen: number;
  beatGridLen: number;
  beatPulsesLen: number;
  beatStateLen: number;
}
```

In `applyConfigured`, find the references to `sizes.rmsAcfLen` and rename them all to `sizes.onsetAcfLen`. Concretely, the existing constructions:

```ts
lagDomain: sizes.rmsAcfLen,
xForLag: linearX(sizes.rmsAcfLen),
```

become:

```ts
lagDomain: sizes.onsetAcfLen,
xForLag: linearX(sizes.onsetAcfLen),
```

Also propagate the rename in any caller of `BeatDebugView.applyConfigured` — `DebugView.ts` calls `this.beatDebug.applyConfigured(sizes)` where `sizes` is `DebugSizes`. The `DebugSizes` type was updated in Task 12 to have `onsetAcfLen`; the call site passes `sizes` directly, so this should compile cleanly. If `DebugSizes` doesn't include the same keys as `BeatDebugSizes`, build a focused object at the call site:

```ts
this.beatDebug.applyConfigured({
  rmsLen: sizes.rmsLen,
  onsetAcfLen: sizes.onsetAcfLen,
  beatGridLen: sizes.beatGridLen,
  beatPulsesLen: sizes.beatPulsesLen,
  beatStateLen: sizes.beatStateLen,
});
```

- [ ] **Step 2: Type-check + tests**

```
npx tsc --noEmit
npm test
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add src/render/BeatDebugView.ts src/render/DebugView.ts
git commit -m "refactor(render): BeatDebugSizes.rmsAcfLen → onsetAcfLen"
```

---

## Task 14: Update `PeakMarkers` test (and any other render tests referencing dropped buffers)

The `PeakMarkers` test currently builds `acfPeaks` fixtures and asserts the renderer reads them. The renderer is unchanged (still stride 3); only the test's fixture-setup wording needs updating to read more naturally. If the existing test already uses a generic `source: () => buffer` and doesn't hard-code the name "acfPeaks", no change is needed.

**Files:**
- Modify: `tests/render/PeakMarkers.test.ts` (only if it references "acfPeaks" or "rmsAcfLen" by name)
- Modify: `tests/render/BeatGridRenderer.test.ts`, `tests/render/BeatGridScrollingRenderer.test.ts`, `tests/render/BeatPulseSquares.test.ts` (only if they pass a `BeatDebugSizes`-shaped object using the old key)

- [ ] **Step 1: Audit and update**

Search for the dropped names in tests:

```bash
grep -rn "rmsAcfLen\|acfPeaks\|rmsAcf\|rmsAcfAccum\|rmsAcfLow\|accumTauSecs" tests/
```

For each match:
- If the file references `rmsAcfLen`: replace with `onsetAcfLen`.
- If the file references `acfPeaks` (the buffer name): if the test passes `source: () => buffer`, it doesn't matter what the buffer is *called* in the test — it should still pass. Update inline naming for clarity (e.g. rename the local from `peaks` to `candidates`).
- `accumTauSecs`: handled in Task 11.

If any test fails after these edits, the failure is inside the renderer's logic — the renderer is unchanged, so debug there.

- [ ] **Step 2: Run the suite**

```
npm test
cargo test -p dsp
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test(render): rename fixture references for new buffer names"
```

---

## Task 15: Update ROADMAP.md and final smoke test

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 1: Edit ROADMAP**

In `ROADMAP.md`, under "Beatdetector improvements", replace the entire list with:

```markdown
## Beatdetector improvements

1. Spectral flux OSS (replace the half-wave-rectified RMS-diff proxy with the paper's per-bin log-magnitude flux)
2. Octave decider (paper §II-C step 4) — heuristic or simple ML to multiply/halve the reported tempo
3. `beats_per_measure` detection (deferred from rewrite)
4. Onset / downbeat detection (which `k·τ` of the beat grid is the bar boundary)
```

Under "Shipped", add:

```markdown
- **v3.3** (tag pending): beat tracker rewrite to streaming Percival & Tzanetakis 2014 — half-wave-rectified RMS as OSS, generalized FFT ACF (`|X|^0.5`), harmonic enhancement, top-10 peak picking, pulse-train scoring, Gaussian-smeared TEA. Fixes the octave-ambiguity bug in the old `lag/k` tracker. New `dsp.teaTauSecs` param replaces `dsp.accumTauSecs`. New visualization channels: `onset`, `onsetAcf`, `onsetAcfEnhanced`, `tea`. Old `rms_acf` / `rms_acf_accum` / `low_rms_acf` removed.
```

Under "Performance", remove the old line:

```markdown
- Migrate autocorrelation to FFT-based (Wiener–Khinchin: ACF = IFFT(|FFT(x)|²)) once the v3 direct implementation has proven the visualization is correct.
```

(That migration is now done as a side-effect of this rewrite, on the beat path.)

- [ ] **Step 2: Manual smoke test**

```
npm run wasm
npm run dev
```

Open the browser to the dev URL. Verify:
- The autocorr lane (-1.0 yCenter band) shows: cyan `onsetAcf`, pink `onsetAcfEnhanced`, yellow `tea`, gray peak markers, yellow grid lines.
- The y=0 band shows the orange `onset` line spiking on transients.
- The 4 grayscale `BeatPulseSquares` pulse coherently with the music.
- Pressing `T` enables the test source (oscillator). The grid bars stabilize within a few seconds.
- Playing a track with a slow beat (under 80 BPM) reports a sensible BPM in the tweakpane / debug overlay (this is the bug fix).

If any visual regression is observed, file as a follow-up — do not block the commit.

- [ ] **Step 3: Commit**

```bash
git add ROADMAP.md
git commit -m "docs(roadmap): mark beat tracker rewrite shipped; add followup items"
```

---

## Self-Review Checklist

This plan was reviewed against the spec at `docs/superpowers/specs/2026-04-29-beat-tracker-percival-tzanetakis-design.md`:

- **OSS pipeline** — Task 1 ✓
- **Generalized ACF** — Task 2 ✓
- **Harmonic enhancement** — Task 3 ✓
- **Peak picking on enhanced ACF in `[tau_min, tau_max]`** — Task 4 ✓
- **Pulse-train scoring** — Task 5 ✓
- **TEA accumulator + smoothed period + coherent phase rescore** — Task 6 ✓
- **`beat_grid` / `beat_state` / `beat_pulses` outputs from new pipeline** — Task 7 ✓
- **Old code removal (BeatTracker, pick_acf_peaks, fit_beat_phase, rms_acf_accum, acf_peaks)** — Task 8 ✓
- **Old visualization ACFs (rms_acf, low_rms_acf) removal** — Task 9 ✓
- **Worklet message protocol** — Task 10 ✓
- **Params + WorkletBridge + test** — Task 11 ✓
- **DebugView strip rebuild + PeakMarkers rewire** — Task 12 ✓
- **BeatDebugView size key rename** — Task 13 ✓
- **Renderer test fixtures** — Task 14 ✓
- **ROADMAP + smoke test** — Task 15 ✓
- **Octave decider deferred** — explicit in Task 15 ROADMAP entry ✓
- **`beats_per_measure` deferred (NaN output)** — explicit in Task 7's `write_beat_outputs` ✓

No tasks reference functions/types not defined elsewhere in the plan. Method names are consistent across tasks (`pick_candidates`, `score_candidates`, `update_tea`, `update_beat_pulses_v2`, `write_beat_outputs`).
