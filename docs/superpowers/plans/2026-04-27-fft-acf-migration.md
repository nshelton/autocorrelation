# FFT-Based Autocorrelation Migration Plan (Wiener–Khinchin)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠️ PREREQUISITE — v3 must ship first.** This plan migrates an existing direct (O(N²)) autocorrelation implementation to an FFT-based (O(N log N)) one, using the direct version as a numerical correctness oracle during the swap. v3 (`In progress` per `ROADMAP.md`: "Buffer autocorrelation on the audio waveform" + "Beat detection via RMS-envelope autocorrelation") provides that direct implementation; this plan is **not executable** until v3 has shipped.
>
> **Assumed v3 API** (verified in Task 1 — update this plan if v3 chose differently):
> - `Dsp::autocorrelation_waveform() -> Vec<f32>` — ACF of the most recent input window (length ≈ window_size)
> - `Dsp::autocorrelation_rms() -> Vec<f32>` — ACF of `rms_history` (length ≈ RMS_HISTORY_LEN)
> - Both produce *biased* linear ACF (sum-of-products divided by N, NOT by N − lag).
>
> If any of these differ in v3, edit Tasks 2–7 in this plan to match before executing.

**Goal:** Replace direct O(N²) autocorrelation with FFT-based O(N log N) using the Wiener–Khinchin identity (ACF = IFFT(|FFT(x)|²)). Apply the same code path to both the waveform ACF and the RMS-envelope ACF used by beat detection. Output must be numerically equivalent to the direct version (within float tolerance).

**Architecture:** Add a single inverse-real-FFT plan to `Dsp` (shared between waveform and RMS-envelope ACF — different lengths handled by separate plans, since `realfft` plans are length-specific; we'll fold both onto the same `acf_fft` helper). The helper zero-pads input of length `N` to length `2N` (mandatory for *linear* ACF — without padding the FFT yields *circular* ACF), forward-FFTs, squares the magnitudes (with imaginary parts zeroed), inverse-FFTs, divides by `2N` (realfft's inverse is unnormalized), and writes the first `N` lags to the output. Existing public ACF methods become thin wrappers around the helper. Direct implementation is deleted last, only after equivalence is proven.

**Tech Stack:** Existing — `realfft` 3.x (already a dependency; pulls in `ComplexToReal` for the inverse plan), Rust + wasm-bindgen.

---

## Task 1: Verify v3 prerequisite + capture the direct API

Read the current state of `crates/dsp/src/lib.rs`. Confirm v3 is shipped with the assumed API; otherwise stop and reconcile this plan with reality.

**Files:** none modified (read only)

- [ ] **Step 1: Confirm a direct autocorrelation API exists**

Run: `grep -n "fn autocorrelation\|fn acf\|fn beat" crates/dsp/src/lib.rs`

Expected: at least two `pub fn` matches — one for the waveform ACF, one for the RMS-envelope ACF. If the grep returns no matches, **STOP** — v3 has not shipped and this plan cannot execute. Tell the user.

- [ ] **Step 2: Capture exact method names, signatures, and ACF output lengths**

Open `crates/dsp/src/lib.rs`. Note:

- The exact method names (this plan assumes `autocorrelation_waveform` and `autocorrelation_rms` — adjust below if different).
- The output length each method produces (this plan assumes `window_size` for waveform ACF and `RMS_HISTORY_LEN` for RMS ACF).
- The normalization (this plan assumes biased: each lag-`k` sum divided by `N`, not `N - k`).
- The line range of the direct implementation — it gets deleted in Task 7.

**If any of these differ from this plan's assumptions, edit Tasks 2–7 to match before continuing.** Specifically: every reference to `autocorrelation_waveform` / `autocorrelation_rms`, every length assumption, and every comparison tolerance in the equivalence test.

- [ ] **Step 3: Capture a baseline reference output**

Run: `cargo test -p dsp` — confirm all v3 ACF tests pass before any migration begins. The migration must leave them passing.

Expected: green; note the count.

---

## Task 2: Add inverse-FFT plumbing to `Dsp`

Add a forward+inverse real-FFT plan pair sized for the waveform ACF (`2 * window_size`) and a separate pair for the RMS-envelope ACF (`2 * RMS_HISTORY_LEN`), plus reusable scratch buffers. Pure plumbing — no behavior change yet, no public API change.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Update imports**

Find:

```rust
use realfft::{RealFftPlanner, RealToComplex};
```

Replace with:

```rust
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
```

- [ ] **Step 2: Add ACF fields to the `Dsp` struct**

Add the following fields to the end of the struct definition (alongside `rms_history` and friends):

```rust
    /// Forward FFT for waveform ACF, sized 2 * window_size for linear (zero-padded) ACF.
    acf_wf_fwd: Arc<dyn RealToComplex<f32>>,
    /// Inverse FFT for waveform ACF, sized 2 * window_size.
    acf_wf_inv: Arc<dyn ComplexToReal<f32>>,
    /// Scratch: zero-padded time-domain input (length 2 * window_size).
    acf_wf_time_in: Vec<f32>,
    /// Scratch: complex frequency buffer (length window_size + 1).
    acf_wf_freq: Vec<Complex<f32>>,
    /// Scratch: inverse-FFT output (length 2 * window_size). First `window_size`
    /// samples are the ACF lags 0..N-1 after dividing by 2 * window_size.
    acf_wf_time_out: Vec<f32>,

    /// Forward FFT for RMS-envelope ACF, sized 2 * RMS_HISTORY_LEN.
    acf_rms_fwd: Arc<dyn RealToComplex<f32>>,
    /// Inverse FFT for RMS-envelope ACF.
    acf_rms_inv: Arc<dyn ComplexToReal<f32>>,
    acf_rms_time_in: Vec<f32>,
    acf_rms_freq: Vec<Complex<f32>>,
    acf_rms_time_out: Vec<f32>,
```

- [ ] **Step 3: Initialize the ACF buffers in `Dsp::new`**

Inside the constructor body, after the existing `let mag_scale = ...` line and before the `Dsp { ... }` literal, add:

```rust
        // Wiener–Khinchin scratch space. 2N zero-padding is required to get
        // *linear* (not circular) ACF from FFT magnitude-squared.
        let acf_wf_n = window_size * 2;
        let mut acf_planner = RealFftPlanner::<f32>::new();
        let acf_wf_fwd = acf_planner.plan_fft_forward(acf_wf_n);
        let acf_wf_inv = acf_planner.plan_fft_inverse(acf_wf_n);
        let acf_wf_time_in = vec![0.0; acf_wf_n];
        let acf_wf_freq = acf_wf_fwd.make_output_vec();
        let acf_wf_time_out = vec![0.0; acf_wf_n];

        let acf_rms_n = RMS_HISTORY_LEN * 2;
        let acf_rms_fwd = acf_planner.plan_fft_forward(acf_rms_n);
        let acf_rms_inv = acf_planner.plan_fft_inverse(acf_rms_n);
        let acf_rms_time_in = vec![0.0; acf_rms_n];
        let acf_rms_freq = acf_rms_fwd.make_output_vec();
        let acf_rms_time_out = vec![0.0; acf_rms_n];
```

Then in the `Dsp { ... }` struct literal, add the new fields at the end (after the existing `smoothing_alpha` / `rms_history` etc.):

```rust
            acf_wf_fwd,
            acf_wf_inv,
            acf_wf_time_in,
            acf_wf_freq,
            acf_wf_time_out,
            acf_rms_fwd,
            acf_rms_inv,
            acf_rms_time_in,
            acf_rms_freq,
            acf_rms_time_out,
```

- [ ] **Step 4: Build & test (no behavior change yet)**

Run: `cargo test -p dsp`
Expected: same green count as Task 1 Step 3 — the new fields exist but no method uses them yet.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): add inverse-FFT plumbing for Wiener-Khinchin ACF (no behavior change)"
```

---

## Task 3: Implement the `acf_fft` private helper (TDD)

A single private method that, given an input slice and the four scratch buffers (forward plan, inverse plan, time-in, freq, time-out), writes the linear ACF lags into the first N entries of `time_out`. This is shared by both public ACF methods.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing test for `acf_fft` correctness on a known signal**

Add this test to the `mod tests` block:

```rust
    /// Naive O(N²) biased linear ACF for use as a numerical oracle.
    /// r[k] = (1/N) * sum_{i=0..N-1-k} x[i] * x[i+k]
    fn naive_biased_acf(x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mut r = vec![0.0_f32; n];
        for k in 0..n {
            let mut s = 0.0_f32;
            for i in 0..(n - k) {
                s += x[i] * x[i + k];
            }
            r[k] = s / n as f32;
        }
        r
    }

    #[test]
    fn acf_fft_matches_naive_on_short_signal() {
        // Window size 16 keeps both plans (32-point FFT) tiny + fast.
        let mut dsp = Dsp::new(16, 48000.0, 8);
        // Mixed-frequency signal — non-trivial ACF.
        let x: Vec<f32> = (0..16)
            .map(|i| {
                let t = i as f32;
                (0.5 * t).sin() + 0.3 * (1.7 * t + 0.4).cos()
            })
            .collect();
        let expected = naive_biased_acf(&x);
        let got = dsp.acf_fft_waveform(&x); // exposed for testing — see Step 3
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "lag {}: fft {} vs naive {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn acf_fft_lag_zero_equals_mean_square() {
        // ACF[0] = (1/N) * sum(x²) — definition of biased ACF at lag 0.
        let mut dsp = Dsp::new(16, 48000.0, 8);
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let got = dsp.acf_fft_waveform(&x);
        assert!(
            (got[0] - mean_sq).abs() < 1e-5,
            "lag 0 = {} vs mean_sq = {}",
            got[0],
            mean_sq
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p dsp`
Expected: compile error — `no method named 'acf_fft_waveform'`.

- [ ] **Step 3: Implement the helper**

In `crates/dsp/src/lib.rs`, inside the `#[wasm_bindgen] impl Dsp { ... }` block, add this method (place it near the other ACF-related methods if v3 grouped them, otherwise at the end before the closing brace):

```rust
    /// Wiener–Khinchin linear autocorrelation of `input` (length must equal
    /// the waveform window size). Returns biased ACF of length `window_size`,
    /// where r[0] = (1/N) * sum(x²) and r[k] = (1/N) * sum_{i} x[i] * x[i+k].
    ///
    /// Uses the cached 2N-point forward and inverse real-FFT plans. The 2N
    /// zero-padding is required: without it, FFT magnitude-squared yields
    /// *circular* ACF (wraps around) rather than the linear ACF the
    /// visualization expects.
    ///
    /// Public for testability (exercised by the equivalence test against
    /// the naive direct version). Not exported via wasm-bindgen — see the
    /// `#[wasm_bindgen]` skip pattern below.
    pub fn acf_fft_waveform(&mut self, input: &[f32]) -> Vec<f32> {
        let n = self.waveform.len();
        debug_assert_eq!(input.len(), n, "acf_fft_waveform expects len = window_size");
        let pad = self.acf_wf_time_in.len(); // 2N
        // Zero-pad input into the scratch buffer.
        self.acf_wf_time_in[..n].copy_from_slice(&input[..n]);
        for v in &mut self.acf_wf_time_in[n..pad] {
            *v = 0.0;
        }
        // Forward FFT.
        let _ = self
            .acf_wf_fwd
            .process(&mut self.acf_wf_time_in, &mut self.acf_wf_freq);
        // Replace each bin with |bin|² (real, imag = 0).
        for c in &mut self.acf_wf_freq {
            let mag_sq = c.re * c.re + c.im * c.im;
            c.re = mag_sq;
            c.im = 0.0;
        }
        // Inverse FFT. realfft's inverse is unnormalized — see scaling math below.
        let _ = self
            .acf_wf_inv
            .process(&mut self.acf_wf_freq, &mut self.acf_wf_time_out);
        // Scaling math: realfft_ifft = pad * numpy_ifft (realfft's inverse is
        // unnormalized — it omits the customary 1/M factor). For the
        // power-spectrum input, numpy_ifft(|X_padded|²)[i] is the
        // unnormalized linear-ACF sum: sum_{j=0..n-1-i} x[j] * x[j+i].
        // Biased ACF divides that sum by n. So:
        //   biased_acf[i] = realfft_ifft_output[i] / (pad * n).
        let scale = 1.0_f32 / (pad as f32 * n as f32);
        let mut out = vec![0.0_f32; n];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.acf_wf_time_out[i] * scale;
        }
        out
    }
```

- [ ] **Step 4: Run tests to verify the helper is correct**

Run: `cargo test -p dsp`
Expected: all v3 tests still pass, plus the two new `acf_fft_*` tests pass.

If `acf_fft_matches_naive_on_short_signal` fails, the scaling is off. The naive oracle in this plan is the source of truth — sanity-check the scale by computing `dsp.acf_fft_waveform(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])`'s lag-0 value. It must equal `1.0 / 8 = 0.125` (biased ACF of a unit impulse). If it equals 1.0, the `pad` factor is missing; if it equals `0.125 / pad`, the divisor is doubled.

- [ ] **Step 5: Add the same helper for RMS-envelope ACF**

Below `acf_fft_waveform`, add:

```rust
    /// Same as `acf_fft_waveform` but operates on the RMS history buffer
    /// (length `RMS_HISTORY_LEN`) using the separately-sized cached plans.
    pub fn acf_fft_rms(&mut self, input: &[f32]) -> Vec<f32> {
        let n = self.rms_history.len();
        debug_assert_eq!(input.len(), n, "acf_fft_rms expects len = RMS_HISTORY_LEN");
        let pad = self.acf_rms_time_in.len();
        self.acf_rms_time_in[..n].copy_from_slice(&input[..n]);
        for v in &mut self.acf_rms_time_in[n..pad] {
            *v = 0.0;
        }
        let _ = self
            .acf_rms_fwd
            .process(&mut self.acf_rms_time_in, &mut self.acf_rms_freq);
        for c in &mut self.acf_rms_freq {
            let mag_sq = c.re * c.re + c.im * c.im;
            c.re = mag_sq;
            c.im = 0.0;
        }
        let _ = self
            .acf_rms_inv
            .process(&mut self.acf_rms_freq, &mut self.acf_rms_time_out);
        // See scaling note in `acf_fft_waveform`.
        let scale = 1.0_f32 / (pad as f32 * n as f32);
        let mut out = vec![0.0_f32; n];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.acf_rms_time_out[i] * scale;
        }
        out
    }
```

Add a parallel test:

```rust
    #[test]
    fn acf_fft_rms_matches_naive_on_short_signal() {
        // We need the rms history buffer length to match for this test —
        // construct via the public path and hand-feed the naive oracle.
        let mut dsp = Dsp::new(8, 48000.0, 4);
        // Build an arbitrary signal of length RMS_HISTORY_LEN.
        let len = dsp.rms_history.len(); // 512 by default
        let x: Vec<f32> = (0..len)
            .map(|i| ((i as f32) * 0.07).sin() + 0.2 * ((i as f32) * 0.31).cos())
            .collect();
        let expected = naive_biased_acf(&x);
        let got = dsp.acf_fft_rms(&x);
        assert_eq!(got.len(), expected.len());
        // Floats accumulate more error at length 512; widen tolerance.
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "lag {}: fft {} vs naive {}",
                i,
                a,
                b
            );
        }
    }
```

- [ ] **Step 6: Run all tests**

Run: `cargo test -p dsp`
Expected: all tests pass — original v3 tests, the 3 new ACF helper tests.

- [ ] **Step 7: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): Wiener-Khinchin ACF helpers (verified vs naive O(N²) oracle)"
```

---

## Task 4: Equivalence test — FFT helpers vs v3 direct ACF

Before swapping the public API over, prove the FFT helpers reproduce v3's direct ACF output on a realistic signal.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Add equivalence tests to `mod tests`**

```rust
    #[test]
    fn acf_fft_waveform_matches_v3_direct() {
        // 1 kHz sine at 48 kHz — same signal v3 tests use elsewhere.
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let sr = 48000.0_f32;
        let freq = 1000.0_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();
        // Drive the direct path (process loads `waveform`).
        dsp.process(&signal);
        let direct = dsp.autocorrelation_waveform();
        // FFT helper takes the input directly.
        let fft = dsp.acf_fft_waveform(&signal);
        assert_eq!(direct.len(), fft.len());
        for (i, (d, f)) in direct.iter().zip(fft.iter()).enumerate() {
            assert!(
                (d - f).abs() < 1e-3,
                "lag {}: direct {} vs fft {}",
                i,
                d,
                f
            );
        }
    }

    #[test]
    fn acf_fft_rms_matches_v3_direct() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        // Drive `process` enough times to fill rms_history with non-trivial values.
        let sr = 48000.0_f32;
        for k in 0..30 {
            // Modulated sine — produces varying RMS per window.
            let amp = 0.5 + 0.4 * (k as f32 * 0.4).sin().abs();
            let signal: Vec<f32> = (0..2048)
                .map(|i| amp * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr).sin())
                .collect();
            dsp.process(&signal);
        }
        let rms_snapshot = dsp.rms_history.clone(); // ← will need pub(crate) or a test-only getter; see Step 2
        let direct = dsp.autocorrelation_rms();
        let fft = dsp.acf_fft_rms(&rms_snapshot);
        assert_eq!(direct.len(), fft.len());
        for (i, (d, f)) in direct.iter().zip(fft.iter()).enumerate() {
            assert!(
                (d - f).abs() < 1e-3,
                "lag {}: direct {} vs fft {}",
                i,
                d,
                f
            );
        }
    }
```

- [ ] **Step 2: If the second test fails to compile because `rms_history` is private, expose it for tests**

Inside the `#[cfg(test)]` block (still in `lib.rs`), the test module has access to private fields *if* the test module is nested in the same module — which it is. So `dsp.rms_history.clone()` should compile. If not (e.g., the field was made `pub(super)` or `pub`), update accordingly. No production code change required.

- [ ] **Step 3: Run the equivalence tests**

Run: `cargo test -p dsp`
Expected: all tests pass, including the two new equivalence tests. If they fail, debug the FFT helper *before* moving on. Likely culprits: scaling (off by `n` or `pad`), missing zero-padding, wrong magnitude formula (real²+imag² vs |c|).

- [ ] **Step 4: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "test(dsp): equivalence between FFT and direct ACF (waveform + RMS)"
```

---

## Task 5: Switch public ACF methods over to the FFT path

Replace the body of `autocorrelation_waveform` and `autocorrelation_rms` to delegate to the FFT helpers. Direct implementation stays in the file (unused) until Task 7 — gives us one more chance to compare.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Replace `autocorrelation_waveform` body**

Find the existing `pub fn autocorrelation_waveform(&mut self) -> Vec<f32>` (or whatever the v3 name is — adjust per Task 1). Replace its entire body with:

```rust
    pub fn autocorrelation_waveform(&mut self) -> Vec<f32> {
        // Wiener-Khinchin via cached 2N FFT plans. Equivalent to the legacy
        // direct implementation (see acf_fft_waveform_matches_v3_direct test).
        let waveform = self.waveform.clone();
        self.acf_fft_waveform(&waveform)
    }
```

- [ ] **Step 2: Replace `autocorrelation_rms` body**

Find `pub fn autocorrelation_rms(&mut self) -> Vec<f32>` (or its v3 equivalent). Replace its body with:

```rust
    pub fn autocorrelation_rms(&mut self) -> Vec<f32> {
        let rms = self.rms_history.clone();
        self.acf_fft_rms(&rms)
    }
```

- [ ] **Step 3: Run all tests**

Run: `cargo test -p dsp`
Expected: every test passes — the ones that exercised the direct path now exercise the FFT path under the hood, and the equivalence test continues to confirm parity with the (now-orphaned) direct helpers.

- [ ] **Step 4: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "refactor(dsp): autocorrelation methods now use Wiener-Khinchin FFT path"
```

---

## Task 6: Browser smoke-test before deleting the direct implementation

Before any code is removed, confirm the visualization is unchanged in the actual browser. The unit tests cover correctness; this catches anything else (worklet performance regressions, message timing, etc).

**Files:** none modified

- [ ] **Step 1: Rebuild WASM**

Run: `npm run wasm`
Expected: succeeds.

- [ ] **Step 2: Manual smoke test**

Run: `npm run dev`. Click "Test 440Hz".

Expected:
- The autocorrelation visualization (whatever shape v3 chose — line, heatmap, etc) looks identical to before this migration.
- For a 440 Hz tone at 48 kHz, the waveform ACF should peak at lag 0 and again at lag ≈ 109 (= 48000/440), with progressively smaller peaks at integer multiples.
- FPS overlay still ~60. No new console warnings.
- Clap or play music via the "Mic" button — ACF reacts as expected.

- [ ] **Step 3: Performance sanity check**

In the browser console, watch for any `[audio] underrun` or "AudioWorklet processor returned false" warnings. The FFT path should be faster than direct (O(N log N) vs O(N²) — at N=2048 that's ~93× fewer multiplies); the worst case is a regression somewhere else (e.g., extra allocation per call). If anything looks slower, profile with `performance.now()` around the worklet message.

- [ ] **Step 4: No commit (no code change in this task)**

---

## Task 7: Remove the direct implementation

Now that the FFT path is verified end-to-end, delete the orphaned direct ACF code.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Identify the direct-implementation lines**

These are the lines that v3 used to compute ACF directly — anything that's no longer called from `autocorrelation_waveform` / `autocorrelation_rms`. Per Task 1 Step 2, you noted the line range. Common shapes: a private helper (`fn direct_acf_biased(x: &[f32]) -> Vec<f32>`), or inlined loops inside the public methods (already replaced in Task 5 — nothing to delete).

If the direct implementation lives in a private helper, delete the helper. If it was inlined and Task 5 already removed it, skip to Step 2.

- [ ] **Step 2: Delete the equivalence-vs-direct tests too**

The `acf_fft_waveform_matches_v3_direct` and `acf_fft_rms_matches_v3_direct` tests in Task 4 referenced the direct path as the oracle. With direct gone, they'd just be testing FFT against itself — delete them.

The `acf_fft_matches_naive_on_short_signal` and `acf_fft_rms_matches_naive_on_short_signal` tests (Task 3) keep the FFT path honest against the inlined `naive_biased_acf` oracle — keep them.

- [ ] **Step 3: Run all tests**

Run: `cargo test -p dsp`
Expected: still green.

- [ ] **Step 4: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "refactor(dsp): remove direct ACF implementation (superseded by FFT path)"
```

---

## Task 8: Final acceptance + roadmap cleanup

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 1: Full Rust suite**

Run: `cargo test -p dsp`
Expected: green.

- [ ] **Step 2: Full JS suite**

Run: `npm test`
Expected: green.

- [ ] **Step 3: Production build**

Run: `npm run build`
Expected: succeeds. WASM bundle should be slightly smaller (direct ACF gone) or same; not larger.

- [ ] **Step 4: Roadmap cleanup**

In `ROADMAP.md`, in `Next → Performance`, remove this bullet:

```
- Migrate autocorrelation to FFT-based (Wiener–Khinchin: ACF = IFFT(|FFT(x)|²)) once the v3 direct implementation has proven the visualization is correct. O(N log N) vs O(N²); reuses the existing realfft planner. Folds the RMS-ACF onto the same code path.
```

- [ ] **Step 5: Commit roadmap**

```bash
git add ROADMAP.md
git commit -m "docs(roadmap): mark FFT-based ACF migration as shipped"
```
