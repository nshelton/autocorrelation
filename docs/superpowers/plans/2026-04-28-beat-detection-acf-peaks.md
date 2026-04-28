# Beat detection: ACF accumulator + peak picking implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a decaying EMA accumulator over the full-band RMS autocorrelation, extract top-10 sub-bin-precise tempo peaks from it, and visualize both via an overlaid accumulator line and a new `PeakMarkers` component on the existing `rms_acf` chart strip.

**Architecture:** Extend the Rust DSP (in worklet) with `rms_acf_accum` (EMA), `acf_peaks` (interleaved `[lag, mag]` × 10, NaN-padded), and `set_accum_tau_secs`. Worklet posts the two new buffers. Main thread renders one new `LineRenderer` overlaid on the existing rms_acf strip, plus a new `PeakMarkers` component (`THREE.LineSegments`, fixed 10-segment geometry, NaN slots collapse to invisible point). One new tunable param `dsp.accumTauSecs` (default 4 s) flows through the existing `ParamStore` → `WorkletBridge` → worklet pipeline.

**Tech Stack:** Rust + `realfft` + `wasm-bindgen` (DSP), TypeScript + Three.js WebGPU + tweakpane (rendering / params), vitest + happy-dom (TS tests), cargo (Rust tests).

**Spec:** `docs/superpowers/specs/2026-04-28-beat-detection-acf-peaks-design.md`

**Build commands:**
- Rust tests: `cargo test -p dsp` (from repo root)
- TS tests: `npm test`
- Rebuild wasm after Rust changes: `npm run wasm` (required before any TS that imports new wasm symbols can typecheck)
- Dev server (manual smoke test): `npm run dev`

---

## File map

**Create:**
- `src/render/PeakMarkers.ts` — vertical-line markers component
- `tests/render/PeakMarkers.test.ts` — tests for above

**Modify:**
- `crates/dsp/src/lib.rs` — accumulator + peak picking + new getters + tests
- `src/audio/dsp-worklet.ts` — `accumTauSecs` param branch, post new feature buffers, extend `configured` payload
- `src/params/schemas.ts` — register `dsp.accumTauSecs`
- `src/params/WorkletBridge.ts` — add `accumTauSecs` to `HOT_KEYS`
- `tests/params/WorkletBridge.test.ts` — extend bootstrap assertions, add change test
- `src/App.ts` — wire new feature buffers, new accumulator `LineRenderer`, new `PeakMarkers`
- `ROADMAP.md` — mark items 1, 2 of "Beatdetector improvements" shipped

---

## Task 1: Rust — accumulator fields, setter, EMA in `process()`

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1.1: Write the failing tests** — append to the `mod tests` block in `crates/dsp/src/lib.rs`:

```rust
#[test]
fn accum_alpha_matches_formula() {
    // alpha = 1 - exp(-dt / tau) at default tau (4.0 s), dt = 1024/48000
    let dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let dt = 1024.0_f32 / 48000.0;
    let expected = 1.0 - (-dt / ACCUM_TAU_DEFAULT_SECS).exp();
    assert!(
        (dsp.accum_alpha - expected).abs() < 1e-6,
        "got {}, expected {}",
        dsp.accum_alpha,
        expected
    );
}

#[test]
fn set_accum_tau_secs_clamps_and_recomputes() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let before = dsp.accum_alpha;
    dsp.set_accum_tau_secs(20.0);
    assert!(dsp.accum_alpha < before, "longer tau should yield smaller alpha");

    // Below clamp: 0.001 should clamp up to 0.05.
    dsp.set_accum_tau_secs(0.001);
    let dt = 1024.0_f32 / 48000.0;
    let expected = 1.0 - (-dt / 0.05).exp();
    assert!((dsp.accum_alpha - expected).abs() < 1e-6, "lower clamp not applied");

    // Above clamp: 1000.0 should clamp down to 60.0.
    dsp.set_accum_tau_secs(1000.0);
    let expected = 1.0 - (-dt / 60.0).exp();
    assert!((dsp.accum_alpha - expected).abs() < 1e-6, "upper clamp not applied");
}

#[test]
fn rms_acf_accum_silent_input_is_zero() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let silent = vec![0.0_f32; 2048];
    for _ in 0..200 {
        dsp.process(&silent);
    }
    let accum = dsp.rms_acf_accum();
    assert_eq!(accum.len(), 256);
    for &v in &accum {
        assert_eq!(v, 0.0, "silent → accumulator must stay zero, got {}", v);
    }
}

#[test]
fn rms_acf_accum_converges_to_instantaneous_for_steady_periodic() {
    // Feed a steady periodic signal long enough for both rms_history to fill
    // AND the accumulator to converge. After convergence,
    // rms_acf_accum[k] ≈ rms_acf[k] for all k.
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let sr = 48000.0_f32;
    // Use a slowly-modulated signal so RMS history has structure (not flat).
    let signal: Vec<f32> = (0..2048)
        .map(|i| {
            let t = i as f32 / sr;
            0.5 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                * (1.0 + 0.3 * (2.0 * std::f32::consts::PI * 4.0 * t).sin())
        })
        .collect();
    // Convergence rule of thumb: ~5τ at default 4 s tau, dt ≈ 21.33 ms → ~940 hops.
    // Use 1500 hops for headroom.
    for _ in 0..1500 {
        dsp.process(&signal);
    }
    let inst = dsp.rms_acf();
    let accum = dsp.rms_acf_accum();
    for (i, (a, b)) in accum.iter().zip(inst.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.05,
            "lag {}: accum {} should track inst {}",
            i, a, b
        );
    }
}
```

- [ ] **Step 1.2: Run tests to verify they fail** — `cargo test -p dsp accum`

Expected: compilation errors (`ACCUM_TAU_DEFAULT_SECS not defined`, `accum_alpha not a field`, `set_accum_tau_secs not a method`, `rms_acf_accum not a method`).

- [ ] **Step 1.3: Add constants near the existing `LOW_BAND_HZ_MAX` / `MID_BAND_HZ_MAX` block at the top of `crates/dsp/src/lib.rs`:**

```rust
/// Default time constant for the rms_acf decaying accumulator (seconds).
/// 4 s gives stable peaks for steady tempo while still tracking gradual
/// tempo changes within ~10–15 seconds.
const ACCUM_TAU_DEFAULT_SECS: f32 = 4.0;
```

- [ ] **Step 1.4: Add `rms_acf_accum` and `accum_alpha` fields to the `Dsp` struct.** Insert right after the `rms_acf` field:

```rust
    rms_acf: Vec<f32>,
    /// Decaying EMA accumulator over `rms_acf`. Same length. Used as the
    /// signal for tempo peak picking — the EMA suppresses per-frame noise
    /// in the instantaneous ACF so true tempo peaks build up.
    rms_acf_accum: Vec<f32>,
    /// Per-process EMA coefficient for `rms_acf_accum`. Computed from
    /// `accum_tau_secs` and the same `dt` used for `smoothing_alpha`:
    /// `alpha = 1 - exp(-dt / tau)`. Tunable via `set_accum_tau_secs`.
    accum_alpha: f32,
```

- [ ] **Step 1.5: Initialize the new fields in `Dsp::new`.** After the existing `parseval_band_scale` line, add:

```rust
        let accum_alpha = 1.0 - (-dt / ACCUM_TAU_DEFAULT_SECS).exp();
```

In the `Dsp { ... }` literal, after `rms_acf: vec![0.0; rms_history_len / 2],` add:

```rust
            rms_acf_accum: vec![0.0; rms_history_len / 2],
            accum_alpha,
```

- [ ] **Step 1.6: Add the `set_accum_tau_secs` setter.** Place it next to `set_smoothing_tau` inside the `#[wasm_bindgen] impl Dsp` block:

```rust
    /// Set the time constant (seconds) for the rms_acf decaying accumulator.
    /// `accum_alpha` is recomputed as `1 - exp(-dt / tau)`. Smaller tau →
    /// faster response, less stable peaks. Clamped to [0.05, 60.0] to avoid
    /// divide-by-zero and runaway settling.
    pub fn set_accum_tau_secs(&mut self, tau_secs: f32) {
        let tau = tau_secs.clamp(0.05, 60.0);
        self.accum_alpha = 1.0 - (-self.dt / tau).exp();
    }
```

- [ ] **Step 1.7: Add the `rms_acf_accum` getter** near the existing `rms_acf` getter:

```rust
    pub fn rms_acf_accum(&self) -> Vec<f32> {
        self.rms_acf_accum.clone()
    }
```

- [ ] **Step 1.8: Add the EMA update in `process()`.** Locate the line `autocorrelate(&self.rms_detrended, &mut self.rms_acf);` inside `process()`. Immediately after it, insert:

```rust
        // EMA-decayed accumulator over the instantaneous full-band ACF.
        // Builds up steady tempo peaks across many hops; suppresses
        // per-frame noise. Same alpha pattern as `smoothing_alpha`.
        for i in 0..self.rms_acf_accum.len() {
            self.rms_acf_accum[i] = self.accum_alpha * self.rms_acf[i]
                + (1.0 - self.accum_alpha) * self.rms_acf_accum[i];
        }
```

- [ ] **Step 1.9: Run tests to verify they pass** — `cargo test -p dsp accum`

Expected: 4 tests pass.

- [ ] **Step 1.10: Run full Rust test suite to ensure no regressions** — `cargo test -p dsp`

Expected: all existing tests still pass.

- [ ] **Step 1.11: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): EMA-decayed accumulator over rms_acf"
```

---

## Task 2: Rust — peak picking (constants, fields, algorithm, tests)

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 2.1: Write the failing tests** — append to the `mod tests` block:

```rust
// ---- Helper: directly seed `rms_acf_accum` for peak-picking tests ----
//
// Peak-picking tests want to assert on a known accumulator shape without
// having to construct an audio signal whose ACF lands at specific lags.
// Test-only setter on the Dsp struct (gated by `#[cfg(test)]`) lets the
// tests overwrite `rms_acf_accum` and call a direct peak-picking helper.

#[test]
fn acf_peaks_silent_input_all_nan() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    let silent = vec![0.0_f32; 2048];
    for _ in 0..50 {
        dsp.process(&silent);
    }
    let peaks = dsp.acf_peaks();
    assert_eq!(peaks.len(), 2 * MAX_PEAKS);
    for (i, &v) in peaks.iter().enumerate() {
        assert!(v.is_nan(), "slot {} should be NaN, got {}", i, v);
    }
}

#[test]
fn acf_peaks_min_lag_enforced() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    // Synthetic accumulator: a single isolated peak at lag 5 (below MIN_PEAK_LAG=10).
    let mut accum = vec![0.0_f32; 256];
    accum[5] = 0.9;
    dsp.test_set_rms_acf_accum(&accum);
    dsp.test_run_peak_picking();
    let peaks = dsp.acf_peaks();
    // No peak should be picked; all slots NaN.
    for &v in &peaks {
        assert!(v.is_nan(), "expected no peak below MIN_PEAK_LAG, got {}", v);
    }
}

#[test]
fn acf_peaks_finds_isolated_peak_with_subbin_offset() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    // Asymmetric triangular peak at integer lag 50:
    //   y0 = accum[49] = 0.6, y1 = accum[50] = 1.0, y2 = accum[51] = 0.8
    // Parabolic interp: δ = 0.5*(y0-y2)/(y0 - 2*y1 + y2) = 0.5*(-0.2)/(-0.6) ≈ 0.1667
    let mut accum = vec![0.0_f32; 256];
    accum[49] = 0.6;
    accum[50] = 1.0;
    accum[51] = 0.8;
    dsp.test_set_rms_acf_accum(&accum);
    dsp.test_run_peak_picking();
    let peaks = dsp.acf_peaks();
    let lag0 = peaks[0];
    assert!(!lag0.is_nan(), "expected a peak in slot 0");
    assert!(
        (lag0 - 50.1667).abs() < 0.01,
        "expected sub-bin lag ≈ 50.1667, got {}",
        lag0
    );
    // Slots 1..MAX_PEAKS must be NaN.
    for i in 1..MAX_PEAKS {
        assert!(peaks[2 * i].is_nan(), "slot {} lag should be NaN", i);
        assert!(peaks[2 * i + 1].is_nan(), "slot {} mag should be NaN", i);
    }
}

#[test]
fn acf_peaks_min_spacing_filters_nearby() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    // Two equal-magnitude lobes at lags 50 and 52 (spacing = 2 < MIN_PEAK_SPACING=3).
    // Both are local maxima individually because lag 51 sits between them with
    // a slightly lower value.
    let mut accum = vec![0.0_f32; 256];
    accum[49] = 0.8;
    accum[50] = 1.0;
    accum[51] = 0.85;
    accum[52] = 1.0;
    accum[53] = 0.8;
    dsp.test_set_rms_acf_accum(&accum);
    dsp.test_run_peak_picking();
    let peaks = dsp.acf_peaks();
    // Exactly one of the two lobes should be picked. Its integer lag rounds
    // to either 50 or 52.
    let mut accepted = 0;
    for i in 0..MAX_PEAKS {
        if !peaks[2 * i].is_nan() {
            accepted += 1;
        }
    }
    assert_eq!(accepted, 1, "min-spacing should leave only one peak");
}

#[test]
fn acf_peaks_top_n_selection_in_descending_magnitude() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    // 15 isolated peaks (well-spaced) with strictly decreasing magnitudes.
    // Spacing = 8 to satisfy MIN_PEAK_SPACING; first peak at lag 16 to clear
    // MIN_PEAK_LAG. Only the top 10 should be picked, in magnitude order.
    let mut accum = vec![0.0_f32; 256];
    for i in 0..15 {
        let lag = 16 + 8 * i;
        let mag = 1.0 - 0.05 * i as f32;
        accum[lag - 1] = mag * 0.5;
        accum[lag] = mag;
        accum[lag + 1] = mag * 0.5;
    }
    dsp.test_set_rms_acf_accum(&accum);
    dsp.test_run_peak_picking();
    let peaks = dsp.acf_peaks();
    // First 10 slots are real peaks, in descending magnitude order.
    let mut last_mag = f32::INFINITY;
    for i in 0..MAX_PEAKS {
        let lag = peaks[2 * i];
        let mag = peaks[2 * i + 1];
        assert!(!lag.is_nan(), "slot {}: expected real peak", i);
        assert!(mag <= last_mag + 1e-5, "slot {}: mag {} > prev {}", i, mag, last_mag);
        last_mag = mag;
    }
}

#[test]
fn acf_peaks_negative_correlations_skipped() {
    let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
    // Negative-magnitude local maximum: accum[50] = -0.1, surrounded by -0.5 / -0.5.
    // Even though -0.1 > -0.5, anti-correlations aren't beats and must be skipped.
    let mut accum = vec![0.0_f32; 256];
    accum[49] = -0.5;
    accum[50] = -0.1;
    accum[51] = -0.5;
    dsp.test_set_rms_acf_accum(&accum);
    dsp.test_run_peak_picking();
    let peaks = dsp.acf_peaks();
    for &v in &peaks {
        assert!(v.is_nan(), "negative peak must be skipped, got {}", v);
    }
}
```

- [ ] **Step 2.2: Run tests to verify they fail** — `cargo test -p dsp acf_peaks`

Expected: compilation errors (`MAX_PEAKS / MIN_PEAK_LAG / MIN_PEAK_SPACING undefined`, `acf_peaks()/test_set_rms_acf_accum/test_run_peak_picking not methods`).

- [ ] **Step 2.3: Add peak-picking constants near the other constants at the top of `lib.rs`:**

```rust
/// Minimum lag (in hops) considered for peak picking. Below this, peaks
/// imply BPM > ~280 (at hop=1024, sr=48000) which isn't a tempo we care
/// about, and the very-low-lag region of the ACF is dominated by the
/// shape of the autocorrelation envelope rather than tempo structure.
const MIN_PEAK_LAG: usize = 10;

/// Maximum number of tempo peaks tracked per hop. Drives the fixed length
/// of `acf_peaks` (= 2 * MAX_PEAKS — interleaved [lag, mag] pairs).
const MAX_PEAKS: usize = 10;

/// Minimum integer-lag distance between accepted peaks, in hops. Without
/// this, the wide lobes of true tempo peaks return multiple "peaks" all
/// clustered around a single underlying peak.
const MIN_PEAK_SPACING: usize = 3;
```

- [ ] **Step 2.4: Add peak fields to the `Dsp` struct.** After the `accum_alpha: f32,` field added in Task 1:

```rust
    /// Detected tempo peaks in `rms_acf_accum`, as interleaved
    /// [lag_frac, mag] pairs. Length = 2 * MAX_PEAKS. Unused slots filled
    /// with `f32::NAN` so the renderer can detect "no peak" with a single
    /// `isNaN` check (0.0 would collide with a valid lag).
    acf_peaks: Vec<f32>,
    /// Preallocated scratch for peak-candidate collection. Capacity is
    /// reserved at construction (`rms_acf_len / 2` — worst case every other
    /// lag is a local max). Cleared (not freed) each `process()` call so
    /// peak picking is allocation-free.
    peak_candidates: Vec<(usize, f32)>,
```

- [ ] **Step 2.5: Initialize the new fields in `Dsp::new`.** In the `Dsp { ... }` literal, after `accum_alpha,` add:

```rust
            acf_peaks: vec![f32::NAN; 2 * MAX_PEAKS],
            peak_candidates: Vec::with_capacity((rms_history_len / 2) / 2 + 1),
```

- [ ] **Step 2.6: Add the peak-picking method as a private (non-`pub`) helper inside the existing `#[wasm_bindgen] impl Dsp` block.** Wasm-bindgen ignores non-`pub` methods, so no impl-block splitting is needed. Place it after the existing public methods, before the closing brace of the `impl` block:

```rust
    /// Pick top-`MAX_PEAKS` tempo peaks in `rms_acf_accum` and write them
    /// into `acf_peaks` as interleaved `[lag_frac, mag]` pairs (NaN-padded).
    ///
    /// Algorithm:
    ///   1. Scan lags `MIN_PEAK_LAG..len-1` for positive local maxima.
    ///   2. Sort candidates by integer-lag magnitude, descending.
    ///   3. Greedy-select with `MIN_PEAK_SPACING` integer-lag separation.
    ///   4. Parabolic sub-bin interpolation on each accepted peak.
    ///
    /// Allocation-free: uses the preallocated `peak_candidates` scratch
    /// (cleared, not freed) and a stack-bounded accepted set.
    fn pick_acf_peaks(&mut self) {
        // Reset output to all-NaN sentinels.
        for slot in self.acf_peaks.iter_mut() {
            *slot = f32::NAN;
        }

        let n = self.rms_acf_accum.len();
        if n < MIN_PEAK_LAG + 2 {
            return;
        }

        // 1. Scan candidates.
        self.peak_candidates.clear();
        for k in MIN_PEAK_LAG..(n - 1) {
            let y1 = self.rms_acf_accum[k];
            if y1 > 0.0 && y1 > self.rms_acf_accum[k - 1] && y1 > self.rms_acf_accum[k + 1] {
                self.peak_candidates.push((k, y1));
            }
        }

        // 2. Sort by magnitude descending.
        self.peak_candidates
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Greedy select with min-spacing.
        let mut accepted: [u32; MAX_PEAKS] = [0; MAX_PEAKS];
        let mut accepted_count: usize = 0;
        for &(k, _mag) in self.peak_candidates.iter() {
            if accepted_count == MAX_PEAKS {
                break;
            }
            let mut too_close = false;
            for i in 0..accepted_count {
                let dist = (k as i32 - accepted[i] as i32).unsigned_abs() as usize;
                if dist < MIN_PEAK_SPACING {
                    too_close = true;
                    break;
                }
            }
            if !too_close {
                accepted[accepted_count] = k as u32;
                accepted_count += 1;
            }
        }

        // 4. Sub-bin parabolic refinement, write output.
        for i in 0..accepted_count {
            let k = accepted[i] as usize;
            let y0 = self.rms_acf_accum[k - 1];
            let y1 = self.rms_acf_accum[k];
            let y2 = self.rms_acf_accum[k + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let (lag_frac, mag) = if denom.abs() < 1e-12 {
                (k as f32, y1)
            } else {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                (k as f32 + delta, y1 - 0.25 * (y0 - y2) * delta)
            };
            self.acf_peaks[2 * i] = lag_frac;
            self.acf_peaks[2 * i + 1] = mag;
        }
    }
```

- [ ] **Step 2.7: Add the `acf_peaks` getter** near the `rms_acf_accum` getter (inside the `#[wasm_bindgen] impl Dsp` block):

```rust
    pub fn acf_peaks(&self) -> Vec<f32> {
        self.acf_peaks.clone()
    }
```

- [ ] **Step 2.8: Call `pick_acf_peaks` from `process()`.** Immediately after the EMA accumulator update added in Task 1:

```rust
        self.pick_acf_peaks();
```

- [ ] **Step 2.9: Add a separate test-only `impl Dsp` block** at the end of the file, just before the existing `#[cfg(test)] mod tests { ... }`. Keeping these helpers outside the `#[wasm_bindgen]` impl avoids any chance of wasm-bindgen trying to export them:

```rust
#[cfg(test)]
impl Dsp {
    pub fn test_set_rms_acf_accum(&mut self, src: &[f32]) {
        let n = src.len().min(self.rms_acf_accum.len());
        self.rms_acf_accum[..n].copy_from_slice(&src[..n]);
        for v in self.rms_acf_accum.iter_mut().skip(n) {
            *v = 0.0;
        }
    }

    pub fn test_run_peak_picking(&mut self) {
        self.pick_acf_peaks();
    }
}
```

- [ ] **Step 2.10: Run tests to verify they pass** — `cargo test -p dsp acf_peaks`

Expected: 6 tests pass.

- [ ] **Step 2.11: Run full Rust test suite to ensure no regressions** — `cargo test -p dsp`

Expected: all tests pass.

- [ ] **Step 2.12: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): top-N tempo peak picking on rms_acf accumulator"
```

---

## Task 3: Worklet — wire `accumTauSecs` param + post new feature buffers

**Files:**
- Modify: `src/audio/dsp-worklet.ts`

This task has no separate test layer (the worklet runs in a real `AudioWorkletGlobalScope` and can't be reasonably unit-tested with `vitest`). The `WorkletBridge` test (Task 5) covers the param-message wire format; the param routing inside the worklet is exercised by manual smoke test (Task 8). The new feature-buffer plumbing is exercised by the `App.ts` work in Task 7 + manual smoke.

- [ ] **Step 3.1: Rebuild wasm so the new methods are visible to TS** — `npm run wasm`

Expected: wasm-pack build completes; `src/wasm-pkg/dsp.d.ts` now declares `set_accum_tau_secs`, `rms_acf_accum`, `acf_peaks`.

- [ ] **Step 3.2: Extend the `WorkletInbound` param key union** at the top of `src/audio/dsp-worklet.ts`:

```ts
type WorkletInbound =
  | { type: "configure"; windowSize: number; rmsHistoryLen: number }
  | { type: "param"; key: "hopSize" | "smoothingTauSecs" | "dbFloor" | "accumTauSecs"; value: number };
```

- [ ] **Step 3.3: Add `accumTauSecs` field with a default** — inside `class DSPProcessor`, near the existing `private smoothingTauSecs = 0.0956;`:

```ts
  private accumTauSecs = 4.0;
```

- [ ] **Step 3.4: Route `accumTauSecs` in `onMessage`.** Inside the `if (msg.type === "param")` block, alongside the existing `smoothingTauSecs` branch:

```ts
      } else if (msg.key === "accumTauSecs") {
        this.accumTauSecs = msg.value;
        if (this.ready && this.dsp) this.dsp.set_accum_tau_secs(msg.value);
      } else if (msg.key === "dbFloor") {
```

(That is: insert the new branch *before* the existing `dbFloor` branch so the chained `else if` reads cleanly.)

- [ ] **Step 3.5: Apply `accumTauSecs` in `applyConfigure`.** After the existing `this.dsp.set_db_floor(this.dbFloor);`:

```ts
    this.dsp.set_accum_tau_secs(this.accumTauSecs);
```

- [ ] **Step 3.6: Extend the `configured` postMessage payload** with the new length field. In `applyConfigure`:

```ts
    this.port.postMessage({
      type: "configured",
      waveformLen: this.windowSize,
      spectrumLen: this.windowSize / 2,
      bufferAcfLen: this.windowSize / 2,
      rmsLen: this.rmsHistoryLen,
      rmsAcfLen: this.rmsHistoryLen / 2,
      acfPeaksLen: 20,
    });
```

- [ ] **Step 3.7: Post the new feature buffers each hop.** In `process()`, alongside the existing `const ra = new Float32Array(this.dsp.rms_acf());`:

```ts
      const raAccum = new Float32Array(this.dsp.rms_acf_accum());
      const peaks = new Float32Array(this.dsp.acf_peaks());
```

Then extend the `postMessage` call:

```ts
      this.port.postMessage(
        {
          type: "features",
          waveform: wf,
          spectrum: sp,
          rms,
          bufferAcf: ba,
          rmsAcf: ra,
          rmsAcfAccum: raAccum,
          acfPeaks: peaks,
          rmsLow,
          rmsMid,
          rmsHigh,
          rmsAcfLow,
        },
        [
          wf.buffer,
          sp.buffer,
          rms.buffer,
          ba.buffer,
          ra.buffer,
          raAccum.buffer,
          peaks.buffer,
          rmsLow.buffer,
          rmsMid.buffer,
          rmsHigh.buffer,
          rmsAcfLow.buffer,
        ],
      );
```

- [ ] **Step 3.8: Verify the file typechecks** — `npx tsc --noEmit -p tsconfig.json` (or `npm run build` if no tsc-only script).

Expected: no TS errors.

- [ ] **Step 3.9: Commit**

```bash
git add src/audio/dsp-worklet.ts
git commit -m "feat(audio): worklet posts rms_acf_accum + acf_peaks; routes accumTauSecs"
```

---

## Task 4: ParamStore schema — register `dsp.accumTauSecs`

**Files:**
- Modify: `src/params/schemas.ts`

- [ ] **Step 4.1: Add the schema** to the `analysisSchemas` array in `src/params/schemas.ts`. Insert as the last entry, after the existing `dsp.dbFloor`:

```ts
  {
    key: "dsp.accumTauSecs",
    label: "ACF accumulator τ (s)",
    kind: "continuous",
    min: 0.05,
    max: 60.0,
    step: 0.1,
    default: 4.0,
    reconfig: false,
  },
```

- [ ] **Step 4.2: Run TS tests to confirm no schema regressions** — `npm test -- tests/params`

Expected: existing param tests pass. (The `WorkletBridge` bootstrap test will fail next task because it currently asserts `expect(calls.length).toBe(4)` — that's expected and fixed in Task 5.)

NOTE: if any existing test fails *now* purely due to the new param being registered (e.g., a snapshot or a strict-count assertion outside `WorkletBridge.test.ts`), that's a real regression — investigate before proceeding.

- [ ] **Step 4.3: Commit**

```bash
git add src/params/schemas.ts
git commit -m "feat(params): register dsp.accumTauSecs (default 4 s, range 0.05-60)"
```

---

## Task 5: WorkletBridge — extend `HOT_KEYS`, update tests

**Files:**
- Modify: `src/params/WorkletBridge.ts`
- Modify: `tests/params/WorkletBridge.test.ts`

- [ ] **Step 5.1: Update the existing bootstrap test and add a new change test.** Replace the existing `bootstrap posts one configure + three param messages...` test, and add a new test for `accumTauSecs` change. The relevant section of `tests/params/WorkletBridge.test.ts` becomes:

```ts
  it("bootstrap posts one configure + four param messages with current store values", () => {
    const store = makeStore();
    const port = makePort();
    const bridge = new WorkletBridge(store, port);
    bridge.bootstrap();
    const calls = (port.postMessage as ReturnType<typeof vi.fn>).mock.calls.map((c) => c[0]);
    expect(calls).toContainEqual({ type: "configure", windowSize: 2048, rmsHistoryLen: 512 });
    expect(calls).toContainEqual({ type: "param", key: "hopSize", value: 1024 });
    expect(calls).toContainEqual({ type: "param", key: "smoothingTauSecs", value: 0.0956 });
    expect(calls).toContainEqual({ type: "param", key: "dbFloor", value: -100 });
    expect(calls).toContainEqual({ type: "param", key: "accumTauSecs", value: 4.0 });
    expect(calls.length).toBe(5);
  });
```

And add (after the existing `smoothingTauSecs change posts a param message...` test):

```ts
  it("accumTauSecs change posts a param message with the hot key (no dsp prefix)", () => {
    const store = makeStore();
    const port = makePort();
    new WorkletBridge(store, port);
    (port.postMessage as ReturnType<typeof vi.fn>).mockClear();
    store.set("dsp.accumTauSecs", 8.0);
    expect(port.postMessage).toHaveBeenCalledWith({
      type: "param",
      key: "accumTauSecs",
      value: 8.0,
    });
  });
```

- [ ] **Step 5.2: Run tests to verify they fail** — `npm test -- tests/params/WorkletBridge.test.ts`

Expected: the bootstrap test fails because the bridge currently posts only 4 messages and rejects `accumTauSecs` as unknown; the new change test fails for the same reason.

- [ ] **Step 5.3: Add `accumTauSecs` to the `HOT_KEYS` tuple** in `src/params/WorkletBridge.ts`:

```ts
const HOT_KEYS = ["hopSize", "smoothingTauSecs", "dbFloor", "accumTauSecs"] as const;
```

- [ ] **Step 5.4: Run tests to verify they pass** — `npm test -- tests/params/WorkletBridge.test.ts`

Expected: all `WorkletBridge` tests pass (5 tests).

- [ ] **Step 5.5: Commit**

```bash
git add src/params/WorkletBridge.ts tests/params/WorkletBridge.test.ts
git commit -m "feat(params): WorkletBridge forwards accumTauSecs hot key"
```

---

## Task 6: `PeakMarkers` component (TDD)

**Files:**
- Create: `src/render/PeakMarkers.ts`
- Test: `tests/render/PeakMarkers.test.ts`

- [ ] **Step 6.1: Write the failing tests** at `tests/render/PeakMarkers.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { BufferGeometry, LineSegments } from "three";
import { PeakMarkers } from "../../src/render/PeakMarkers";

const linearXForLag = (lag: number, lagDomain: number) =>
  lagDomain <= 1 ? 0 : (lag / (lagDomain - 1)) * 2 - 1;

describe("PeakMarkers", () => {
  it("constructs LineSegments with 2 * maxPeaks vertices", () => {
    const data = new Float32Array(20).fill(NaN);
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
    });
    expect(pm.object3d).toBeInstanceOf(LineSegments);
    const geom = (pm.object3d as LineSegments).geometry as BufferGeometry;
    expect(geom.getAttribute("position").count).toBe(20);
    expect(geom.getAttribute("color").count).toBe(20);
  });

  it("update() with all-NaN source collapses every segment to (0, yCenter, 0) with black color", () => {
    const data = new Float32Array(20).fill(NaN);
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
    });
    pm.update();
    const geom = (pm.object3d as LineSegments).geometry as BufferGeometry;
    const pos = geom.getAttribute("position");
    const col = geom.getAttribute("color");
    for (let i = 0; i < 20; i++) {
      expect(pos.getX(i)).toBe(0);
      expect(pos.getY(i)).toBeCloseTo(-1.0);
      expect(pos.getZ(i)).toBe(0);
      expect(col.getX(i)).toBe(0);
      expect(col.getY(i)).toBe(0);
      expect(col.getZ(i)).toBe(0);
    }
  });

  it("update() places real peaks as vertical segments at xForLag(lag)", () => {
    const data = new Float32Array(20).fill(NaN);
    // Two peaks: lag 64 (mag 0.9), lag 128 (mag 0.5).
    data[0] = 64;
    data[1] = 0.9;
    data[2] = 128;
    data[3] = 0.5;
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
    });
    pm.update();
    const pos = ((pm.object3d as LineSegments).geometry as BufferGeometry).getAttribute("position");

    // Peak 0 (lag=64): x = (64/255)*2 - 1 ≈ -0.498
    const expectedX0 = (64 / 255) * 2 - 1;
    expect(pos.getX(0)).toBeCloseTo(expectedX0);
    expect(pos.getX(1)).toBeCloseTo(expectedX0);
    // Top vertex (yCenter + ySpan) and bottom vertex (yCenter - ySpan).
    const ys = [pos.getY(0), pos.getY(1)];
    expect(Math.max(...ys)).toBeCloseTo(-1.0 + 0.4);
    expect(Math.min(...ys)).toBeCloseTo(-1.0 - 0.4);

    // Peak 1 (lag=128).
    const expectedX1 = (128 / 255) * 2 - 1;
    expect(pos.getX(2)).toBeCloseTo(expectedX1);
    expect(pos.getX(3)).toBeCloseTo(expectedX1);

    // Slot 2 (NaN) collapsed to center.
    expect(pos.getX(4)).toBe(0);
    expect(pos.getY(4)).toBeCloseTo(-1.0);
  });

  it("color brightens at slot 0 and dims toward slot maxPeaks-1", () => {
    const data = new Float32Array(20);
    for (let i = 0; i < 10; i++) {
      data[2 * i] = 20 + 5 * i;
      data[2 * i + 1] = 0.9;
    }
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
      baseColor: 0xffff00, // pure yellow → R=1, G=1, B=0
    });
    pm.update();
    const col = ((pm.object3d as LineSegments).geometry as BufferGeometry).getAttribute("color");
    // Slot 0: full brightness → R=1, G=1.
    expect(col.getX(0)).toBeCloseTo(1.0);
    expect(col.getY(0)).toBeCloseTo(1.0);
    // Slot 9: dimmest → ~0.25 of full.
    expect(col.getX(2 * 9)).toBeCloseTo(0.25);
    expect(col.getY(2 * 9)).toBeCloseTo(0.25);
    // Both endpoints of a segment share the same color.
    expect(col.getX(0)).toBe(col.getX(1));
    expect(col.getY(0)).toBe(col.getY(1));
  });

  it("dispose() releases geometry + material", () => {
    const data = new Float32Array(20).fill(NaN);
    const pm = new PeakMarkers({
      source: () => data,
      maxPeaks: 10,
      lagDomain: 256,
      yCenter: -1.0,
      ySpan: 0.4,
      xForLag: linearXForLag,
    });
    const geom = (pm.object3d as LineSegments).geometry as BufferGeometry;
    let geomDisposed = false;
    geom.addEventListener("dispose", () => {
      geomDisposed = true;
    });
    pm.dispose();
    expect(geomDisposed).toBe(true);
  });
});
```

- [ ] **Step 6.2: Run tests to verify they fail** — `npm test -- tests/render/PeakMarkers.test.ts`

Expected: import error — `PeakMarkers` does not exist.

- [ ] **Step 6.3: Create the `PeakMarkers` component** at `src/render/PeakMarkers.ts`:

```ts
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  ColorRepresentation,
  DynamicDrawUsage,
  LineBasicMaterial,
  LineSegments,
  Object3D,
} from "three";

export interface PeakMarkersOptions {
  source: () => Float32Array;
  maxPeaks: number;
  lagDomain: number;
  yCenter: number;
  ySpan: number;
  xForLag: (lag: number, lagDomain: number) => number;
  baseColor?: ColorRepresentation;
}

export class PeakMarkers {
  readonly object3d: Object3D;
  private source: () => Float32Array;
  private maxPeaks: number;
  private lagDomain: number;
  private yCenter: number;
  private ySpan: number;
  private xForLag: (lag: number, lagDomain: number) => number;
  private baseR: number;
  private baseG: number;
  private baseB: number;
  private positions: Float32Array;
  private colors: Float32Array;
  private positionAttribute: BufferAttribute;
  private colorAttribute: BufferAttribute;

  constructor(opts: PeakMarkersOptions) {
    this.source = opts.source;
    this.maxPeaks = opts.maxPeaks;
    this.lagDomain = opts.lagDomain;
    this.yCenter = opts.yCenter;
    this.ySpan = opts.ySpan;
    this.xForLag = opts.xForLag;

    const base = new Color(opts.baseColor ?? 0xffff66);
    this.baseR = base.r;
    this.baseG = base.g;
    this.baseB = base.b;

    const vertexCount = 2 * this.maxPeaks;
    this.positions = new Float32Array(vertexCount * 3);
    this.colors = new Float32Array(vertexCount * 3);

    this.positionAttribute = new BufferAttribute(this.positions, 3);
    this.positionAttribute.setUsage(DynamicDrawUsage);
    this.colorAttribute = new BufferAttribute(this.colors, 3);
    this.colorAttribute.setUsage(DynamicDrawUsage);

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", this.positionAttribute);
    geometry.setAttribute("color", this.colorAttribute);

    const material = new LineBasicMaterial({ vertexColors: true });
    this.object3d = new LineSegments(geometry, material);
  }

  update(): void {
    const src = this.source();
    const denom = this.maxPeaks > 1 ? this.maxPeaks - 1 : 1;
    for (let i = 0; i < this.maxPeaks; i++) {
      const lag = src[2 * i];
      const top = i * 2;
      const bot = i * 2 + 1;
      if (Number.isNaN(lag)) {
        this.writeVertex(top, 0, this.yCenter, 0, 0);
        this.writeVertex(bot, 0, this.yCenter, 0, 0);
        continue;
      }
      const x = this.xForLag(lag, this.lagDomain);
      const brightness = 1.0 - 0.75 * (i / denom);
      this.writeVertex(top, x, this.yCenter + this.ySpan, 0, brightness);
      this.writeVertex(bot, x, this.yCenter - this.ySpan, 0, brightness);
    }
    this.positionAttribute.needsUpdate = true;
    this.colorAttribute.needsUpdate = true;
  }

  dispose(): void {
    const seg = this.object3d as LineSegments;
    (seg.geometry as BufferGeometry).dispose();
    (seg.material as LineBasicMaterial).dispose();
    seg.parent?.remove(seg);
  }

  private writeVertex(vertexIndex: number, x: number, y: number, z: number, brightness: number): void {
    const off = vertexIndex * 3;
    this.positions[off] = x;
    this.positions[off + 1] = y;
    this.positions[off + 2] = z;
    this.colors[off] = this.baseR * brightness;
    this.colors[off + 1] = this.baseG * brightness;
    this.colors[off + 2] = this.baseB * brightness;
  }
}
```

- [ ] **Step 6.4: Run tests to verify they pass** — `npm test -- tests/render/PeakMarkers.test.ts`

Expected: 5 tests pass.

- [ ] **Step 6.5: Run full TS test suite to ensure no regressions** — `npm test`

Expected: all tests pass.

- [ ] **Step 6.6: Commit**

```bash
git add src/render/PeakMarkers.ts tests/render/PeakMarkers.test.ts
git commit -m "feat(render): PeakMarkers component for ACF tempo peaks"
```

---

## Task 7: `App.ts` — wire accumulator line + peak markers

**Files:**
- Modify: `src/App.ts`

This task has no automated test (the App is the composition root and isn't unit-tested). It's exercised by manual smoke test in Task 8.

- [ ] **Step 7.1: Add a new `import` for `PeakMarkers`** at the top of `src/App.ts`, near the existing `import { LineRenderer }`:

```ts
import { PeakMarkers } from "./render/PeakMarkers";
```

- [ ] **Step 7.2: Add new fields on `App`** alongside the existing `private rmsAcfLine?: LineRenderer;`:

```ts
  private rmsAcfAccumLine?: LineRenderer;
  private peakMarkers?: PeakMarkers;
```

- [ ] **Step 7.3: Extend the `features` message handler.** In the `node.port.onmessage = (e) => {...}` block, where the existing message-shape type is declared, add the two new optional fields:

```ts
      const msg = e.data as
        | { type: "features"; waveform?: Float32Array; spectrum?: Float32Array; rms?: Float32Array; bufferAcf?: Float32Array; rmsAcf?: Float32Array; rmsAcfAccum?: Float32Array; acfPeaks?: Float32Array; rmsLow?: Float32Array; rmsMid?: Float32Array; rmsHigh?: Float32Array; rmsAcfLow?: Float32Array }
        | { type: "configured"; waveformLen: number; spectrumLen: number; bufferAcfLen: number; rmsLen: number; rmsAcfLen: number; acfPeaksLen: number };
```

Then in the `if (msg.type === "features")` branch, alongside the existing `if (msg.rmsAcf) ...`:

```ts
        if (msg.rmsAcfAccum) this.store.set("rmsAcfAccum", msg.rmsAcfAccum);
        if (msg.acfPeaks) this.store.set("acfPeaks", msg.acfPeaks);
```

- [ ] **Step 7.4: Update `rebuildLineRenderers` signature** to receive `acfPeaksLen`:

```ts
  private rebuildLineRenderers(sizes: {
    waveformLen: number;
    spectrumLen: number;
    bufferAcfLen: number;
    rmsLen: number;
    rmsAcfLen: number;
    acfPeaksLen: number;
  }): void {
```

- [ ] **Step 7.5: Dispose the new renderers at the top of `rebuildLineRenderers`.** Extend the existing dispose loop:

```ts
    for (const line of [
      this.waveformLine,
      this.bufferAcfLine,
      this.spectrumLine,
      this.lowRmsLine,
      this.midRmsLine,
      this.highRmsLine,
      this.rmsLine,
      this.lowRmsAcfLine,
      this.rmsAcfLine,
      this.rmsAcfAccumLine,
    ]) {
      line?.dispose();
    }
    this.peakMarkers?.dispose();
```

- [ ] **Step 7.6: Seed the new feature-store buffers** in `rebuildLineRenderers`. After the existing `this.store.set("rmsAcf", new Float32Array(sizes.rmsAcfLen));`:

```ts
    this.store.set("rmsAcfAccum", new Float32Array(sizes.rmsAcfLen));
    const peaksInit = new Float32Array(sizes.acfPeaksLen);
    peaksInit.fill(NaN);
    this.store.set("acfPeaks", peaksInit);
```

- [ ] **Step 7.7: Build the new accumulator line + peak markers.** After the existing `this.scene.add(this.rmsAcfLine.object3d);`:

```ts
    this.rmsAcfAccumLine = new LineRenderer({
      source: () => this.store.get("rmsAcfAccum"),
      layout: linearLayout(-1.0, 0.4),
      color: 0x66ffff,
    });
    this.scene.add(this.rmsAcfAccumLine.object3d);

    this.peakMarkers = new PeakMarkers({
      source: () => this.store.get("acfPeaks"),
      maxPeaks: 10,
      lagDomain: sizes.rmsAcfLen,
      yCenter: -1.0,
      ySpan: 0.4,
      // Match linearLayout's x-mapping exactly (i / (n-1)) * 2 - 1.
      xForLag: (lag, n) => (n <= 1 ? 0 : (lag / (n - 1)) * 2 - 1),
      baseColor: 0xffff66,
    });
    this.scene.add(this.peakMarkers.object3d);
```

- [ ] **Step 7.8: Update the render loop** to call `update()` on the new renderers. Inside the `loop` function, alongside the existing `this.rmsAcfLine?.update();`:

```ts
      this.rmsAcfAccumLine?.update();
      this.peakMarkers?.update();
```

- [ ] **Step 7.9: Verify the file typechecks** — `npm run build` (or `npx tsc --noEmit -p tsconfig.json`).

Expected: no TS errors.

- [ ] **Step 7.10: Run the full TS test suite** — `npm test`.

Expected: all tests pass.

- [ ] **Step 7.11: Commit**

```bash
git add src/App.ts
git commit -m "feat(app): render rms_acf accumulator overlay + peak markers"
```

---

## Task 8: Manual smoke test + ROADMAP update

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 8.1: Rebuild wasm if any Rust files have changed since the last `npm run wasm`** — `npm run wasm`.

Expected: clean build.

- [ ] **Step 8.2: Start the dev server** — `npm run dev`.

Expected: server up at `http://localhost:5173`.

- [ ] **Step 8.3: Manual smoke checklist (browser).** Open the page, allow mic (or press `T` for test source), then verify:

  - The existing pink rms_acf line still renders at the bottom strip.
  - A new cyan line is overlaid in the same y-band — should be visibly smoother / slower-moving than the pink one.
  - Yellow vertical line markers appear on the bottom strip after a few seconds, brightest near the front of the list (highest peak), dimmer toward the dimmer ones.
  - Press key `6` to focus the rms_acf chart preset and confirm the markers sit on top of the cyan accumulator's local maxima.
  - Open the tweakpane param panel and find the new "ACF accumulator τ (s)" slider. Drag it from default 4 s down to 0.5 s — the cyan line should become much noisier / closer to the pink one. Drag back to 8 s — line should become very smooth.
  - Press `T` to switch to test source; confirm peaks still appear at the test sine's autocorrelation lag.

If any of these fail, **do not proceed to Step 8.4.** Investigate, fix, write a regression test if appropriate, and re-run.

- [ ] **Step 8.4: Update `ROADMAP.md`.** In the "Beatdetector improvements" section, mark items 1 and 2 as shipped. Replace:

```
## Beatdetector improvements
1. add decaying accumulator to the rms autocorrelation
2. find peaks for BPM
3. generate a sawtooth wave -assume 4 beats ? assume we got the measure ?
4. onset detection for down beat - this is hard
```

with:

```
## Beatdetector improvements
1. ~~add decaying accumulator to the rms autocorrelation~~ — shipped
2. ~~find peaks for BPM~~ (peak picking shipped; BPM grid-fit deferred)
3. generate a sawtooth wave -assume 4 beats ? assume we got the measure ?
4. onset detection for down beat - this is hard
5. BPM grid-fit: take detected peaks, find the quarter-note grid that hits the most peaks, output BPM
```

Also append a line to the "Shipped" list at the bottom:

```
- **v3.2** (tag pending): rms_acf decaying EMA accumulator + top-10 sub-bin tempo peak picking with PeakMarkers visualization; new `dsp.accumTauSecs` analysis param
```

- [ ] **Step 8.5: Commit**

```bash
git add ROADMAP.md
git commit -m "docs(roadmap): mark beat-detection items 1, 2 shipped"
```

---

## Self-review checklist (run before handoff)

- [ ] **Spec coverage:**
  - Decaying accumulator → Task 1 ✓
  - Peak picking with sub-bin interpolation, top 10, lags ≥ 10, min spacing 3 → Task 2 ✓
  - PeakMarkers vertical-line visualization → Task 6 ✓
  - Accumulator overlay on existing chart → Task 7 ✓
  - `dsp.accumTauSecs` tunable param → Tasks 3, 4, 5 ✓
  - NaN sentinel for empty peak slots → Tasks 2, 6, 7 ✓
  - Allocation-free peak picking → Task 2 (preallocated `peak_candidates`, stack accepted set) ✓
  - Tests for accumulator EMA, clamps, peak picking, NaN propagation, color-by-rank → Tasks 1, 2, 5, 6 ✓

- [ ] **Type / name consistency:**
  - `accumTauSecs` (TS hot key) ↔ `accum_tau_secs` (only as parameter name in Rust setter; field is `accum_alpha`) ✓
  - `rmsAcfAccum` / `acfPeaks` (TS) ↔ `rms_acf_accum` / `acf_peaks` (Rust) — convention matches existing code ✓
  - `MAX_PEAKS = 10` ↔ `acfPeaksLen: 20` ↔ `2 * MAX_PEAKS` ↔ tests use `20` and `10` consistently ✓
  - `PeakMarkers` class + `PeakMarkersOptions` interface ↔ test imports + App imports match ✓
  - `xForLag` callback signature `(lag: number, lagDomain: number) => number` consistent across Task 6 + Task 7 ✓
  - `linearLayout(-1.0, 0.4)` y-band shared between rms_acf line, accumulator line, and peak markers (yCenter=-1.0, ySpan=0.4) ✓

- [ ] **Placeholder scan:** No "TBD", "implement later", or empty steps. Every code-touching step shows the code.
