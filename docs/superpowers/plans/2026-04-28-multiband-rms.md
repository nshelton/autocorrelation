# Multi-Band RMS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-band (low/mid/high) RMS time-series and a low-band RMS-ACF beat detector to the Rust DSP. Expose them through the worklet to the main thread, where they render as a R/G/B overlay on the existing white RMS line and a red overlay on the existing rms_acf line.

**Architecture:** Bands are derived from the existing FFT — sum `|FFT[k]|²` over each band's bin range, multiply by a Parseval-correct windowing scale, take sqrt → band RMS commensurate with the time-domain full-band RMS (so `low² + mid² + high² ≈ full²`). The Rust `Dsp` gains three rolling band-RMS histories (length 512, mirroring `rms_history`) and one detrended-ACF on the low-band history (length 256, mirroring `rms_acf`). Worklet posts four new transferable Float32Arrays per hop; App overlays four new `LineRenderer`s. No new dependencies, no real-time-domain filters.

**Tech Stack:** Rust + `realfft` (existing), TypeScript AudioWorklet (existing), Three.js WebGPU `LineRenderer` (existing). Spec at `docs/superpowers/specs/2026-04-28-multiband-rms-design.md`.

---

## Task 1: Rust DSP — bin-range derivation + Parseval scale plumbing

Add the constants, helper, and fields needed for band computations. No `process()` change yet — this task only adds infrastructure that the next two tasks consume. Constructor wiring + tests prove the values are correct at the existing default settings.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the three failing tests**

Add these tests at the end of the existing `mod tests` block in `crates/dsp/src/lib.rs`:

```rust
    #[test]
    fn bin_for_hz_snaps_at_default_settings() {
        // 150 Hz at sr=48000, N=2048: 150 * 2048 / 48000 = 6.4 → 6.
        // 1500 Hz: 1500 * 2048 / 48000 = 64.0 → 64.
        assert_eq!(bin_for_hz(150.0, 48000.0, 2048), 6);
        assert_eq!(bin_for_hz(1500.0, 48000.0, 2048), 64);
    }

    #[test]
    fn band_bin_ends_at_default_settings() {
        let dsp = Dsp::new(2048, 48000.0, 1024);
        assert_eq!(dsp.low_band_bin_end, 6);
        assert_eq!(dsp.mid_band_bin_end, 64);
    }

    #[test]
    fn parseval_band_scale_matches_formula() {
        // parseval_band_scale = 2 / (N · Σ hann²)
        let dsp = Dsp::new(2048, 48000.0, 1024);
        let n = 2048usize;
        let hann_energy: f32 = (0..n)
            .map(|i| {
                let h = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();
                h * h
            })
            .sum();
        let expected = 2.0 / (n as f32 * hann_energy);
        assert!(
            (dsp.parseval_band_scale - expected).abs() < 1e-10,
            "got {}, expected {}",
            dsp.parseval_band_scale,
            expected
        );
    }
```

- [ ] **Step 2: Run tests — verify compile failure**

Run: `cargo test -p dsp`
Expected: compile errors — `cannot find function 'bin_for_hz'`, `no field 'low_band_bin_end'`, `no field 'mid_band_bin_end'`, `no field 'parseval_band_scale'`.

- [ ] **Step 3: Add the new constants**

Just below the existing `SMOOTHING_TAU_SECS` / `DB_FLOOR` / `RMS_HISTORY_LEN` / `RMS_ACF_LEN` constants block in `crates/dsp/src/lib.rs`, add:

```rust
/// Crossover from low band to mid band, in Hz. Drum-friendly default:
/// fits the kick fundamental (typically 50–90 Hz) cleanly inside "low"
/// without bleeding much into snare body.
const LOW_BAND_HZ_MAX: f32 = 150.0;
/// Crossover from mid band to high band, in Hz.
const MID_BAND_HZ_MAX: f32 = 1500.0;
```

- [ ] **Step 4: Add the `bin_for_hz` helper**

After the existing `autocorrelate` free function in `crates/dsp/src/lib.rs`, add:

```rust
/// Snap a frequency in Hz to the nearest one-sided real-FFT bin index,
/// clamped to [1, N/2 - 1] (DC and Nyquist are excluded by design).
fn bin_for_hz(hz: f32, sample_rate: f32, n: usize) -> usize {
    let bin = (hz * n as f32 / sample_rate).round() as usize;
    bin.clamp(1, n / 2 - 1)
}
```

- [ ] **Step 5: Add the three new fields to `Dsp`**

In the `Dsp` struct definition, add these fields at the end (after the existing `smoothing_alpha`):

```rust
    /// Inclusive last bin index of the low band. Low band = bins 1..=low_band_bin_end.
    /// Bin 0 (DC) is always skipped.
    low_band_bin_end: usize,
    /// Inclusive last bin index of the mid band. Mid = (low_end+1)..=mid_band_bin_end.
    /// High = (mid_end+1)..=N/2-1 (Nyquist excluded).
    mid_band_bin_end: usize,
    /// Parseval scale: converts Σ|X[k]|² over a band → band RMS² in time-domain
    /// units. Equals 2 / (N · Σ hann²). The 2 accounts for the one-sided real
    /// spectrum; Σ hann² is the window's energy correction.
    parseval_band_scale: f32,
```

- [ ] **Step 6: Initialize the new fields in `Dsp::new`**

Inside `Dsp::new`, find this line:

```rust
        let smoothing_alpha = 1.0 - (-dt / SMOOTHING_TAU_SECS).exp();
```

Add immediately after it:

```rust
        let low_band_bin_end = bin_for_hz(LOW_BAND_HZ_MAX, sample_rate, window_size);
        let mid_band_bin_end = bin_for_hz(MID_BAND_HZ_MAX, sample_rate, window_size);
        let hann_energy: f32 = hann.iter().map(|h| h * h).sum();
        let parseval_band_scale = 2.0 / (window_size as f32 * hann_energy);
```

In the `Dsp { ... }` struct literal at the end of `new()`, add the new fields after `smoothing_alpha`:

```rust
            smoothing_alpha,
            low_band_bin_end,
            mid_band_bin_end,
            parseval_band_scale,
```

- [ ] **Step 7: Run tests — verify they pass**

Run: `cargo test -p dsp`
Expected: 20 tests pass (the 17 from the prior session plus the 3 new ones).

- [ ] **Step 8: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): bin-range derivation + Parseval scale plumbing"
```

---

## Task 2: Rust DSP — three-band instantaneous RMS + rolling histories

Compute each band's RMS from the FFT every hop, push to a 512-sample rolling history per band. Adds three public getters (`low_rms_history()`, `mid_rms_history()`, `high_rms_history()`).

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Add to the existing `mod tests` block:

```rust
    #[test]
    fn pure_low_band_sine_lands_in_low() {
        // Bin-aligned: 4 × (48000/2048) = 93.75 Hz, in the low band (bins 1..=6).
        // Hann main lobe is 4 bins wide; bin 4 ± 2 = bins 2..6, all in low.
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let sr = 48000.0_f32;
        let freq = 93.75_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let low = *dsp.low_rms_history().last().unwrap();
        let mid = *dsp.mid_rms_history().last().unwrap();
        let high = *dsp.high_rms_history().last().unwrap();
        assert!((low - 0.7071).abs() < 0.05, "low {} should be ≈ 0.707", low);
        assert!(mid < 0.05, "mid {} should be near zero", mid);
        assert!(high < 0.05, "high {} should be near zero", high);
    }

    #[test]
    fn pure_mid_band_sine_lands_in_mid() {
        // Bin-aligned: 30 × (48000/2048) = 703.125 Hz, in the mid band (bins 7..=64).
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let sr = 48000.0_f32;
        let freq = 703.125_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let low = *dsp.low_rms_history().last().unwrap();
        let mid = *dsp.mid_rms_history().last().unwrap();
        let high = *dsp.high_rms_history().last().unwrap();
        assert!((mid - 0.7071).abs() < 0.05, "mid {} should be ≈ 0.707", mid);
        assert!(low < 0.05, "low {} should be near zero", low);
        assert!(high < 0.05, "high {} should be near zero", high);
    }

    #[test]
    fn pure_high_band_sine_lands_in_high() {
        // Bin-aligned: 100 × (48000/2048) = 2343.75 Hz, in the high band (bins 65..=1023).
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let sr = 48000.0_f32;
        let freq = 2343.75_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let low = *dsp.low_rms_history().last().unwrap();
        let mid = *dsp.mid_rms_history().last().unwrap();
        let high = *dsp.high_rms_history().last().unwrap();
        assert!((high - 0.7071).abs() < 0.05, "high {} should be ≈ 0.707", high);
        assert!(low < 0.05, "low {} should be near zero", low);
        assert!(mid < 0.05, "mid {} should be near zero", mid);
    }

    #[test]
    fn parseval_consistency_across_bands() {
        // Multi-tone: one bin-aligned sine in each band, with different amplitudes.
        // Expected: sqrt(low² + mid² + high²) ≈ time-domain full RMS within ~5%
        // (slack for the stationarity approximation in the Parseval derivation).
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let sr = 48000.0_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| {
                let t = i as f32 / sr;
                let two_pi = 2.0 * std::f32::consts::PI;
                1.0  * (two_pi * 93.75   * t).sin()  // low,  amp 1.0
                + 0.5 * (two_pi * 703.125 * t).sin()  // mid,  amp 0.5
                + 0.25 * (two_pi * 2343.75 * t).sin() // high, amp 0.25
            })
            .collect();
        dsp.process(&signal);
        let low = *dsp.low_rms_history().last().unwrap();
        let mid = *dsp.mid_rms_history().last().unwrap();
        let high = *dsp.high_rms_history().last().unwrap();
        let full = *dsp.rms_history().last().unwrap();
        let summed = (low * low + mid * mid + high * high).sqrt();
        let rel_err = (summed - full).abs() / full;
        assert!(
            rel_err < 0.05,
            "Parseval mismatch: summed={}, full={}, rel_err={}",
            summed,
            full,
            rel_err
        );
    }

    #[test]
    fn band_rms_silence_is_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        dsp.process(&vec![0.0_f32; 2048]);
        assert_eq!(*dsp.low_rms_history().last().unwrap(), 0.0);
        assert_eq!(*dsp.mid_rms_history().last().unwrap(), 0.0);
        assert_eq!(*dsp.high_rms_history().last().unwrap(), 0.0);
    }

    #[test]
    fn low_rms_history_shifts_oldest_out() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        // Process two distinct loud signals; the second's value should land at the
        // end, the first's value second-from-end. We don't need exact values —
        // just that newer-than-old ordering holds and the oldest entry was shifted.
        let loud_low: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 93.75 * (i as f32 / 48000.0)).sin())
            .collect();
        let silent = vec![0.0_f32; 2048];
        dsp.process(&loud_low);   // pushes a non-zero into history
        dsp.process(&silent);     // pushes a zero
        let h = dsp.low_rms_history();
        let n = h.len();
        // Newest entry (silent) is 0; second-newest (loud) is non-zero.
        assert_eq!(h[n - 1], 0.0, "newest should be silent");
        assert!(h[n - 2] > 0.5, "second-newest should be the loud sine, got {}", h[n - 2]);
    }
```

- [ ] **Step 2: Run tests — verify compile failure**

Run: `cargo test -p dsp`
Expected: compile errors — `no method named 'low_rms_history'`, `no method named 'mid_rms_history'`, `no method named 'high_rms_history'` on `Dsp`.

- [ ] **Step 3: Add the rolling-history fields to `Dsp`**

In the `Dsp` struct definition, add these fields at the end (after `parseval_band_scale`):

```rust
    low_rms_history:  Vec<f32>,
    mid_rms_history:  Vec<f32>,
    high_rms_history: Vec<f32>,
```

- [ ] **Step 4: Initialize the new history vectors in `Dsp::new`**

In the `Dsp { ... }` struct literal at the end of `Dsp::new`, add after `parseval_band_scale`:

```rust
            low_rms_history:  vec![0.0; RMS_HISTORY_LEN],
            mid_rms_history:  vec![0.0; RMS_HISTORY_LEN],
            high_rms_history: vec![0.0; RMS_HISTORY_LEN],
```

- [ ] **Step 5: Add the per-band RMS computation to `process()`**

In `Dsp::process`, find the existing forward-FFT call:

```rust
        // Forward real FFT
        let _ = self
            .fft
            .process(&mut self.fft_buffer, &mut self.freq_buffer);
```

Add immediately after it:

```rust
        // --- Per-band RMS via Parseval-correct FFT-bin energy summation. ---
        // band_rms = sqrt(parseval_band_scale · Σ|X[k]|² over band).
        // Bands cover bins 1..=low_end, low_end+1..=mid_end, mid_end+1..=N/2-1.
        // (DC at bin 0 and Nyquist at bin N/2 are excluded by design.)
        let nyquist_bin = self.freq_buffer.len() - 1; // N/2
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
        let low_rms  = (low_e  * self.parseval_band_scale).sqrt();
        let mid_rms  = (mid_e  * self.parseval_band_scale).sqrt();
        let high_rms = (high_e * self.parseval_band_scale).sqrt();

        // Shift each band history left, append newest at the end (oldest at index 0).
        // Same pattern as the existing time-domain rms_history.
        for h in [
            (&mut self.low_rms_history, low_rms),
            (&mut self.mid_rms_history, mid_rms),
            (&mut self.high_rms_history, high_rms),
        ] {
            let (history, value) = h;
            history.copy_within(1.., 0);
            let last = history.len() - 1;
            history[last] = value;
        }
```

- [ ] **Step 6: Add the three public getters**

Inside the existing `#[wasm_bindgen] impl Dsp { ... }` block, after the existing `pub fn rms_history(&self) -> Vec<f32>` method, add:

```rust
    pub fn low_rms_history(&self) -> Vec<f32> {
        self.low_rms_history.clone()
    }

    pub fn mid_rms_history(&self) -> Vec<f32> {
        self.mid_rms_history.clone()
    }

    pub fn high_rms_history(&self) -> Vec<f32> {
        self.high_rms_history.clone()
    }
```

- [ ] **Step 7: Run tests — verify they pass**

Run: `cargo test -p dsp`
Expected: 26 tests pass (20 from Task 1 + 6 new).

- [ ] **Step 8: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): per-band RMS (low/mid/high) with rolling histories"
```

---

## Task 3: Rust DSP — low-band RMS-ACF + relocate full-band ACF after FFT

Add a detrended autocorrelation on the low-band RMS history (parallel to the existing full-band `rms_acf`), and tidy up by moving the existing `rms_acf` computation to after the FFT so all ACFs sit together.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Add to the existing `mod tests` block:

```rust
    #[test]
    fn low_rms_acf_has_correct_length() {
        let dsp = Dsp::new(2048, 48000.0, 1024);
        assert_eq!(dsp.low_rms_acf().len(), 256);
    }

    #[test]
    fn low_rms_acf_constant_input_is_zero() {
        // Fill low_rms_history with a constant non-zero band-RMS by feeding the
        // same loud bin-aligned low-frequency sine repeatedly. Detrended ACF on
        // a constant should be zero everywhere.
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let sr = 48000.0_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 93.75 * (i as f32 / sr)).sin())
            .collect();
        // RMS_HISTORY_LEN (512) calls fully fills the band history with a
        // constant value (the steady-state low_rms ≈ 0.707).
        for _ in 0..512 {
            dsp.process(&signal);
        }
        let acf = dsp.low_rms_acf();
        for &v in &acf {
            assert!(
                v.abs() < 1e-4,
                "expected near-zero detrended ACF for constant input, got {}",
                v
            );
        }
    }
```

- [ ] **Step 2: Run tests — verify compile failure**

Run: `cargo test -p dsp`
Expected: compile errors — `no method named 'low_rms_acf'` on `Dsp`.

- [ ] **Step 3: Add the low-band-ACF fields to `Dsp`**

In the `Dsp` struct definition, add at the end (after `high_rms_history`):

```rust
    /// Scratch: low_rms_history with its mean subtracted, used as input to
    /// `autocorrelate`. Without detrending, the average band level creates a
    /// DC bias that drowns out tempo peaks (same rationale as `rms_detrended`
    /// for the full-band ACF).
    low_rms_detrended: Vec<f32>,
    /// Detrended autocorrelation of `low_rms_history`. Length = RMS_ACF_LEN.
    low_rms_acf: Vec<f32>,
```

- [ ] **Step 4: Initialize them in `Dsp::new`**

In the `Dsp { ... }` struct literal, add after the three new history fields:

```rust
            low_rms_detrended: vec![0.0; RMS_HISTORY_LEN],
            low_rms_acf:       vec![0.0; RMS_ACF_LEN],
```

- [ ] **Step 5: Move existing `rms_acf` computation + add low-band ACF**

In `Dsp::process`, find the existing block that computes `rms_acf`:

```rust
        autocorrelate(&self.waveform, &mut self.buffer_acf);

        // RMS-envelope ACF: detrend (subtract mean) then autocorrelate.
        // Without detrending, average loudness creates a DC bias that
        // drowns out tempo peaks.
        let mean = self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32;
        for (dst, src) in self.rms_detrended.iter_mut().zip(self.rms_history.iter()) {
            *dst = src - mean;
        }
        autocorrelate(&self.rms_detrended, &mut self.rms_acf);
```

Replace it with the `buffer_acf` call only (the rms_acf block moves later):

```rust
        autocorrelate(&self.waveform, &mut self.buffer_acf);
```

Then, in the same `process()` body, find the per-band loop you added in Task 2 (ends with the `for h in [...] { ... }` block that updates the band histories). Add **after** that loop, but **before** the existing spectrum dB-normalization loop:

```rust
        // RMS-envelope ACFs: detrend (subtract mean) then autocorrelate.
        // Computed here, after the FFT and band-RMS updates, so all
        // ACF computations sit together. Full-band ACF moved here from
        // its old pre-FFT location; behavior is unchanged.
        let full_mean = self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32;
        for (dst, src) in self.rms_detrended.iter_mut().zip(self.rms_history.iter()) {
            *dst = src - full_mean;
        }
        autocorrelate(&self.rms_detrended, &mut self.rms_acf);

        let low_mean = self.low_rms_history.iter().sum::<f32>() / self.low_rms_history.len() as f32;
        for (dst, src) in self.low_rms_detrended.iter_mut().zip(self.low_rms_history.iter()) {
            *dst = src - low_mean;
        }
        autocorrelate(&self.low_rms_detrended, &mut self.low_rms_acf);
```

- [ ] **Step 6: Add the public getter**

In the `#[wasm_bindgen] impl Dsp { ... }` block, after the existing `pub fn rms_acf` getter, add:

```rust
    pub fn low_rms_acf(&self) -> Vec<f32> {
        self.low_rms_acf.clone()
    }
```

- [ ] **Step 7: Run tests — verify they pass**

Run: `cargo test -p dsp`
Expected: 28 tests pass (26 from Task 2 + 2 new). Crucially, the existing `rms_acf_constant_input_is_zero` and `acf_of_silence_is_zero` tests must still pass — they're the regression coverage proving the rms_acf relocation didn't change behavior.

- [ ] **Step 8: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): low-band RMS-ACF + relocate full-band ACF after FFT"
```

---

## Task 4: Worklet — post the four new features

Rebuild the WASM bindings (the new getters need to be visible to TS) and extend the worklet's `postMessage` payload + transferable list with the four new arrays.

**Files:**
- Modify: `src/audio/dsp-worklet.ts`

- [ ] **Step 1: Rebuild WASM**

Run: `npm run wasm`
Expected: succeeds; `src/wasm-pkg/dsp.js` and `src/wasm-pkg/dsp_bg.wasm` regenerated. The TS bindings now include `low_rms_history()`, `mid_rms_history()`, `high_rms_history()`, and `low_rms_acf()`.

- [ ] **Step 2: Extend the worklet's post-FFT block**

In `src/audio/dsp-worklet.ts`, find the existing block inside the `while (this.hopCounter >= HOP_SIZE)` loop:

```ts
      this.dsp.process(this.window);
      const wf = new Float32Array(this.dsp.waveform());
      const sp = new Float32Array(this.dsp.spectrum());
      const rms = new Float32Array(this.dsp.rms_history());
      const ba = new Float32Array(this.dsp.buffer_acf());
      const ra = new Float32Array(this.dsp.rms_acf());
      this.port.postMessage(
        { type: "features", waveform: wf, spectrum: sp, rms, bufferAcf: ba, rmsAcf: ra },
        [wf.buffer, sp.buffer, rms.buffer, ba.buffer, ra.buffer],
      );
```

Replace with:

```ts
      this.dsp.process(this.window);
      const wf = new Float32Array(this.dsp.waveform());
      const sp = new Float32Array(this.dsp.spectrum());
      const rms = new Float32Array(this.dsp.rms_history());
      const ba = new Float32Array(this.dsp.buffer_acf());
      const ra = new Float32Array(this.dsp.rms_acf());
      const rmsLow = new Float32Array(this.dsp.low_rms_history());
      const rmsMid = new Float32Array(this.dsp.mid_rms_history());
      const rmsHigh = new Float32Array(this.dsp.high_rms_history());
      const rmsAcfLow = new Float32Array(this.dsp.low_rms_acf());
      this.port.postMessage(
        {
          type: "features",
          waveform: wf,
          spectrum: sp,
          rms,
          bufferAcf: ba,
          rmsAcf: ra,
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
          rmsLow.buffer,
          rmsMid.buffer,
          rmsHigh.buffer,
          rmsAcfLow.buffer,
        ],
      );
```

- [ ] **Step 3: TypeScript + build check**

Run: `npx tsc --noEmit`
Run: `npm run build`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/audio/dsp-worklet.ts
git commit -m "feat(audio): worklet posts per-band RMS + low-band ACF"
```

---

## Task 5: App.ts — store primings, port dispatch, four new LineRenderers

Wire the four new features through to four new `LineRenderer` instances. Bands overlay on the existing white-rms strip with R/G/B colors; low-band ACF overlays on the existing rms_acf strip in red. Existing white and pink lines stay drawn on top via scene-add ordering.

**Files:**
- Modify: `src/App.ts`

- [ ] **Step 1: Add new fields to the `App` class**

In `src/App.ts`, find the existing private-field block at the top of `class App`:

```ts
  private rig!: CameraRig;
  private waveformLine!: LineRenderer;
  private spectrumLine!: LineRenderer;
  private rmsLine!: LineRenderer;
  private bufferAcfLine!: LineRenderer;
  private rmsAcfLine!: LineRenderer;
```

Add four new field declarations right after `rmsAcfLine`:

```ts
  private lowRmsLine!: LineRenderer;
  private midRmsLine!: LineRenderer;
  private highRmsLine!: LineRenderer;
  private lowRmsAcfLine!: LineRenderer;
```

- [ ] **Step 2: Prime four new FeatureStore keys**

Find the existing primings block in `App.start`:

```ts
    this.store.set("waveform", new Float32Array(2048));
    this.store.set("spectrum", new Float32Array(1024));
    this.store.set("rms", new Float32Array(512));
    this.store.set("bufferAcf", new Float32Array(1024));
    this.store.set("rmsAcf", new Float32Array(256));
```

Replace with:

```ts
    this.store.set("waveform", new Float32Array(2048));
    this.store.set("spectrum", new Float32Array(1024));
    this.store.set("rms", new Float32Array(512));
    this.store.set("bufferAcf", new Float32Array(1024));
    this.store.set("rmsAcf", new Float32Array(256));
    this.store.set("rmsLow", new Float32Array(512));
    this.store.set("rmsMid", new Float32Array(512));
    this.store.set("rmsHigh", new Float32Array(512));
    this.store.set("rmsAcfLow", new Float32Array(256));
```

- [ ] **Step 3: Construct the three band-RMS LineRenderers BEFORE the existing white rmsLine**

In `App.start`, find the existing `rmsLine` construction block (the one with `linearLayout(-0.5, 0.4)` and color `0xffffff`):

```ts
    this.rmsLine = new LineRenderer({
      source: () => this.store.get("rms"),
      layout: linearLayout(-0.5, 0.4),
      color: 0xffffff,
    });
    scene.add(this.rmsLine.object3d);
```

Insert the three new band-RMS LineRenderers **immediately before** this block, so they get added to the scene first and the white rms line draws on top of them at intersections:

```ts
    this.lowRmsLine = new LineRenderer({
      source: () => this.store.get("rmsLow"),
      layout: linearLayout(-0.5, 0.4),
      color: 0xff4444,
    });
    scene.add(this.lowRmsLine.object3d);

    this.midRmsLine = new LineRenderer({
      source: () => this.store.get("rmsMid"),
      layout: linearLayout(-0.5, 0.4),
      color: 0x44ff44,
    });
    scene.add(this.midRmsLine.object3d);

    this.highRmsLine = new LineRenderer({
      source: () => this.store.get("rmsHigh"),
      layout: linearLayout(-0.5, 0.4),
      color: 0x4488ff,
    });
    scene.add(this.highRmsLine.object3d);

    this.rmsLine = new LineRenderer({
      source: () => this.store.get("rms"),
      layout: linearLayout(-0.5, 0.4),
      color: 0xffffff,
    });
    scene.add(this.rmsLine.object3d);
```

(That is: replace the original 5-line rmsLine block with the 21-line block above, which contains the three new bands followed by the unchanged white rmsLine.)

- [ ] **Step 4: Construct the low-band ACF LineRenderer BEFORE the existing rmsAcfLine**

Find the existing `rmsAcfLine` construction block:

```ts
    this.rmsAcfLine = new LineRenderer({
      source: () => this.store.get("rmsAcf"),
      layout: linearLayout(-1.0, 0.4),
      color: 0xff99cc,
    });
    scene.add(this.rmsAcfLine.object3d);
```

Insert the low-band ACF LineRenderer **immediately before** this block:

```ts
    this.lowRmsAcfLine = new LineRenderer({
      source: () => this.store.get("rmsAcfLow"),
      layout: linearLayout(-1.0, 0.4),
      color: 0xff4444,
    });
    scene.add(this.lowRmsAcfLine.object3d);

    this.rmsAcfLine = new LineRenderer({
      source: () => this.store.get("rmsAcf"),
      layout: linearLayout(-1.0, 0.4),
      color: 0xff99cc,
    });
    scene.add(this.rmsAcfLine.object3d);
```

- [ ] **Step 5: Extend the port-message handler**

Find the existing `node.port.onmessage` block:

```ts
    node.port.onmessage = (e) => {
      const msg = e.data as {
        type: string;
        waveform?: Float32Array;
        spectrum?: Float32Array;
        rms?: Float32Array;
        bufferAcf?: Float32Array;
        rmsAcf?: Float32Array;
      };
      if (msg.type !== "features") return;
      if (msg.waveform) this.store.set("waveform", msg.waveform);
      if (msg.spectrum) this.store.set("spectrum", msg.spectrum);
      if (msg.rms) this.store.set("rms", msg.rms);
      if (msg.bufferAcf) this.store.set("bufferAcf", msg.bufferAcf);
      if (msg.rmsAcf) this.store.set("rmsAcf", msg.rmsAcf);
    };
```

Replace with:

```ts
    node.port.onmessage = (e) => {
      const msg = e.data as {
        type: string;
        waveform?: Float32Array;
        spectrum?: Float32Array;
        rms?: Float32Array;
        bufferAcf?: Float32Array;
        rmsAcf?: Float32Array;
        rmsLow?: Float32Array;
        rmsMid?: Float32Array;
        rmsHigh?: Float32Array;
        rmsAcfLow?: Float32Array;
      };
      if (msg.type !== "features") return;
      if (msg.waveform) this.store.set("waveform", msg.waveform);
      if (msg.spectrum) this.store.set("spectrum", msg.spectrum);
      if (msg.rms) this.store.set("rms", msg.rms);
      if (msg.bufferAcf) this.store.set("bufferAcf", msg.bufferAcf);
      if (msg.rmsAcf) this.store.set("rmsAcf", msg.rmsAcf);
      if (msg.rmsLow) this.store.set("rmsLow", msg.rmsLow);
      if (msg.rmsMid) this.store.set("rmsMid", msg.rmsMid);
      if (msg.rmsHigh) this.store.set("rmsHigh", msg.rmsHigh);
      if (msg.rmsAcfLow) this.store.set("rmsAcfLow", msg.rmsAcfLow);
    };
```

- [ ] **Step 6: Add the four new `update()` calls in the render loop**

Find the existing render loop body:

```ts
    const loop = (now: number) => {
      this.fps.begin();
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;
      this.rig.update(dt);
      this.waveformLine.update();
      this.bufferAcfLine.update();
      this.spectrumLine.update();
      this.rmsLine.update();
      this.rmsAcfLine.update();
      renderer.render(scene, camera);
      this.fps.end();
      requestAnimationFrame(loop);
    };
```

Replace with:

```ts
    const loop = (now: number) => {
      this.fps.begin();
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;
      this.rig.update(dt);
      this.waveformLine.update();
      this.bufferAcfLine.update();
      this.spectrumLine.update();
      this.lowRmsLine.update();
      this.midRmsLine.update();
      this.highRmsLine.update();
      this.rmsLine.update();
      this.lowRmsAcfLine.update();
      this.rmsAcfLine.update();
      renderer.render(scene, camera);
      this.fps.end();
      requestAnimationFrame(loop);
    };
```

- [ ] **Step 7: TypeScript + build check**

Run: `npx tsc --noEmit`
Run: `npm run build`
Expected: clean.

- [ ] **Step 8: Manual verification — test source (fastest path)**

Run: `npm run dev`. Click "Test 440Hz" (no permission prompts).

440 Hz lands in the mid band (bin 19 ≈ 445 Hz). Expected at the bottom RMS strip:

- The **green** mid-band line should be near the same height as the **white** full-rms line — both showing the energy of the 440 Hz tone.
- The **red** low-band line should sit near zero (no sub-bass content in a pure tone).
- The **blue** high-band line should also sit near zero.
- The four lines should not flicker or jitter beyond what the white line does today.

Press `4` to jump to the rms preset for a closer look. Press `6` for the rms-acf preset — the new red low-band ACF line should appear overlaid with the existing pink full-band ACF line. With a steady 440 Hz tone, both ACFs should be nearly flat at zero (no rhythmic structure).

- [ ] **Step 9: Manual verification — mic input with music**

Reload, click "Mic", grant permission, play music with a clear kick drum (any modern dance/hip-hop track works). Expected:

- The **red** low-band line spikes on every kick.
- The **green** mid-band line tracks vocals / snare.
- The **blue** high-band line tracks hats / cymbals.
- The **white** full-rms line draws on top at line crossings (visible reference).
- Press `6` for rms-acf: the red low-band ACF line should show clearer rhythmic peaks than the pink full-band one (kick is a more concentrated tempo cue than full-spectrum loudness).

Stop the dev server.

- [ ] **Step 10: Commit**

```bash
git add src/App.ts
git commit -m "feat(app): render multi-band RMS overlay + low-band ACF overlay"
```

---

## Task 6: Final acceptance

**Files:** none modified

- [ ] **Step 1: All Rust tests**

Run: `cargo test -p dsp`
Expected: 28 passed (17 pre-existing + 3 from Task 1 + 6 from Task 2 + 2 from Task 3).

- [ ] **Step 2: All JS tests**

Run: `npm test`
Expected: 20 passed (no JS-side coverage of the new features; this is a regression check that existing tests still pass).

- [ ] **Step 3: Production build**

Run: `npm run build`
Expected: succeeds; bundle size grows by less than 1 KB (all new code paths are tiny).

- [ ] **Step 4: Smoke-test the production build**

Run: `npm run preview`. Open the printed URL. Repeat the manual checks from Task 5 Steps 8 and 9 against the production build to confirm parity with dev mode.

Stop the preview server.
