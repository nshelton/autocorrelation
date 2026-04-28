# FFT 25% Overlap (HOP_SIZE=512) + Time-Based Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop the FFT hop from 1024 to 512 samples (25% overlap → ~94 Hz spectrum/RMS update rate, ~halved head-of-buffer-to-screen latency). Decouple smoothing dynamics from hop size by replacing the per-process-call alpha constant with a time-based EMA computed from a fixed time constant and the actual hop interval — so future hop changes don't require retuning by hand.

**Architecture:** Add a `smoothing_alpha: f32` field to `Dsp`, computed in `Dsp::new` from `SMOOTHING_TAU_SECS` (target ≈ 95.6 ms — chosen to preserve the legacy alpha = 0.2 behavior at sr=48000, hop=1024). Constructor takes additional `(sample_rate: f32, hop_size: usize)` arguments; alpha = `1 - (-dt/tau).exp()` where `dt = hop_size / sample_rate`. The worklet passes the AudioWorklet global `sampleRate` and `HOP_SIZE` to the constructor and changes `HOP_SIZE` from 1024 to 512.

**Tech Stack:** Existing — Rust + `realfft`, TypeScript AudioWorklet, `wasm-pack` toolchain.

---

## Task 1: Time-based smoothing constant in Rust DSP

Replace the hard-coded `SMOOTHING_ALPHA: f32 = 0.2` with a time-based EMA computed in the constructor from `SMOOTHING_TAU_SECS` and the actual `dt = hop_size / sample_rate`. Constructor signature changes — every existing call site (tests + worklet) must update.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing tests for the new alpha behavior**

Add three new tests to the existing `mod tests` block at the bottom of `crates/dsp/src/lib.rs`:

```rust
    #[test]
    fn smoothing_alpha_matches_time_constant_formula() {
        // alpha = 1 - exp(-dt/tau) where dt = hop_size / sample_rate
        let dsp = Dsp::new(2048, 48000.0, 1024);
        let dt = 1024.0_f32 / 48000.0;
        let expected = 1.0 - (-dt / SMOOTHING_TAU_SECS).exp();
        assert!(
            (dsp.smoothing_alpha - expected).abs() < 1e-6,
            "alpha {} != expected {}",
            dsp.smoothing_alpha,
            expected
        );
    }

    #[test]
    fn smoothing_alpha_at_legacy_settings_is_approximately_0_2() {
        // SMOOTHING_TAU_SECS is chosen so that at sr=48000, hop=1024
        // alpha ≈ 0.2 — i.e., the legacy hard-coded value is preserved.
        let dsp = Dsp::new(2048, 48000.0, 1024);
        assert!(
            (dsp.smoothing_alpha - 0.2).abs() < 0.005,
            "expected alpha ≈ 0.2 at legacy settings, got {}",
            dsp.smoothing_alpha
        );
    }

    #[test]
    fn smoothing_alpha_shrinks_at_smaller_hop() {
        // Halving hop ≈ halves alpha (small-dt regime: 1 - exp(-x) ≈ x).
        // Wall-clock dynamics stay the same; per-call coefficient changes.
        let large = Dsp::new(2048, 48000.0, 1024);
        let small = Dsp::new(2048, 48000.0, 512);
        assert!(
            small.smoothing_alpha < large.smoothing_alpha,
            "small {} should be < large {}",
            small.smoothing_alpha,
            large.smoothing_alpha
        );
        let ratio = small.smoothing_alpha / large.smoothing_alpha;
        assert!(
            (0.45..=0.55).contains(&ratio),
            "expected ratio ≈ 0.5, got {}",
            ratio
        );
    }
```

These reference `SMOOTHING_TAU_SECS` and the `smoothing_alpha` field — both intentionally not yet defined. The new constructor signature also doesn't compile yet.

- [ ] **Step 2: Run tests to verify compile failure**

Run: `cargo test -p dsp`
Expected: compilation errors — `cannot find value 'SMOOTHING_TAU_SECS'`, `no field 'smoothing_alpha'`, `Dsp::new` argument count mismatch on the new tests.

- [ ] **Step 3: Update existing tests to use the new constructor signature**

Each existing test that calls `Dsp::new(N)` must be updated to `Dsp::new(N, 48000.0, hop)` where `hop` is any reasonable value (these tests do not exercise smoothing rate; they just need to satisfy the new signature). Replace each call site in `crates/dsp/src/lib.rs`'s `mod tests`:

| Test name | Old call | New call |
|---|---|---|
| `process_then_waveform_returns_input` | `Dsp::new(8)` | `Dsp::new(8, 48000.0, 4)` |
| `spectrum_has_window_size_div_2_bins` | `Dsp::new(2048)` | `Dsp::new(2048, 48000.0, 1024)` |
| `silent_input_yields_low_spectrum` | `Dsp::new(2048)` | `Dsp::new(2048, 48000.0, 1024)` |
| `rms_of_unit_amplitude_constant_is_one` | `Dsp::new(8)` | `Dsp::new(8, 48000.0, 4)` |
| `rms_of_silence_is_zero` | `Dsp::new(8)` | `Dsp::new(8, 48000.0, 4)` |
| `rms_history_shifts_oldest_out` | `Dsp::new(4)` | `Dsp::new(4, 48000.0, 4)` |
| `loud_sine_produces_a_peak` | `Dsp::new(2048)` | `Dsp::new(2048, 48000.0, 1024)` |

- [ ] **Step 4: Replace the constants block at the top of `crates/dsp/src/lib.rs`**

Find:

```rust
const SMOOTHING_ALPHA: f32 = 0.2;
const DB_FLOOR: f32 = -100.0;
const RMS_HISTORY_LEN: usize = 512;
```

Replace with:

```rust
/// Spectrum smoothing time constant in seconds. Chosen to preserve the
/// legacy alpha ≈ 0.2 behavior at sr=48000, hop=1024:
///   alpha = 1 - exp(-dt/tau), dt = 1024/48000 = 21.33 ms
///   0.2 ≈ 1 - exp(-21.33ms / 95.6ms)
const SMOOTHING_TAU_SECS: f32 = 0.0956;
const DB_FLOOR: f32 = -100.0;
const RMS_HISTORY_LEN: usize = 512;
```

- [ ] **Step 5: Add `smoothing_alpha` to the `Dsp` struct**

Find the struct definition and add the field at the end:

```rust
#[wasm_bindgen]
pub struct Dsp {
    waveform: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_buffer: Vec<f32>,
    freq_buffer: Vec<Complex<f32>>,
    spectrum: Vec<f32>,
    hann: Vec<f32>,
    /// Magnitude scale factor that converts raw FFT bin magnitude to
    /// amplitude-equivalent units (so a unit-amplitude sine peaks at ~1.0).
    /// Equals 2/sum(hann) — the 2 accounts for the one-sided real spectrum,
    /// and sum(hann) ≈ N/2 corrects for window attenuation.
    mag_scale: f32,
    rms_history: Vec<f32>,
    /// Per-process-call EMA coefficient. Computed from `SMOOTHING_TAU_SECS`
    /// and the wall-clock dt between hops (`hop_size / sample_rate`), so
    /// changing `hop_size` does NOT change perceived smoothing dynamics.
    smoothing_alpha: f32,
}
```

- [ ] **Step 6: Replace `Dsp::new` with the new signature and alpha computation**

Find the existing constructor:

```rust
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize) -> Dsp {
```

Replace the entire `pub fn new(...)` body with:

```rust
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize, sample_rate: f32, hop_size: usize) -> Dsp {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let freq_buffer = fft.make_output_vec();
        let spectrum = vec![0.0; freq_buffer.len() - 1]; // drop DC
        let hann: Vec<f32> = (0..window_size)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos()
            })
            .collect();
        let mag_scale = 2.0 / hann.iter().sum::<f32>();
        let dt = hop_size as f32 / sample_rate;
        let smoothing_alpha = 1.0 - (-dt / SMOOTHING_TAU_SECS).exp();
        Dsp {
            waveform: vec![0.0; window_size],
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            spectrum,
            hann,
            mag_scale,
            rms_history: vec![0.0; RMS_HISTORY_LEN],
            smoothing_alpha,
        }
    }
```

- [ ] **Step 7: Use `self.smoothing_alpha` in `process()` instead of the const**

Find this line in `Dsp::process`:

```rust
            self.spectrum[out_i] =
                SMOOTHING_ALPHA * normalized + (1.0 - SMOOTHING_ALPHA) * self.spectrum[out_i];
```

Replace with:

```rust
            self.spectrum[out_i] = self.smoothing_alpha * normalized
                + (1.0 - self.smoothing_alpha) * self.spectrum[out_i];
```

- [ ] **Step 8: Run all Rust tests**

Run: `cargo test -p dsp`
Expected: 10 tests pass — the original 7 plus the 3 new alpha tests.

- [ ] **Step 9: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "refactor(dsp): time-based EMA smoothing decoupled from hop size"
```

---

## Task 2: Drop hop to 512 samples in worklet

Switch `HOP_SIZE` to 512 (25% overlap → ~94 Hz update rate) and pass the AudioWorklet's `sampleRate` global plus `HOP_SIZE` to the new `Dsp` constructor.

**Files:**
- Modify: `src/audio/dsp-worklet.ts`

- [ ] **Step 1: Rebuild WASM with the new constructor signature**

Run: `npm run wasm`
Expected: succeeds. `src/wasm-pkg/dsp.js` and `src/wasm-pkg/dsp_bg.wasm` regenerated. The TypeScript binding for `Dsp.constructor` now requires three arguments.

- [ ] **Step 2: Drop `HOP_SIZE` to 512 in `src/audio/dsp-worklet.ts`**

Find:

```ts
const WINDOW_SIZE = 2048;
const HOP_SIZE = 1024;
```

Replace with:

```ts
const WINDOW_SIZE = 2048;
const HOP_SIZE = 512;
```

- [ ] **Step 3: Pass sample rate and hop size to the `Dsp` constructor**

Find the `boot` method:

```ts
  private async boot(wasmModule: WebAssembly.Module) {
    await init({ module_or_path: wasmModule });
    this.dsp = new Dsp(WINDOW_SIZE);
    this.ready = true;
  }
```

Replace with:

```ts
  private async boot(wasmModule: WebAssembly.Module) {
    await init({ module_or_path: wasmModule });
    // `sampleRate` is a global in AudioWorkletGlobalScope (typed by
    // @types/audioworklet, referenced at the top of this file).
    this.dsp = new Dsp(WINDOW_SIZE, sampleRate, HOP_SIZE);
    this.ready = true;
  }
```

- [ ] **Step 4: TypeScript + build check**

Run: `npx tsc --noEmit`
Run: `npm run build`
Expected: clean.

- [ ] **Step 5: Manual verification — test source (fastest path)**

Run: `npm run dev`. Click "Test 440Hz" (no permission prompts).

Expected:
- Spectrum line shows a single clean peak around 440 Hz, visually as stable as before.
- The peak should not appear noticeably jitterier or "buzzy" than the legacy hop=1024 build. If it does, smoothing is too aggressive at the new hop and `SMOOTHING_TAU_SECS` should be raised (e.g., to 0.15 or 0.2). Re-run `cargo test -p dsp` after any change — the legacy-equivalence test is fine to break here, since the new tau is the new source of truth.
- FPS overlay still ~60.
- Console: existing `[audio]` diagnostic lines appear; no new errors or warnings.

- [ ] **Step 6: Manual verification — mic source**

Reload the page. Click "Mic" and speak. The waveform, spectrum, and RMS lines should all behave identically to before, just with smoother / more responsive updates (RMS in particular should visibly track transients faster — newest sample on the right edge updates ~2× per render frame instead of every other render frame).

Stop the dev server.

- [ ] **Step 7: Commit**

```bash
git add src/audio/dsp-worklet.ts
git commit -m "perf(dsp): 25% overlap (HOP_SIZE=512) for ~94Hz spectrum/RMS updates"
```

---

## Task 3: Final acceptance + roadmap cleanup

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 1: All Rust tests**

Run: `cargo test -p dsp`
Expected: 10 passed.

- [ ] **Step 2: All JS tests**

Run: `npm test`
Expected: 19 passed (no JS-side coverage of the hop change, but a regression check that nothing else broke).

- [ ] **Step 3: Production build**

Run: `npm run build`
Expected: succeeds; bundle size unchanged within noise.

- [ ] **Step 4: Roadmap cleanup**

Edit `ROADMAP.md`. In the `Next → Performance` section, remove this bullet:

```
- Drop FFT hop to 25% overlap (HOP_SIZE=512) for ~94 Hz update rate and ~halved head-of-buffer-to-screen latency. Requires retuning `SMOOTHING_ALPHA` (currently 0.2, per-process-call) — preferably convert to a time-based EMA so the constant stops being hop-coupled.
```

The other Performance bullet (Wiener–Khinchin migration) stays.

- [ ] **Step 5: Commit roadmap cleanup**

```bash
git add ROADMAP.md
git commit -m "docs(roadmap): mark 25% hop + time-based smoothing as shipped"
```
