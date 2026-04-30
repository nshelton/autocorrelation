# Beat tracker rewrite: Percival & Tzanetakis 2014

Date: 2026-04-29

## Goal

Replace the current peak-driven `BeatTracker` (with its `lag/k` folding update rule) with a streaming adaptation of Percival & Tzanetakis 2014, "Streamlined Tempo Estimation Based on Autocorrelation". The current tracker has an octave-ambiguity bug: every divisor of the true beat period in `[period_min, period_max]` is a stable fixed point, so the tracker frequently locks at a sub-multiple (e.g. the 16th-note grid) of the actual tempo. The paper's pulse-train scoring step explicitly disambiguates octaves by evaluating beat templates against the onset signal, which is the structural fix this spec adopts.

## Scope

In:

- New onset signal (OSS) derived from `rms_history` as a half-wave-rectified time difference. Spectral-flux-based OSS deferred — the OSS pipeline is structured so the source can be swapped later without touching downstream code.
- Generalized autocorrelation (zero-pad → FFT → `|X|^0.5` → IFFT) of the onset signal, replacing the existing direct-time-domain ACF on the beat path.
- Harmonic enhancement: `A_e[τ] = A[τ] + A[2τ] + A[4τ]`.
- Top-10 peak picking on the enhanced ACF over the lag range corresponding to the configured BPM bounds.
- Pulse-train scoring per candidate: cross-correlate three weighted impulse trains (the paper's `Φ₁ + Φ₂/2 + Φ₁.₅/2`) against the onset frame at all integer phases; combine `max-φ corr` and `var-φ corr` into a per-candidate score, pick the winner.
- Streaming TEA (Tempo Estimate Accumulator): EMA-decayed Gaussian-smeared lag accumulator. Replaces the current `rms_acf_accum`.
- Output the smoothed period, a coherent phase (re-scored at the smoothed period), BPM, and confidence each frame. Drive existing 4-cycle saw-wave `beat_pulses` from the new (period, phase).
- Visualization: keep one ACF strip, overlay `onset_acf` (cyan), `onset_acf_enhanced` (pink), `tea` (yellow); revive `PeakMarkers` reading the new `candidates` buffer; add a new `onset` line strip.

Out (deferred):

- Spectral-flux OSS (Section II-A of the paper). Hook left in place; replacement is local to the OSS computation step.
- Octave decider (paper Section II-C step 4). Output `τ_smoothed` directly.
- `beats_per_measure` detection. `beat_state[2..4]` outputs `NaN`. Existing measure-detection code is removed; a future spec can reintroduce it once the period estimate is solid.
- Per-band onset paths (`low_rms_acf` and friends are dropped; per-band onset processing can be added later if useful).

## Architecture

```
audio ──► AudioWorklet
         ├─ FFT / spectrum   (unchanged)
         ├─ Multi-band RMS   (unchanged)
         └─ Beat path:
            rms_history  ──► onset = max(0, rms[t]-rms[t-1])
                              │
                              ▼
                          onset_history (sliding, length N)
                              │
                              ▼
                          generalized ACF  ──► onset_acf (length N/2)
                              │
                              ▼
                          harmonic enhance ──► onset_acf_enhanced
                              │
                              ▼
                          peak pick (top 10, [τ_min, τ_max])
                              │
                              ▼
                          pulse-train scoring  ──► (period_inst, phase_inst, score_inst)
                              │
                              ▼
                          TEA update + argmax  ──► tau_smoothed
                              │
                              ▼
                          phase scan @ tau_smoothed ──► phase_smoothed
                              │
                              ▼
                          beat_grid / beat_state / beat_pulses
                              │
                              ▼
                          worklet → main thread → DebugView
```

No new threading boundaries. Per-frame cost is dominated by the gen-ACF FFTs (size `2·rms_history_len` = 1024 by default — single forward + single inverse). All other steps are O(N) with small constants.

## Rust DSP changes (`crates/dsp/src/lib.rs`)

### Constants

```rust
const BEAT_TRACKER_MIN_BPM: f32 = 40.0;     // was 80.0
const BEAT_TRACKER_MAX_BPM: f32 = 220.0;    // was 190.0
const GEN_ACF_C: f32 = 0.5;                 // magnitude compression — paper-recommended
const HARMONIC_MULTIPLES: [usize; 2] = [2, 4];
const MAX_PEAKS: usize = 10;                // unchanged
const MIN_PEAK_SPACING: usize = 3;          // unchanged
const PULSE_N: usize = 4;                   // pulses per template
const TEA_GAUSSIAN_SIGMA: f32 = 5.0;        // lag samples — paper's σ
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;
```

Removed: `BEAT_TRACKER_INITIAL_BPM`, `BEAT_TRACKER_INITIAL_BEATS_PER_MEASURE`, `BEAT_TRACKER_SIGMA_LAG`, `BEAT_TRACKER_ALPHA_MAX`, `BEAT_TRACKER_CONF_SMOOTHING`, `BEAT_TRACKER_MEASURE_SWITCH_MARGIN`, `BEAT_TRACKER_MEASURE_CANDIDATES`, `BEAT_PHASE_STEP_HOPS`, `MIN_PEAK_LAG`, `ACCUM_TAU_DEFAULT_SECS`. `BEAT_PULSE_CYCLES` and `BEAT_PULSES_LEN` stay (still drive the saw outputs).

### `Dsp` struct — new and removed fields

Removed: `BeatTracker` struct entirely. `rms_acf`, `rms_acf_accum`, `accum_alpha`, `acf_peaks`, `peak_candidates`, `rms_detrended`, `low_rms_acf`, `low_rms_detrended` and their getters.

Added:
```rust
onset_history: Vec<f32>,           // length = rms_history.len()
prev_rms: f32,                     // remembered across calls for the diff

// Generalized ACF compute
gen_acf_fft_forward: Arc<dyn RealToComplex<f32>>,    // size 2N
gen_acf_fft_inverse: Arc<dyn ComplexToReal<f32>>,    // size 2N
gen_acf_time_buf: Vec<f32>,                          // length 2N
gen_acf_freq_buf: Vec<Complex<f32>>,                 // length N+1

onset_acf: Vec<f32>,               // length N/2
onset_acf_enhanced: Vec<f32>,      // length N/2

// Peak picking
candidates: Vec<f32>,              // length 3 * MAX_PEAKS = 30 — [lag, mag, sharpness] stride 3
peak_candidates: Vec<(usize, f32)>,// preallocated scratch (kept name from old impl)

// Pulse-train scoring scratch
pulse_x: [f32; MAX_PEAKS],         // max-φ corr per candidate
pulse_v: [f32; MAX_PEAKS],         // var-φ corr per candidate
pulse_phi: [f32; MAX_PEAKS],       // best phase per candidate
pulse_score: [f32; MAX_PEAKS],     // normalized X+V

// Streaming TEA
tea: Vec<f32>,                     // length N/2
tea_alpha: f32,                    // 1 - exp(-dt / τ_tea)

// Cached lag bounds (recomputed on construction; not exposed)
tau_min: usize,
tau_max: usize,
```

`beat_grid`, `beat_state`, `beat_pulses`, `beat_position` keep their current shapes and meanings. `update_beat_pulses` simplifies (no more "free-run when phase is NaN" branch; phase is always real when `score_inst > 0`, all-NaN otherwise).

### Per-`process()` algorithm

After the existing RMS computation that produces `rms` and updates `rms_history`:

#### 1. OSS update

```rust
let onset = (rms - self.prev_rms).max(0.0);
self.prev_rms = rms;
self.onset_history.copy_within(1.., 0);
*self.onset_history.last_mut().unwrap() = onset;
```

#### 2. Generalized ACF

```rust
let n = self.onset_history.len();
self.gen_acf_time_buf[..n].copy_from_slice(&self.onset_history);
self.gen_acf_time_buf[n..].fill(0.0);

self.gen_acf_fft_forward.process(&mut self.gen_acf_time_buf, &mut self.gen_acf_freq_buf)?;

for x in self.gen_acf_freq_buf.iter_mut() {
    let mag = (x.re * x.re + x.im * x.im).sqrt();
    *x = Complex::new(mag.powf(GEN_ACF_C), 0.0);
}

self.gen_acf_fft_inverse.process(&mut self.gen_acf_freq_buf, &mut self.gen_acf_time_buf)?;

let zero = self.gen_acf_time_buf[0].max(1e-12);
for i in 0..self.onset_acf.len() {
    self.onset_acf[i] = self.gen_acf_time_buf[i] / zero;
}
```

`onset_acf[0]` is the autocorrelation at zero lag — used as a normalizer so the displayed signal is in `[~0, 1]` regardless of overall onset magnitude. `1e-12` floor prevents NaN on silent input.

#### 3. Harmonic enhancement

```rust
for tau in 0..self.onset_acf_enhanced.len() {
    let mut sum = self.onset_acf[tau];
    for &mult in &HARMONIC_MULTIPLES {
        let idx = tau * mult;
        if idx < self.onset_acf.len() {
            sum += self.onset_acf[idx];
        }
    }
    self.onset_acf_enhanced[tau] = sum;
}
```

Integer indexing. Sub-bin precision lives in the next step's parabolic refinement.

#### 4. Peak picking

```rust
self.candidates.iter_mut().for_each(|v| *v = f32::NAN);

self.peak_candidates.clear();
for k in self.tau_min.max(1)..self.tau_max.min(self.onset_acf_enhanced.len() - 1) {
    let y = self.onset_acf_enhanced[k];
    if y > 0.0 && y > self.onset_acf_enhanced[k-1] && y > self.onset_acf_enhanced[k+1] {
        self.peak_candidates.push((k, y));
    }
}
self.peak_candidates.sort_unstable_by(|a, b|
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

let mut accepted: [u32; MAX_PEAKS] = [0; MAX_PEAKS];
let mut count = 0;
for &(k, _) in &self.peak_candidates {
    if count == MAX_PEAKS { break; }
    if accepted[..count].iter().all(|&j| (k as i32 - j as i32).unsigned_abs() as usize >= MIN_PEAK_SPACING) {
        accepted[count] = k as u32;
        count += 1;
    }
}

for i in 0..count {
    let k = accepted[i] as usize;
    let (y0, y1, y2) = (self.onset_acf_enhanced[k-1], self.onset_acf_enhanced[k], self.onset_acf_enhanced[k+1]);
    let denom = y0 - 2.0*y1 + y2;
    let (lag_frac, mag) = if denom.abs() < 1e-12 {
        (k as f32, y1)
    } else {
        let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
        (k as f32 + delta, y1 - 0.25 * (y0 - y2) * delta)
    };
    self.candidates[3*i]   = lag_frac;
    self.candidates[3*i+1] = mag;
    self.candidates[3*i+2] = -denom;     // sharpness
}
```

`tau_min` / `tau_max` derived from the BPM bounds at construction:
```rust
self.tau_min = (60.0 / BEAT_TRACKER_MAX_BPM / dt).floor() as usize;
self.tau_max = ((60.0 / BEAT_TRACKER_MIN_BPM / dt).ceil() as usize)
                  .min(onset_acf_len - 2);
```

At default `sr=48000`, `hop_size=1024` ⇒ `dt = 21.33 ms` ⇒ `tau_min = 12`, `tau_max = 70`.

#### 5. Pulse-train scoring

For each candidate `i` with sub-bin lag `τ`:
```rust
let phi_max = (τ.ceil() as usize).max(1);
let mut sum_corr = 0.0;
let mut sum_corr2 = 0.0;
let mut best_corr = -1.0;
let mut best_phi = 0;
let last = self.onset_history.len() - 1;

for phi in 0..phi_max {
    let mut corr = 0.0;
    // Φ₁ at k·τ, weight 1.0
    for k in 0..PULSE_N {
        let off = (k as f32 * τ).round() as i32;
        let pos = last as i32 - phi as i32 - off;
        if pos >= 0 && (pos as usize) < self.onset_history.len() {
            corr += 1.0 * self.onset_history[pos as usize];
        }
    }
    // Φ₂ at k·2τ, weight 0.5
    for k in 0..PULSE_N {
        let off = (k as f32 * 2.0 * τ).round() as i32;
        let pos = last as i32 - phi as i32 - off;
        if pos >= 0 && (pos as usize) < self.onset_history.len() {
            corr += 0.5 * self.onset_history[pos as usize];
        }
    }
    // Φ₁.₅ at (k+0.5)·τ, weight 0.5
    for k in 0..PULSE_N {
        let off = ((k as f32 + 0.5) * τ).round() as i32;
        let pos = last as i32 - phi as i32 - off;
        if pos >= 0 && (pos as usize) < self.onset_history.len() {
            corr += 0.5 * self.onset_history[pos as usize];
        }
    }
    sum_corr  += corr;
    sum_corr2 += corr * corr;
    if corr > best_corr { best_corr = corr; best_phi = phi; }
}

let n = phi_max as f32;
let mean = sum_corr / n;
let var = (sum_corr2 / n - mean * mean).max(0.0);

self.pulse_x[i]   = best_corr.max(0.0);
self.pulse_v[i]   = var;
self.pulse_phi[i] = best_phi as f32;
```

Phase convention: `phi = 0` ⇒ a beat *just landed* on the most recent onset sample. Pulses are placed at `last - phi - k·τ` (going backward in time). Out-of-frame pulses are skipped (paper-faithful: "if an index of the impulse train falls outside the OSS frame, that pulse is omitted").

After the candidate loop, normalize and pick winner:
```rust
let sum_x: f32 = self.pulse_x[..count].iter().sum();
let sum_v: f32 = self.pulse_v[..count].iter().sum();
for i in 0..count {
    let xn = if sum_x > 0.0 { self.pulse_x[i] / sum_x } else { 0.0 };
    let vn = if sum_v > 0.0 { self.pulse_v[i] / sum_v } else { 0.0 };
    self.pulse_score[i] = xn + vn;
}

let (i_star, score_inst) = (0..count)
    .map(|i| (i, self.pulse_score[i]))
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    .unwrap_or((0, 0.0));

let period_inst = self.candidates[3 * i_star];
let phase_inst  = self.pulse_phi[i_star];
```

If `count == 0` (silent input), set `period_inst = NaN`, `phase_inst = NaN`, `score_inst = 0.0`.

#### 6. TEA update + smoothed output

```rust
let inv_2sigma_sq = 1.0 / (2.0 * TEA_GAUSSIAN_SIGMA * TEA_GAUSSIAN_SIGMA);
let alpha = self.tea_alpha;

if score_inst > 0.0 && period_inst.is_finite() {
    for tau in 0..self.tea.len() {
        let delta = tau as f32 - period_inst;
        let gauss = (-delta * delta * inv_2sigma_sq).exp();
        self.tea[tau] = (1.0 - alpha) * self.tea[tau] + alpha * gauss;
    }
} else {
    for v in self.tea.iter_mut() { *v *= 1.0 - alpha; }
}

// argmax in [tau_min, tau_max] with parabolic refine
let mut best_i = self.tau_min;
let mut best_v = -1.0f32;
for i in self.tau_min..=self.tau_max {
    if self.tea[i] > best_v { best_v = self.tea[i]; best_i = i; }
}
let mut tau_smoothed = best_i as f32;
if best_i > self.tau_min && best_i < self.tau_max {
    let (y0, y1, y2) = (self.tea[best_i-1], self.tea[best_i], self.tea[best_i+1]);
    let denom = y0 - 2.0*y1 + y2;
    if denom.abs() > 1e-12 {
        let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
        tau_smoothed = best_i as f32 + delta;
    }
}
```

Phase coherent with the smoothed period — re-run the inner pulse-train scan as a one-candidate pass:
```rust
let phase_smoothed = score_phase_for_tau(tau_smoothed, &self.onset_history);
```

`score_phase_for_tau` is the inner φ-loop from step 5, refactored into a free function returning the best φ. Cost: ~70 phases × 12 muls = ~850 ops/frame.

If `score_inst == 0` (silent), output `NaN` for period and phase to let consumers detect "no fit" — same convention the renderers already handle.

#### 7. Output

```rust
self.beat_grid[0] = if score_inst > 0.0 { tau_smoothed } else { f32::NAN };
self.beat_grid[1] = if score_inst > 0.0 { phase_smoothed } else { f32::NAN };
self.beat_grid[2] = score_inst;

self.beat_state[0] = if tau_smoothed > 0.0 { 60.0 / (tau_smoothed * self.dt) } else { f32::NAN };
self.beat_state[1] = score_inst;
self.beat_state[2] = f32::NAN;
self.beat_state[3] = f32::NAN;

update_beat_pulses(tau_smoothed, phase_smoothed, score_inst, &mut self.beat_position, &mut self.beat_pulses);
```

`update_beat_pulses` simplifies to: if `period.is_nan()` → all NaN; else `beat_position = (phase / period) frac-aligned`, write 4 saws. The free-run-during-silence branch goes away because the new pipeline either has a real (period, phase) or signals "no fit" via NaN, and the renderer side already handles holding last value.

### Setter / param wiring

Add:
```rust
pub fn set_tea_tau_secs(&mut self, tau_secs: f32) {
    let tau = tau_secs.clamp(0.05, 60.0);
    self.tea_alpha = 1.0 - (-self.dt / tau).exp();
}
```

Remove: `set_accum_tau_secs`.

`crates/dsp/src/lib.rs` test changes follow in the implementation plan; keep the spec's tests section short — see "Testing".

## Worklet bridge (`src/audio/dsp-worklet.ts`)

`features` postMessage:
- Add: `onset` (stride-1, length `rmsLen` = `rms_history_len`), `onsetAcf`, `onsetAcfEnhanced`, `tea`, `candidates` (stride-3, length 30).
- Remove: `rmsAcf`, `rmsAcfAccum`, `acfPeaks`. `lowRmsAcf` removed (low-band ACF dropped entirely from the DSP).
- Transfer all new buffers' `.buffer` for zero-copy.

`configured`:
- Add: `onsetLen` (= `rmsLen`), `onsetAcfLen`, `teaLen` (= `onsetAcfLen`), `candidatesLen` (= 30).
- Remove: `rmsAcfLen`, `acfPeaksLen`, `lowRmsAcfLen`.

Param routing:
- `accumTauSecs` removed.
- `teaTauSecs` added → `dsp.set_tea_tau_secs(value)`.

The processor caches `teaTauSecs` (default 4.0), passes it to `set_tea_tau_secs` in `applyConfigure`, mirroring the existing tau-param pattern.

## Param store (`src/params/schemas.ts`, `src/params/WorkletBridge.ts`)

Add:
```ts
{ key: "dsp.teaTauSecs", label: "TEA τ (s)", kind: "continuous",
  min: 0.2, max: 30.0, step: 0.1, default: 4.0, reconfig: false },
```

Remove `dsp.accumTauSecs` schema entry.

`HOT_KEYS`: add `"teaTauSecs"`, remove `"accumTauSecs"`.

## Main thread / renderer (`src/render/`, `src/store/`, `src/App.ts`)

`DebugView.applyConfigured` allocates new store buffers: `onset` (length `onsetLen`, fill 0), `onsetAcf`, `onsetAcfEnhanced`, `tea` (length `onsetAcfLen`, fill 0), `candidates` (length `candidatesLen`, fill NaN). Removes `rmsAcf`, `rmsAcfAccum`, `acfPeaks`, `lowRmsAcf`.

`DebugView.applyFeatures` writes those buffers from the worklet message. The branches for the removed buffers go away.

Render strips:
- New `onset` strip with `LineRenderer` + `linearLayout(0.0, 0.4)` — slots into the empty y=0 band between bufferAcf (yCenter=0.5) and the multiband RMS lane (yCenter=-0.5). Color `0xff9966`. The existing camera presets don't currently use this band so no preset adjustment is needed; consider adding a new preset for it in the implementation plan if desired.
- The existing rms_acf strip becomes the **onset_acf strip**:
  - Base: `onset_acf` (cyan `0x66ffff`).
  - Overlay 1: `onset_acf_enhanced` (pink `0xff99cc`).
  - Overlay 2: `tea` (yellow `0xffff66`).
  - `PeakMarkers` (revived): reads `candidates`, anchored to this strip's y-band, `xForLag` matching the strip's x-mapping. Slot 0 brightest, slot 9 dimmest (existing rank-brightness convention).
  - `BeatGridRenderer` (kept): reads `beat_grid[0]` = `tau_smoothed`, draws `k·τ` markers on this strip.
- The low_rms_acf strip is removed (and the corresponding camera preset adjusted — verify `App.ts` preset 6 still makes sense or repurpose).

`BeatGridScrollingRenderer` still reads `beat_grid[0..2]` = `(period, phase)`. With the new pipeline, `phase` is real whenever there's a fit, so the scrolling grid becomes more reliable. No code change to the renderer itself.

`BeatPulseSquares` unchanged.

`BeatDebugView` follows along — `rmsAcfLen` → `onsetAcfLen` in its `BeatDebugSizes`; renderer max-line counts adjust if needed (tau_max=70 ⇒ `maxLines = ceil(onsetAcfLen / tau_min) ≈ 22`, comfortably under the existing 32).

## Testing

### Rust (`crates/dsp/src/lib.rs`)

- `gen_acf_silent_input_no_nan` — silent input across many frames produces all-zero `onset_acf`, no NaNs.
- `gen_acf_periodic_input_peaks_at_period` — synthetic onset_history with period P (e.g. P=20) yields `onset_acf` peaks at lag P, 2P, … within ±1 lag.
- `harmonic_enhancement_boosts_fundamental` — synthetic `onset_acf` with peaks at P, 2P, 4P; assert `enhanced[P] > enhanced[2P] > enhanced[4P]` (since enhanced[P] sums all three).
- `peak_picking_filters_to_tau_range` — peaks placed at lags inside and outside `[τ_min, τ_max]`; only in-range peaks appear in `candidates`.
- `peak_picking_top_n_descending` — 15 distinct local maxima with decreasing magnitudes; assert top 10 picked in descending order.
- `pulse_train_scores_correctly_for_synthetic_oss` — onset_history with kicks every 30 samples (period 30) and small noise; assert pulse-train scoring picks `period_inst` close to 30 and a coherent phase.
- `pulse_train_disambiguates_octave` — onset_history with peaks at 30, 60, 90 (period 30 with strong harmonics); the candidates list will include both 30 and 60; assert pulse-train scoring prefers 30 (the fundamental). **This is the regression test for the bug this rewrite fixes.**
- `tea_silent_input_decays_to_zero` — silent input over many frames; TEA values monotonically decrease, no NaN.
- `tea_periodic_input_locks_to_period` — repeated periodic input; TEA argmax converges to the true period within tolerance.
- `set_tea_tau_secs_clamps_and_recomputes` — bounds checks + alpha update.
- `tau_min_max_at_default_settings` — assert tau_min ≈ 12, tau_max ≈ 70 for default sr/hop.
- `bpm_output_matches_period` — assert `beat_state[0] == 60 / (beat_grid[0] * dt)` to floating-point tolerance.

### TypeScript (`tests/`)

- `tests/params/WorkletBridge.test.ts` — replace `accumTauSecs` test with `teaTauSecs` forwarding.
- `tests/render/PeakMarkers.test.ts` — already exists; update to read from a `candidates` (stride 3) buffer instead of `acfPeaks`. Test logic unchanged.
- Existing `tests/render/BeatPulseSquares.test.ts`, `tests/render/BeatDebugView.test.ts` (if present) — adjust any sizes/keys that reference the removed buffers.

### Manual

- Run on a track with a clear sub-80-BPM beat (e.g. ambient / dub at ~60 BPM): yellow `BeatGridRenderer` lines align with the visible ACF peaks (verifies the bug fix).
- Run on a track with strong sub-divisions (the original "image #1" test case): yellow lines align with the *fundamental*, not the sub-division.
- Silence → grid bars vanish (NaN); restart audio → lock-on within a few seconds (TEA τ = 4 s).

## Risks and notes

- **Gen-ACF FFT size scales with `rms_history_len`.** The `realfft` planner allocates twiddle factors per size; `applyConfigure` rebuilds `Dsp` whenever `rmsHistoryLen` changes, so the planner cost is bounded to that already-recreated path.
- **Pulse rounding loses some sub-bin precision.** Each pulse position is rounded independently against full-precision `τ`, so pulses don't all snap onto the same integer grid. Within the integer-φ scan, residual sub-bin error is < 1 sample. Sub-bin φ is left for future work.
- **TEA argmax is driven by score-weighted Gaussian votes; magnitude-weighting is left out.** Adding a `score_inst`-weighted update (i.e. multiply `gauss` by `score_inst` before EMA) is a one-line change if the unweighted version proves jittery in practice.
- **Octave decider is intentionally absent.** The pulse-train scoring is the paper's main octave-disambiguator. The Section II-C step-4 ML decider is only relevant for offline batch scoring against a labeled corpus, which we don't have. If half-time / double-time errors persist on real tracks, revisit then.
- **`update_beat_pulses` simplification removes a fallback.** The current code free-runs the saws during silence; the new code outputs NaN. `BeatPulseSquares` already handles NaN by holding the last value, which is the more honest behavior.
- **Camera preset 6 (low_rms_acf strip) becomes orphaned.** Either repurpose it for the onset strip or remove it; flagged for the implementation plan.

## Roadmap impact

After implementation, update `ROADMAP.md` "Beatdetector improvements":

- Mark items 1, 2, 3 (accumulator, peak picking, sawtooth/grid) shipped (the prior partial coverage is superseded by this rewrite).
- Item 4 (onset/downbeat) reclassified: onset signal is now a first-class part of the pipeline; downbeat detection (= which `k·τ` is the bar boundary) remains deferred and is closely related to the missing `beats_per_measure` detection.
- Add a new entry under "Performance" or similar: "Spectral-flux OSS" replaces "Migrate autocorrelation to FFT-based" (the FFT-based ACF is now done, just on the beat path; full-band/low-band visualization ACFs are gone).
