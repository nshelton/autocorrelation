# Multi-Band RMS Design

**Date:** 2026-04-28
**Status:** Design approved, awaiting plan

## Goal

Add per-band RMS time-series for three frequency bands (low / mid / high) on top of the existing full-band RMS, plus a low-band RMS-ACF beat detector alongside the existing full-band one. Render the per-band lines overlaid (R/G/B) on the same Y-strip as the existing white full-band RMS, and the new low-band ACF overlaid on the existing rms_acf strip. Per-band levels also become available to future consumers (parameter mapping, modulation sources) via the FeatureStore.

## Architecture

The bands are computed in the existing Rust DSP, where the FFT already runs every hop, by summing `|FFT[k]|²` over each band's bin range and applying a Parseval-correct windowing scale. This produces band RMS values that are physically commensurate with the existing time-domain full RMS — i.e. `low² + mid² + high² ≈ full²` — which makes the four-line overlay readable and gives downstream consumers values with concrete physical meaning rather than arbitrary normalized levels.

The audio path is unchanged in shape. `Dsp::process` gains three new band-RMS computations after the FFT and a second detrended-ACF computation on the low-band history. The worklet posts four new `Float32Array` features per hop alongside the existing five. The main-thread `FeatureStore` gains four new keys; the render layer adds three colored `LineRenderer` instances overlaid on the existing RMS strip plus one `LineRenderer` overlaid on the existing rms_acf strip.

No real time-domain filters (IIR/FIR) are introduced. All band information is derived from the spectral representation already being computed, so per-hop CPU cost is one extra small loop and four extra `autocorrelate`/EMA/postMessage payloads — nothing audio-rate.

## Tech Stack

- **Rust + `realfft`** — existing crate, no new dependencies. The forward FFT and `freq_buffer` are already populated; we read from it.
- **TypeScript AudioWorklet** — existing pattern, no new APIs.
- **Three.js WebGPU `LineRenderer`** — existing class, no changes. Four new instances configured with the existing `linearLayout` helper.
- **`@types/audioworklet`** — already provides `sampleRate` global typing used by the worklet.

## File Structure

**New files:** none.

**Modified files:**

- `crates/dsp/src/lib.rs` — new constants (`LOW_BAND_HZ_MAX`, `MID_BAND_HZ_MAX`); new fields (`low_band_bin_end`, `mid_band_bin_end`, `parseval_band_scale`, three band histories of length 512, low_rms_detrended scratch, low_rms_acf of length 256); `Dsp::new` derives bin indices and Parseval scale; `process()` gains a per-band loop after the FFT and a second detrended-ACF computation; existing `rms_acf` computation moves to after the FFT for symmetry with the new low_rms_acf; four new public getters; eight new tests.
- `src/audio/dsp-worklet.ts` — extend the `postMessage` payload with `rmsLow`, `rmsMid`, `rmsHigh`, `rmsAcfLow` and their transferable buffers.
- `src/App.ts` — prime four new `FeatureStore` keys; extend the port-message type and dispatch; construct three new colored `LineRenderer`s for band RMS and one for low_rms_acf; add them to the scene before the existing white-rms and existing rms_acf lines so the existing lines draw on top at line crossings.

## Components

### `crates/dsp/src/lib.rs`

**New constants** at module scope:

```rust
const LOW_BAND_HZ_MAX: f32 = 150.0;
const MID_BAND_HZ_MAX: f32 = 1500.0;
```

**New `Dsp` fields:**

```rust
// Low band = bins 1..=low_band_bin_end. Bin 0 (DC) is always skipped.
// Mid = (low_end+1)..=mid_band_bin_end.
// High = (mid_end+1)..=N/2-1. Nyquist (bin N/2) is excluded.
low_band_bin_end: usize,
mid_band_bin_end: usize,

/// Parseval scale converting Σ|X[k]|² in a band → band RMS² in
/// time-domain units. Equals 2 / (N · Σ(hann²)). The 2 accounts
/// for the one-sided real spectrum; Σ(hann²) is the window's
/// energy correction. Derivation in the math section below.
parseval_band_scale: f32,

low_rms_history:  Vec<f32>,  // 512, mirrors existing rms_history shape
mid_rms_history:  Vec<f32>,  // 512
high_rms_history: Vec<f32>,  // 512

low_rms_detrended: Vec<f32>,  // 512, scratch for low_rms_acf
low_rms_acf:       Vec<f32>,  // 256, mirrors rms_acf shape (RMS_ACF_LEN)
```

**Constructor changes** (`Dsp::new`):

```rust
fn bin_for_hz(hz: f32, sample_rate: f32, n: usize) -> usize {
    let bin = (hz * n as f32 / sample_rate).round() as usize;
    bin.clamp(1, n / 2 - 1)
}

let low_band_bin_end  = bin_for_hz(LOW_BAND_HZ_MAX,  sample_rate, window_size);
let mid_band_bin_end  = bin_for_hz(MID_BAND_HZ_MAX,  sample_rate, window_size);
// Note: callers must pick LOW_BAND_HZ_MAX < MID_BAND_HZ_MAX. At the
// existing 150/1500 Hz settings and 48 kHz / 2048-pt FFT, the snapped
// bin boundaries are 6 and 64.

let hann_energy = self.hann.iter().map(|h| h * h).sum::<f32>();
let parseval_band_scale = 2.0 / (window_size as f32 * hann_energy);
```

**`process()` order** (existing logic preserved; new and moved steps marked):

1. Copy waveform.
2. Compute time-domain full RMS, push to `rms_history`.
3. `autocorrelate(waveform) → buffer_acf`.
4. Apply Hann window, fill `fft_buffer`.
5. Forward FFT → `freq_buffer`.
6. **NEW: per-band loop.** For each band, compute `Σ|freq_buffer[k]|²` over the band's bin range, multiply by `parseval_band_scale`, take sqrt → that band's instantaneous RMS. Shift the corresponding history left, append at end.
7. Spectrum dB-normalize loop (existing).
8. **MOVED:** detrend `rms_history` → `autocorrelate` → `rms_acf`.
9. **NEW:** detrend `low_rms_history` → `autocorrelate` → `low_rms_acf`.

**New public getters** (mirror existing pattern — `pub fn ... -> Vec<f32>` clones):

```rust
pub fn low_rms_history(&self)  -> Vec<f32>
pub fn mid_rms_history(&self)  -> Vec<f32>
pub fn high_rms_history(&self) -> Vec<f32>
pub fn low_rms_acf(&self)      -> Vec<f32>
```

### Math: Parseval-correct band RMS

Given Hann-windowed input `w[n] · x[n]` of length N, with one-sided real-FFT output `X[k]` for `k = 0..N/2`:

By Parseval's theorem on the full N-bin spectrum:

```
Σ_{k=0..N-1} |X[k]|² = N · Σ_{n=0..N-1} (w[n] · x[n])²
```

Since `X[N-k] = conj(X[k])` for `k = 1..N/2-1`, the one-sided sum doubles every interior bin:

```
Σ_{k=0..N-1} |X[k]|²  =  |X[0]|² + |X[N/2]|² + 2 · Σ_{k=1..N/2-1} |X[k]|²
```

Under the assumption that `x[n]` is locally stationary across the analysis window (a standard approximation for short-time spectral analysis), the windowing factor in the time-domain energy splits multiplicatively:

```
Σ (w[n] · x[n])²  ≈  (Σ w[n]²) / N · Σ x[n]²
```

Combining these and isolating the band B (a subset of one-sided bins excluding 0 and N/2):

```
band_x_energy   ≈  2 · Σ_{k in B} |X[k]|²  /  Σ w[n]²
band_x_rms²     =  band_x_energy / N
                =  (2 / (N · Σ w[n]²))  ·  Σ_{k in B} |X[k]|²
                =  parseval_band_scale  ·  Σ_{k in B} |X[k]|²
```

So:

```
band_x_rms = sqrt( parseval_band_scale · Σ_{k in B} |X[k]|² )
```

The unit-test for this is straightforward: a unit-amplitude sine has time-domain RMS `≈ 1/√2 ≈ 0.707`; placing one entirely in a band should give that band a value near 0.707 and the other bands near 0. A multi-frequency signal should satisfy `√(low² + mid² + high²) ≈ full_rms` within ~5% (the slack accounts for stationarity error and energy leaked across bin boundaries).

### `src/audio/dsp-worklet.ts`

Extend the existing post-FFT block:

```ts
const wf  = new Float32Array(this.dsp.waveform());
const sp  = new Float32Array(this.dsp.spectrum());
const rms = new Float32Array(this.dsp.rms_history());
const ba  = new Float32Array(this.dsp.buffer_acf());
const ra  = new Float32Array(this.dsp.rms_acf());
const rmsLow    = new Float32Array(this.dsp.low_rms_history());
const rmsMid    = new Float32Array(this.dsp.mid_rms_history());
const rmsHigh   = new Float32Array(this.dsp.high_rms_history());
const rmsAcfLow = new Float32Array(this.dsp.low_rms_acf());

this.port.postMessage(
  { type: "features",
    waveform: wf, spectrum: sp, rms,
    bufferAcf: ba, rmsAcf: ra,
    rmsLow, rmsMid, rmsHigh, rmsAcfLow },
  [wf.buffer, sp.buffer, rms.buffer, ba.buffer, ra.buffer,
   rmsLow.buffer, rmsMid.buffer, rmsHigh.buffer, rmsAcfLow.buffer],
);
```

### `src/App.ts`

Prime four new `FeatureStore` keys at the same spot where existing primings happen:

```ts
this.store.set("rmsLow",    new Float32Array(512));
this.store.set("rmsMid",    new Float32Array(512));
this.store.set("rmsHigh",   new Float32Array(512));
this.store.set("rmsAcfLow", new Float32Array(256));
```

Extend the port-message type and dispatch (mirrors existing pattern).

Construct three new `LineRenderer`s for band RMS reusing the **same `layout` arguments as the existing white-rms line** (whatever Y-offset and height that line uses today — the plan will read App.ts to capture those values verbatim). They are **added to the scene before the existing white-rms line** so the white line draws on top at line crossings:

```ts
const bandRmsLayout = /* identical linearLayout(yOffset, height) to the existing white rms */;
this.lowRmsLine  = new LineRenderer({ source: () => this.store.get("rmsLow"),
                                       layout: bandRmsLayout,
                                       color: 0xff4444 });
this.midRmsLine  = new LineRenderer({ source: () => this.store.get("rmsMid"),
                                       layout: bandRmsLayout,
                                       color: 0x44ff44 });
this.highRmsLine = new LineRenderer({ source: () => this.store.get("rmsHigh"),
                                       layout: bandRmsLayout,
                                       color: 0x4488ff });
scene.add(this.lowRmsLine.object3d);
scene.add(this.midRmsLine.object3d);
scene.add(this.highRmsLine.object3d);
// the existing white-rms line is already added by App.ts later, so it draws on top.
```

Construct one new `LineRenderer` for `rmsAcfLow` reusing the same `layout` as the existing `rms_acf` line (identical Y-offset and height). Color: red (`0xff4444`) to track the low-band RMS color. **Added to the scene before** the existing `rms_acf` line for the same reason.

Per-frame `update()` calls for the four new lines are added in the render loop alongside the existing ones.

## Data Flow

```
Audio source ──► AudioWorklet ──► Dsp::process(window) ──┐
                                                          │ existing: waveform, spectrum, rms, buffer_acf, rms_acf
                                                          └ NEW:      rmsLow, rmsMid, rmsHigh, rmsAcfLow
                                                          │
                                  port.postMessage with all 9 transferable Float32Array features
                                                          │
                                                          ▼
                              FeatureStore (5 existing + 4 new keys)
                                                          │
                                                          ▼
                              Render loop: existing 5 LineRenderers + 4 new ones
                                  - 3 band-RMS lines added before white-rms (white on top)
                                  - 1 low-rms-acf line added before existing rms_acf (existing on top)
```

## Error Handling

No new failure modes. The bin-range derivation in `Dsp::new` clamps to `[1, N/2-1]` so degenerate sample-rate / window-size combinations don't index out of bounds (worst case, all bands collapse to a single bin — observable in tests, not a panic).

If the user's chosen `LOW_BAND_HZ_MAX ≥ MID_BAND_HZ_MAX` (currently impossible since both are `const f32`), the resulting bin layout would have an empty mid band — the code would still run and produce zeros for that band. Not a panic; not worth a runtime check given the constants are compile-time.

## Testing

Eight new Rust unit tests in `crates/dsp/src/lib.rs::tests`:

1. **Bin range derivation.** At sr=48000, N=2048, assert `low_band_bin_end == 6` and `mid_band_bin_end == 64`.
2. **Pure 100 Hz sine** (low band) → low band history's newest value ≈ 0.707, mid ≈ 0, high ≈ 0 within tolerance after enough iterations.
3. **Pure 1 kHz sine** (mid band) → mid ≈ 0.707, low ≈ 0, high ≈ 0.
4. **Pure 5 kHz sine** (high band) → high ≈ 0.707, low ≈ 0, mid ≈ 0.
5. **Parseval consistency.** For a known multi-tone signal (e.g., sum of sines at 100 / 800 / 5000 Hz with chosen amplitudes), `√(low² + mid² + high²) ≈ full_rms` within 5% tolerance (covers stationarity error + bin-boundary leakage).
6. **Silence** → after one process call with all-zero input, all three band histories' newest sample is exactly 0.0.
7. **Band history shifts oldest out** — mirrors the existing `rms_history_shifts_oldest_out` test, applied to one of the band histories.
8. **`low_rms_acf` constant input is zero** — fill `low_rms_history` via repeated process calls of a constant non-zero signal, assert detrended ACF is near zero. Mirrors the existing `rms_acf_constant_input_is_zero`.

No new TS unit tests. The worklet message extension and render-layer additions are integration-tested manually in the browser (same pattern as v2 and v3).

## Future Integration

Two natural follow-ups, neither in scope for this design:

- **v3.1 ParamStore.** `LOW_BAND_HZ_MAX` and `MID_BAND_HZ_MAX` are obvious candidates for the ParamStore once that ships — they'd become tweakable runtime sliders (with the worklet reconfig path that v3.1 is building). For now they stay as compile-time constants.
- **Per-band beat-detection visualization.** The new `rmsAcfLow` is exposed and rendered, but no automatic tempo extraction is included; that's a separate downstream signal-processing concern that should ride on whatever the existing full-band ACF visualization is doing today.
