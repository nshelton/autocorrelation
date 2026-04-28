# Beat detection: ACF accumulator + peak picking + visualization

Date: 2026-04-28

## Goal

Lay the foundation for tempo/beat detection by extracting reliable, sub-bin-precise tempo peaks from the full-band RMS autocorrelation. This spec covers items 1–3 of the roadmap's "Beatdetector improvements" entry. BPM determination via grid-fitting (item 3 of the user's three-point ask) and onset-based downbeat detection (roadmap item 4) are deferred — they depend on having stable peaks first, which is what this delivers.

## Scope

In:

1. A decaying accumulator over `rms_acf` (full-band, instantaneous). The instantaneous version stays as-is.
2. Peak picking on the accumulator: top 10 local maxima at lags ≥ 10, min spacing 3 lags, sub-bin parabolic refinement.
3. Visualization: accumulator overlaid on the existing rms_acf chart strip; vertical line markers at peak lags via a new `PeakMarkers` render component.

Out (deferred):

- Low-band `low_rms_acf` — apply same machinery later if desired.
- BPM determination from peaks (multiples / quarter-note grid fitting).
- Onset / downbeat detection.
- FFT-based ACF migration (separate roadmap item).

## Architecture

The existing pipeline split is preserved:

- **Rust DSP (in worklet)** gains the accumulator, peak picking, and two new outputs.
- **Worklet** posts the two new buffers each hop.
- **Main thread** renders one new line strip (accumulator) and a new `PeakMarkers` component (vertical lines).
- **ParamStore** gains one new tunable param: `dsp.accumTauSecs`.

No new threading boundaries; no FFT-based ACF migration; no changes to the existing rms_acf, low_rms_acf, or any spectrum/RMS strip.

## Rust DSP changes (`crates/dsp/src/lib.rs`)

### New constants

```rust
const ACCUM_TAU_DEFAULT_SECS: f32 = 4.0;
const MIN_PEAK_LAG: usize = 10;
const MAX_PEAKS: usize = 10;
const MIN_PEAK_SPACING: usize = 3;
```

### New `Dsp` fields

- `rms_acf_accum: Vec<f32>` — same length as `rms_acf` (= `rms_history_len / 2`). Decaying accumulator over the instantaneous full-band ACF.
- `accum_alpha: f32` — EMA coefficient. Computed from `accum_tau_secs` and the existing `dt = hop_size / sample_rate` (already on the struct), via `alpha = 1 - exp(-dt / tau)`. Same convention as `smoothing_alpha`.
- `acf_peaks: Vec<f32>` — fixed length `2 * MAX_PEAKS = 20`. Interleaved `[lag0, mag0, lag1, mag1, ...]`. Slots without a peak are filled with `f32::NAN` so the renderer can detect "no peak" with a single `isNaN` check (0.0 collides with valid lag values).
- `peak_candidates: Vec<(usize, f32)>` — preallocated scratch for peak picking. Capacity reserved in `new()` at `rms_acf_len / 2` (worst case: every other lag is a local max). Cleared each call; no per-frame heap allocation.

### New setter

```rust
pub fn set_accum_tau_secs(&mut self, tau_secs: f32) {
    let tau = tau_secs.clamp(0.05, 60.0);
    self.accum_alpha = 1.0 - (-self.dt / tau).exp();
}
```

Bounds: 0.05 s lower bound avoids divide-by-zero and a runaway alpha; 60 s upper bound matches musical sensibility. Pattern mirrors the existing `set_smoothing_tau`.

### `process()` changes

After the existing `autocorrelate(&self.rms_detrended, &mut self.rms_acf)`:

1. **EMA accumulator update**:
   ```rust
   for i in 0..self.rms_acf_accum.len() {
       self.rms_acf_accum[i] =
           self.accum_alpha * self.rms_acf[i]
           + (1.0 - self.accum_alpha) * self.rms_acf_accum[i];
   }
   ```
2. **Peak picking** (see algorithm below) — writes into `acf_peaks`.

### Peak-picking algorithm

Operates on `rms_acf_accum` (not the instantaneous trace).

1. **Scan candidates**: walk lags `MIN_PEAK_LAG..len-1`. Index `k` is a candidate iff `accum[k] > accum[k-1] && accum[k] > accum[k+1] && accum[k] > 0`. Negative correlations are skipped (anti-correlations aren't beats). Push `(k, accum[k])` into `peak_candidates` (capacity reserved; no realloc).
2. **Sort** `peak_candidates` by magnitude descending using `sort_by` (in-place).
3. **Greedy select with min-spacing**: walk sorted candidates; accept index `k` if its integer lag is at least `MIN_PEAK_SPACING` away from every already-accepted peak's integer lag. Stop when 10 are accepted or candidates exhausted. Accepted peaks tracked in a stack-allocated `[u32; MAX_PEAKS]` with a count.
4. **Sub-bin parabolic refinement** for each accepted peak:
   ```
   y0, y1, y2 = accum[k-1], accum[k], accum[k+1]
   denom = y0 - 2*y1 + y2
   if denom.abs() < 1e-12:
       lag_frac = k as f32; mag = y1
   else:
       δ = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
       lag_frac = k as f32 + δ
       mag      = y1 - 0.25 * (y0 - y2) * δ
   ```
5. **Write output**: for slot `i in 0..accepted_count`, write `[lag_frac, mag]` into `acf_peaks[2i..2i+2]`. For `i in accepted_count..MAX_PEAKS`, write `[NaN, NaN]`.

The candidate sort uses *integer-lag* magnitudes; the *output* magnitude is the parabolic-refined value. Standard convention; avoids re-sorting after refinement.

### New getters

```rust
pub fn rms_acf_accum(&self) -> Vec<f32> { self.rms_acf_accum.clone() }
pub fn acf_peaks(&self) -> Vec<f32>     { self.acf_peaks.clone() }
```

Per-call `clone()` is the existing wasm-bindgen pattern; out of scope to change here.

## Worklet bridge (`src/audio/dsp-worklet.ts`)

Extend the `WorkletInbound` `param` discriminated union:

```ts
| { type: "param"; key: "hopSize" | "smoothingTauSecs" | "dbFloor" | "accumTauSecs"; value: number };
```

Route `accumTauSecs` to `dsp.set_accum_tau_secs(value)` in the existing param handler — same pattern as `smoothingTauSecs`.

Extend the `features` postMessage with two new transferred buffers:

```ts
{
  type: "features",
  // ... existing fields ...
  rmsAcfAccum: new Float32Array(this.dsp.rms_acf_accum()),
  acfPeaks:    new Float32Array(this.dsp.acf_peaks()),
}
// transfer list adds: rmsAcfAccum.buffer, acfPeaks.buffer
```

Extend the `configured` message with `acfPeaksLen: 20` for parity with the existing `*Len` fields.

In `applyConfigure`: after constructing the new `Dsp`, call `this.dsp.set_accum_tau_secs(this.accumTauSecs)` (mirroring the existing tau/dbFloor pattern). Track `accumTauSecs` as a private field on the processor with default `4.0`.

## Param wiring

### `src/params/schemas.ts`

Add to `analysisSchemas`:

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

### `src/params/WorkletBridge.ts`

Add `"accumTauSecs"` to the `HOT_KEYS` tuple. The existing `bootstrap()` and `handleChange()` will then forward it without further changes.

## Main thread (`src/App.ts`)

### Feature handler

In the `features` message branch, write the two new buffers into the store:

```ts
if (msg.rmsAcfAccum) this.store.set("rmsAcfAccum", msg.rmsAcfAccum);
if (msg.acfPeaks)    this.store.set("acfPeaks",    msg.acfPeaks);
```

### `rebuildLineRenderers`

- Seed the store with empty buffers for `rmsAcfAccum` (length `rmsAcfLen`) and `acfPeaks` (length 20, filled with NaN).
- Construct a new `LineRenderer` for the accumulator, overlaid on the existing rms_acf strip:
  ```ts
  this.rmsAcfAccumLine = new LineRenderer({
    source: () => this.store.get("rmsAcfAccum"),
    layout: linearLayout(-1.0, 0.4),  // same y-band as rmsAcfLine
    color:  0x66ffff,                  // cyan vs. existing 0xff99cc pink
  });
  ```
- Construct a new `PeakMarkers` component (see below) reading `acfPeaks`, anchored to the same y-band.
- Add both to the scene; dispose them at the top of `rebuildLineRenderers` alongside the other line renderers; call `update()` on both each frame in `loop`.

## `PeakMarkers` component (`src/render/PeakMarkers.ts`)

New module, parallel in shape to `LineRenderer`.

### Interface

```ts
export interface PeakMarkersOptions {
  source: () => Float32Array;                                  // [lag0, mag0, lag1, mag1, ...]
  maxPeaks: number;                                            // 10
  lagDomain: number;                                           // ACF buffer length, e.g. 256
  yCenter: number;                                             // -1.0 to match linearLayout
  ySpan: number;                                               // 0.4 to match linearLayout amplitude
  xForLag: (lag: number, lagDomain: number) => number;         // matches the chart's x-mapping
  baseColor?: number;                                          // default 0xffff66
}
```

`xForLag` is passed by the App so the markers and the underlying rms_acf chart use the same x-mapping by construction. Avoids the markers drifting if the chart's layout changes.

### Geometry

- One `THREE.LineSegments` with a `BufferGeometry` containing fixed buffers:
  - `position`: `Float32Array(2 * maxPeaks * 3)`, `DynamicDrawUsage`. Each peak occupies two consecutive vertices.
  - `color`: `Float32Array(2 * maxPeaks * 3)`, `DynamicDrawUsage`. Both endpoints of a segment share the same RGB so the segment is solid.
- `THREE.LineBasicMaterial({ vertexColors: true })`.

### `update()` (per frame)

For slot `i in 0..maxPeaks`:

1. Read `lag = src[2i]`, `mag = src[2i+1]` (mag currently unused for layout; reserved for future intensity coding).
2. If `Number.isNaN(lag)`: write both vertices to `(0, yCenter, 0)` and color `(0, 0, 0)`. Segment collapses to a black point at the chart center, effectively invisible. Avoids branchy geometry resizing.
3. Else:
   - `x = xForLag(lag, lagDomain)`.
   - Top vertex: `(x, yCenter + ySpan, 0)`. Bottom vertex: `(x, yCenter - ySpan, 0)`.
   - Color: `baseColor` decoded once at construction, multiplied per-frame by `1 - 0.75 * (i / (maxPeaks - 1))` so slot 0 is full brightness and slot 9 is ~25%.

Set `position.needsUpdate = true` and `color.needsUpdate = true` after the loop.

### `dispose()`

Disposes geometry + material. Same convention as `LineRenderer.dispose()`.

## Testing

### Rust (`crates/dsp/src/lib.rs`)

- `accum_alpha_matches_formula` — at default tau, `accum_alpha == 1 - exp(-dt / ACCUM_TAU_DEFAULT_SECS)`.
- `set_accum_tau_secs_clamps_and_recomputes` — passing values outside `[0.05, 60.0]` clamps; alpha changes after the call.
- `accumulator_silent_input_is_zero` — silent input across many calls keeps the accumulator at zero.
- `accumulator_converges_to_instantaneous_for_steady_input` — feed a synthetic periodic signal until convergence; assert `rms_acf_accum[k] ≈ rms_acf[k]` within tolerance.
- `peaks_silent_input_all_nan` — silent input → every slot in `acf_peaks` is NaN.
- `peaks_synthetic_periodic_finds_correct_lag` — feed an `rms_history` with a known period (via repeated `process` calls or a test-only setter); assert peak 0's `lag_frac` matches within 0.5.
- `peaks_subbin_interpolation_offset` — synthetic accumulator with an asymmetric triangular peak where parabolic interp predicts a known δ; assert the output `lag_frac` matches.
- `peaks_min_spacing_enforced` — accumulator with two equal-magnitude lobes 2 lags apart; assert exactly one is in the output.
- `peaks_min_lag_enforced` — peak at lag 5; assert it is **not** picked (below `MIN_PEAK_LAG = 10`).
- `peaks_top_n_selection` — accumulator with 15 distinct local maxima of decreasing magnitude; assert exactly the top 10 are picked, in descending magnitude order.

Allocation-free peak picking is guaranteed structurally (preallocated `peak_candidates`, stack-bounded accepted set) and documented inline; no runtime test.

### TypeScript (`tests/`)

- `tests/params/WorkletBridge.test.ts` — extend to cover `accumTauSecs` forwarding (parallels existing `smoothingTauSecs` test).
- `tests/render/PeakMarkers.test.ts` — new file (uses `happy-dom`, like `tests/render/CameraRig.test.ts`):
  - Constructor produces `2 * maxPeaks` vertices.
  - `update()` with all-NaN source → all segments collapsed to chart center.
  - `update()` with two real peaks → first two segments at expected x, remaining 8 collapsed.
  - Color-by-rank: brightest at slot 0, dimmest at slot N-1.
  - `dispose()` releases geometry + material.

## Risks and notes

- **Color overlap on the chart**: cyan accumulator + pink instantaneous + yellow peak markers in the same y-band may get visually busy. If it does, fix is purely cosmetic (palette tweak) and doesn't affect the architecture.
- **`xForLag` coupling**: the markers must use the same x-mapping as the rms_acf line. Passing the function in (rather than reimplementing) is the safest design.
- **MAX_PEAKS coupled to a hard-coded 20-slot output**: changing `MAX_PEAKS` requires changing both ends of the `acf_peaks` length contract. Documented inline at the constant; not configurable at runtime.
- **WorkletBridge has uncommitted local changes** at the time of writing. The plan should respect those when integrating.

## Roadmap impact

After implementation, update `ROADMAP.md` "Beatdetector improvements":

- Mark items 1 and 2 (accumulator, peak picking) shipped.
- Item 3 (sawtooth/grid) and item 4 (onset/downbeat) remain deferred.
