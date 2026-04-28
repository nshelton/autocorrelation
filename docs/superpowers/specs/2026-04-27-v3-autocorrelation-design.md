# v3 — Autocorrelation Design

**Date:** 2026-04-27
**Status:** Design approved, awaiting plan

## Goal

Add buffer autocorrelation and RMS-envelope autocorrelation to the visualizer. The two new features are rendered as line strips in the existing Three.js scene, alongside waveform / spectrum / rms. The RMS-envelope ACF makes beat structure visible; the buffer ACF makes pitch / periodicity visible. No derived numeric output (e.g., BPM readout) ships in v3 — the plot itself is the deliverable.

## Architecture

All math lives in the existing Rust DSP crate (`crates/dsp/src/lib.rs`). The `Dsp` struct gains two new internal buffers and two new public getters; `process()` computes both ACFs each hop. The worklet posts the two new arrays alongside the existing features in a single message; `App` stores them in `FeatureStore` and wires them to two new `LineRenderer` instances.

**Algorithm:** Direct time-domain autocorrelation, O(N²). Chosen for simplicity and to ship the visualization quickly. Migration to FFT-based (Wiener–Khinchin) is on the roadmap once the visualization is proven correct.

**Pre-processing rules:**
- Buffer ACF runs on the **raw, unwindowed** waveform. The Hann window is FFT-only — autocorrelation wants the unbiased signal.
- RMS-envelope ACF runs on **mean-subtracted** rms_history. Without detrending, the DC bias from average loudness drowns out the tempo peaks.

**Normalization:** Both ACFs are divided by their lag-0 value, so the y-axis is `[-1, 1]` regardless of input loudness. Silent input (lag-0 = 0) falls back to all zeros, no NaN.

## Tech Stack

- Rust (`crates/dsp/`) — no new dependencies.
- TypeScript / Vite (`src/`) — no new dependencies.
- Three.js WebGPURenderer (`src/render/`) — reuses existing `LineRenderer`, `LineLayouts`, `CameraRig`.

## File Structure

**Modify:**
- `crates/dsp/src/lib.rs` — new fields, new getters, new `autocorrelate` free function, new computations in `process()`, new tests.
- `src/audio/dsp-worklet.ts` — read two new arrays from the WASM struct, post them with the existing features message, list their `ArrayBuffer`s in the transferables array.
- `src/App.ts` — prime the store with two new zero-filled buffers; construct two new `LineRenderer`s; adjust line y-positions to fit five lines; add two camera presets and update existing presets to fit the wider vertical range; extend `presetKeys` with `"5"` and `"6"`; extend the `onmessage` handler to set the two new store keys.

**No changes:** `index.html`, `src/main.ts`, `src/audio/AudioSource.ts`, `src/render/LineRenderer.ts`, `src/render/LineLayouts.ts`, `src/render/CameraRig.ts`, `src/store/FeatureStore.ts`, `src/ui/Stats.ts`, `vite.config.ts`.

## Components

### `autocorrelate` (Rust free function in `lib.rs`)

```rust
fn autocorrelate(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    for k in 0..output.len() {
        let mut sum = 0.0f32;
        for i in 0..(n - k) {
            sum += input[i] * input[i + k];
        }
        output[k] = sum;
    }
    let zero = output[0];
    if zero > 0.0 {
        for v in output.iter_mut() { *v /= zero; }
    } else {
        output.fill(0.0);
    }
}
```

- Input length: any. Output length: any ≤ input length (caller controls how many lags to compute).
- Output is in-place via `&mut [f32]` so callers can keep a reusable buffer.
- Normalization is by ACF[0]; silent input collapses to all zeros.

### `Dsp` struct additions

New constants:
```rust
const BUFFER_ACF_LEN: usize = 1024;  // window_size / 2
const RMS_ACF_LEN: usize = 256;      // RMS_HISTORY_LEN / 2
```

New fields:
```rust
buffer_acf: Vec<f32>,         // length BUFFER_ACF_LEN
rms_acf: Vec<f32>,            // length RMS_ACF_LEN
rms_detrended: Vec<f32>,      // length RMS_HISTORY_LEN, scratch for mean-subtraction
```

`Dsp::new(window_size)` initializes all three with `vec![0.0; …]`.

In `process()`, after the existing RMS-history shift and before the FFT block:

```rust
// Buffer ACF on the raw signal (not the windowed FFT input).
autocorrelate(&self.waveform, &mut self.buffer_acf);

// RMS-envelope ACF on the mean-subtracted history.
let mean = self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32;
for (dst, src) in self.rms_detrended.iter_mut().zip(self.rms_history.iter()) {
    *dst = src - mean;
}
autocorrelate(&self.rms_detrended, &mut self.rms_acf);
```

New getters (mirror the existing pattern):
```rust
pub fn buffer_acf(&self) -> Vec<f32> { self.buffer_acf.clone() }
pub fn rms_acf(&self) -> Vec<f32> { self.rms_acf.clone() }
```

### `dsp-worklet.ts` message change

```ts
const ba = new Float32Array(this.dsp.buffer_acf());
const ra = new Float32Array(this.dsp.rms_acf());
this.port.postMessage(
  { type: "features", waveform: wf, spectrum: sp, rms,
    bufferAcf: ba, rmsAcf: ra },
  [wf.buffer, sp.buffer, rms.buffer, ba.buffer, ra.buffer],
);
```

The message-shape change is additive: existing fields keep their names and types.

### `App.ts` changes

**Store priming** (must happen before LineRenderer construction, prevents the WebGPU BufferAttribute resize bug from v1):
```ts
this.store.set("bufferAcf", new Float32Array(1024));
this.store.set("rmsAcf", new Float32Array(256));
```

**Two new LineRenderers:**
```ts
this.bufferAcfLine = new LineRenderer({
  source: () => this.store.get("bufferAcf"),
  layout: linearLayout(0.5, 0.4),
  color: 0xcc99ff,
});
scene.add(this.bufferAcfLine.object3d);

this.rmsAcfLine = new LineRenderer({
  source: () => this.store.get("rmsAcf"),
  layout: linearLayout(-1.0, 0.4),
  color: 0xff99cc,
});
scene.add(this.rmsAcfLine.object3d);
```

**Layout (five lines, paired by relationship):**

```
y = +1.0   waveform        cyan      0x66ffcc
y = +0.5   buffer-ACF      purple    0xcc99ff   ← derived from waveform
y =  0.0   spectrum        orange    0xffaa66
y = -0.5   rms             white     0xffffff
y = -1.0   rms-ACF         pink      0xff99cc   ← derived from rms
```

Each line keeps `height = 0.4`, so the ±0.2 fluctuation regions don't overlap their neighbors.

**Update existing line layouts:**
- Waveform: `linearLayout(0.6, 0.5)` → `linearLayout(1.0, 0.4)`
- Spectrum: `logSpectrumLayout(0.0, 0.5)` → `logSpectrumLayout(0.0, 0.4)`
- RMS: `linearLayout(-0.6, 0.5)` → `linearLayout(-0.5, 0.4)`

**Camera presets:**
- `front`: position `(0, 0, 4)`, target `(0, 0, 0)` — backed up from z=3 to fit ±1.0 vertical.
- `side`: position `(4, 0, 0)`, target `(0, 0, 0)` — backed up to match.
- `spectrum`: unchanged — position `(0, 0, 1.4)`, target `(0, 0, 0)`.
- `rms`: position `(0, -0.5, 1.4)`, target `(0, -0.5, 0)` — y shifts from −0.6 to −0.5.
- **new** `buffer-acf`: position `(0, 0.5, 1.4)`, target `(0, 0.5, 0)`.
- **new** `rms-acf`: position `(0, -1.0, 1.4)`, target `(0, -1.0, 0)`.

**Keybinds** (`presetKeys` map):
```ts
const presetKeys: Record<string, string> = {
  "1": "front",
  "2": "side",
  "3": "spectrum",
  "4": "rms",
  "5": "buffer-acf",
  "6": "rms-acf",
};
```

**`onmessage` handler** gains two lines:
```ts
if (msg.bufferAcf) this.store.set("bufferAcf", msg.bufferAcf);
if (msg.rmsAcf) this.store.set("rmsAcf", msg.rmsAcf);
```

**Render loop** gains two `update()` calls:
```ts
this.bufferAcfLine.update();
this.rmsAcfLine.update();
```

## Data Flow

```
audio in → AudioWorkletProcessor.process()
                ↓
            Dsp.process()  (Rust/WASM)
                ↓
   { waveform, spectrum, rms, bufferAcf, rmsAcf }   ← getter calls into Vec<f32>.clone()
                ↓
   port.postMessage(features, [transferables])
                ↓
   App.onmessage  →  FeatureStore.set(key, Float32Array)
                ↓
   render loop → LineRenderer.update() reads from store via source()
                ↓
   GPU draw call (Three.js WebGPURenderer)
```

The path is identical to v2's; v3 just adds two more keys.

## Error Handling

- **Silent input → div-by-zero in normalization:** Guarded inside `autocorrelate` (output filled with zeros if lag-0 ≤ 0). Verified by test.
- **Constant input → ACF of detrended-then-zero signal:** Mean subtraction drives `rms_detrended` to all zeros, and the normalization guard turns the result into all zeros (not NaN). Verified by test.
- **Worklet/App message field absence:** Existing `if (msg.fieldName)` pattern means any missing field is silently skipped. No new error surfaces are introduced.
- **WebGPU BufferAttribute resize:** Mitigated identically to v1/v2 — store is primed with correctly-sized zero arrays before LineRenderer construction.

## Testing

### Rust unit tests (`crates/dsp/src/lib.rs`)

1. **`acf_of_silence_is_zero`** — feed all-zero input, verify both `buffer_acf()` and `rms_acf()` are all zeros, no NaN.
2. **`acf_zero_lag_is_one_for_nonzero_signal`** — feed any nonzero signal, verify `buffer_acf()[0] == 1.0` (allow `1e-6` tolerance).
3. **`buffer_acf_has_correct_length`** — `assert_eq!(dsp.buffer_acf().len(), 1024)` on a `Dsp::new(2048)`.
4. **`rms_acf_has_correct_length`** — `assert_eq!(dsp.rms_acf().len(), 256)`.
5. **`acf_of_sine_peaks_at_period`** — feed a 1 kHz sine at 48 kHz sample rate (period 48 samples). Find argmax in `buffer_acf()[1..]`. Assert the peak lag is in `47..=49`.
6. **`rms_acf_constant_input_is_zero`** — call `process()` repeatedly with the same constant signal until rms_history fills (≥ 512 calls of size 4, or one call of size ≥ 512). Assert `rms_acf()` is all zeros (mean-subtracted constant is zero, ACF of zero is zero by the normalization guard).
7. **`autocorrelate_helper_correctness`** — call the free function directly with `[1.0, 2.0, 3.0, 4.0]` and a 3-element output buffer. Hand-compute the expected normalized values and assert with `1e-6` tolerance. Independent of `Dsp` scaffolding.

### TypeScript tests

No new TS tests. The worklet and App changes are mechanical (additive fields, additive LineRenderers, layout-constant tweaks). Existing tests for `LineRenderer`, `LineLayouts`, `CameraRig`, and `FeatureStore` continue to pass and cover the components used.

### Manual acceptance checklist

Run after final commit, before tagging v3.0.0:

- [ ] `cargo test -p dsp` — all tests pass (existing 7 + new 7 = 14).
- [ ] `npm run test` — all existing JS tests pass (20).
- [ ] `npx tsc --noEmit` — clean.
- [ ] `npm run build` — clean.
- [ ] Open the app, click Test 440Hz. Press 5 (buffer-acf): see a clear peak near lag 109 (= 48000/440).
- [ ] Press 6 (rms-acf): for a steady tone, expect ACF near zero everywhere (no rhythmic content); for a 4-on-the-floor track via tab capture, expect a clear peak at the beat lag.
- [ ] Press 1 (front): see all five lines stacked, none overlapping.
- [ ] Press 2/3/4 (other presets): no regression.
- [ ] FPS overlay reports >60 throughout.

## Out of Scope (Deferred to ROADMAP)

- FFT-based ACF migration (Wiener–Khinchin) — listed in ROADMAP "Performance".
- Derived BPM / beat-pulse overlay — explicitly not v3; covered if/when the rms-ACF plot proves the math.
- Reduced FFT hop / 94 Hz updates — listed in ROADMAP "Performance".
- HMR, synth source, dedupe trace — listed in ROADMAP "Developer experience".
- Particles + post-processing — listed in ROADMAP "Visual".
