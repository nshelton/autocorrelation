# DSP Crate Modularization

## Goal

Split `crates/dsp/src/lib.rs` (currently 1799 lines, one `Dsp` struct with ~40 fields and 21 wasm-bindgen methods) into named pipeline stages. Replace the per-buffer wasm-bindgen surface with a single string-keyed buffer registry whose key vocabulary is identical across Rust and JS. Collapse the per-param typed setters into a single `set_param(key, value)`.

## Motivation

Today every output buffer requires hand-wiring in five places:

1. A `Vec<f32>` field on `Dsp`.
2. A getter method on `Dsp` (`pub fn spectrum(&self) -> Vec<f32> { self.spectrum.clone() }`).
3. A `*Len` field on `ConfiguredOutbound` in the worklet.
4. A field on the `features` message + an entry in the transfer list.
5. A `store.set(...)` call in `DebugView.applyConfigured` and another in `applyFeatures`.

Five separate edits, three different naming conventions (`rms_history` → `rmsLen` → `rms` on the wire → `"rms"` in FeatureStore), and the giant `Dsp::process` reads as 100 lines of unnamed pipeline.

After this refactor, adding a buffer is one Rust field + two match arms, all in `buffers.rs`. The JS side learns about it for free via `dsp.buffer_descriptors()`.

## Out of scope

- No DSP algorithm changes. Every existing pipeline stage produces identical numerical output.
- No JS-side renderer changes beyond `applyConfigured`/`applyFeatures` loops. `LineRenderer`, `BeatDebugView`, `CameraRig`, ParamStore, ParamPanel are untouched.
- No new buffers, no new params.

## Architecture

### Module split

```
crates/dsp/src/
├── lib.rs       Dsp struct, wasm-bindgen surface, process() orchestrator
├── buffers.rs   Buffers struct, name → slice lookup (the JS contract)
├── spectrum.rs  windowing + forward FFT + spectrum smoothing + per-band RMS + spectral flux
├── acf.rs       generalized ACF, harmonic enhancement, time-domain autocorrelate
└── beat.rs      candidate picking + phase scoring + TEA + beat outputs
```

### State ownership

```rust
pub struct Dsp {
    buffers: Buffers,
    spectrum: SpectrumState,
    acf: AcfState,
    beat: BeatState,
    dt: f32,
    db_floor: f32,
}
```

Each `*State` lives in its own module file alongside the function that uses it. No more flat sea of fields on `Dsp`.

### Canonical key vocabulary

The 15 buffer keys, used **identically** as Rust struct field names, registry match-arm strings, worklet message field names, and FeatureStore keys:

```
waveform, spectrum, bufferAcf,
rms, rmsLow, rmsMid, rmsHigh,
onset, onsetAcf, onsetAcfEnhanced,
tea, candidates,
beatGrid, beatPulses, beatState
```

The 3 param keys (matching existing ParamStore suffixes):

```
smoothingTauSecs, teaTauSecs, dbFloor
```

camelCase is enforced by `#[allow(non_snake_case)]` on the `Buffers` struct so Rust accepts the JS-style names verbatim. Module file names and method names stay snake_case (Rust convention) — the identity rule applies to **string keys** only, not to method/file identifiers.

### `Buffers` shape

```rust
// buffers.rs
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
    pub fn new(window_size: usize, rms_history_len: usize) -> Self { /* ... */ }

    /// String → slice lookup. Used at the wasm boundary.
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

    /// List of (name, length) for `Dsp::buffer_descriptors`. The order is
    /// stable; JS iterates in this order.
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

Public fields means stages access typed slices directly (`buffers.spectrum.copy_from_slice(...)`). String lookup is only for the wasm boundary. Adding a new buffer = field + 2 match arms, all in this one file.

### Stage interfaces

```rust
// spectrum.rs
pub struct SpectrumState {
    hann: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_buffer: Vec<f32>,
    freq_buffer: Vec<Complex<f32>>,
    mag_scale: f32,
    prev_mag: Vec<f32>,
    smoothing_alpha: f32,
    low_band_bin_end: usize,
    mid_band_bin_end: usize,
    parseval_band_scale: f32,
}

impl SpectrumState {
    pub fn new(window_size: usize, sample_rate: f32) -> Self;

    /// Writes the smoothed normalized [0,1] `spectrum`. Returns
    /// `(low_rms, mid_rms, high_rms, flux)` — three Parseval-correct band-RMS
    /// scalars (caller pushes into history buffers) and the spectral-flux
    /// onset value (consumed by the onset stage).
    pub fn process(
        &mut self,
        input: &[f32],
        spectrum: &mut [f32],
        db_floor: f32,
    ) -> (f32, f32, f32, f32);

    pub fn set_smoothing_tau(&mut self, tau_secs: f32, dt: f32);
}
```

```rust
// acf.rs
pub struct AcfState {
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    time_buf: Vec<f32>,
    freq_buf: Vec<Complex<f32>>,
}

impl AcfState {
    pub fn new(rms_history_len: usize) -> Self;
    pub fn process(&mut self, onset: &[f32], onset_acf: &mut [f32], onset_acf_enhanced: &mut [f32]);
}

// Free functions, used directly by Dsp::process and tests.
pub fn autocorrelate(input: &[f32], output: &mut [f32]);
```

```rust
// beat.rs
pub struct BeatState {
    tau_min: usize, tau_max: usize,
    cand_scratch: Vec<(usize, f32)>,
    pulse_x: [f32; MAX_PEAKS],
    pulse_v: [f32; MAX_PEAKS],
    pulse_phi: [f32; MAX_PEAKS],
    pulse_score: [f32; MAX_PEAKS],
    period_inst: f32, phase_inst: f32, score_inst: f32,
    tea_alpha: f32,
    tau_smoothed: f32, phase_smoothed: f32,
    beat_position: f32,
}

impl BeatState {
    pub fn new(rms_history_len: usize, dt: f32) -> Self;
    pub fn process(
        &mut self,
        onset: &[f32], onset_acf_enhanced: &[f32],
        candidates: &mut [f32], tea: &mut [f32],
        beat_grid: &mut [f32], beat_state: &mut [f32], beat_pulses: &mut [f32],
        dt: f32,
    );
    pub fn set_tea_tau(&mut self, tau_secs: f32, dt: f32);
}
```

The internal helpers `pick_candidates`, `score_candidates`, `update_tea`, `write_beat_outputs`, `update_beat_pulses` become private methods or free functions inside `beat.rs`. `score_phase_for_tau` moves into `beat.rs`; `compute_harmonic_enhanced` moves into `acf.rs`.

### `Dsp::process`

```rust
pub fn process(&mut self, input: &[f32]) {
    let n = input.len().min(self.buffers.waveform.len());
    self.buffers.waveform[..n].copy_from_slice(&input[..n]);

    // Time-domain RMS history.
    push_history(&mut self.buffers.rms, full_rms(input));

    // FFT → spectrum + per-band RMS values + spectral flux.
    let (low_rms, mid_rms, high_rms, flux) = self.spectrum.process(
        input,
        &mut self.buffers.spectrum,
        self.db_floor,
    );
    push_history(&mut self.buffers.rmsLow, low_rms);
    push_history(&mut self.buffers.rmsMid, mid_rms);
    push_history(&mut self.buffers.rmsHigh, high_rms);
    push_history(&mut self.buffers.onset, flux);

    // ACF + harmonic enhancement.
    self.acf.process(
        &self.buffers.onset,
        &mut self.buffers.onsetAcf,
        &mut self.buffers.onsetAcfEnhanced,
    );

    // Beat tracker (candidates → score → TEA → beat outputs).
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

    // Time-domain ACF of the waveform.
    acf::autocorrelate(&self.buffers.waveform, &mut self.buffers.bufferAcf);
}
```

(`SpectrumState::process` returns `(low_rms, mid_rms, high_rms, flux)` so the band-RMS values can be pushed into the histories owned by `Buffers`. Histories are not part of the spectrum state.)

`push_history` is a tiny free function in `lib.rs` (or `buffers.rs`):
```rust
fn push_history(buf: &mut [f32], value: f32) {
    if buf.is_empty() { return; }
    buf.copy_within(1.., 0);
    let last = buf.len() - 1;
    buf[last] = value;
}
```

### wasm-bindgen surface

```rust
#[wasm_bindgen]
impl Dsp {
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize, sample_rate: f32, hop_size: usize, rms_history_len: usize) -> Dsp;

    pub fn process(&mut self, input: &[f32]);

    /// Copy of the named buffer's current contents. Returns empty Vec for
    /// unknown names (callers should rely on `buffer_descriptors` for the list).
    pub fn get_buffer(&self, name: &str) -> Vec<f32>;

    /// `[{name: "spectrum", length: 1024}, ...]` — stable order matching
    /// `Buffers::descriptors`. Called once per `configured` from the worklet.
    pub fn buffer_descriptors(&self) -> Vec<JsValue>;

    /// Set a tunable param. Unknown keys are silently ignored.
    /// Recognized keys: "smoothingTauSecs", "teaTauSecs", "dbFloor".
    pub fn set_param(&mut self, key: &str, value: f32);
}
```

Five public methods, down from 21. The 15 typed buffer getters and the 3 typed param setters all collapse here.

wasm-bindgen translates Rust `snake_case` method names to JS `camelCase` automatically. JS calls are `dsp.getBuffer("spectrum")`, `dsp.bufferDescriptors()`, `dsp.setParam("teaTauSecs", v)`. The Rust source stays idiomatic; only the **string keys** passed as arguments need to be identical across sides.

### Worklet ↔ main message protocol

```ts
// configured: collapses 11 *Len fields into one descriptors list.
type ConfiguredOutbound = {
  type: "configured";
  buffers: { name: string; length: number }[];
};

// features: collapses 15 named fields into one dict.
type FeaturesOutbound = {
  type: "features";
  buffers: { [name: string]: Float32Array };
};
```

The `sync` message and HMR semantics described in `CLAUDE.md` are unchanged — `lastConfigured` still caches the most recent payload.

### JS-side knock-on

**`src/audio/dsp-worklet.ts`:** `applyConfigure` calls `dsp.buffer_descriptors()` once and caches it. `process()` loops the cached list, calls `dsp.get_buffer(name)`, wraps each in `Float32Array`, posts one `{ type: "features", buffers }` dict with all bufs in the transfer list. The 15 hand-written getter calls + 15 transfer-list entries collapse to a loop.

**`src/render/DebugView.ts`:** `applyConfigured(cfg)` walks `cfg.buffers` and calls `store.set(name, new Float32Array(length))` for each. `applyFeatures(msg)` walks `msg.buffers` and calls `store.set(name, buf)` for each. Renderers continue reading `store.get("spectrum")` etc. unchanged.

**`src/render/BeatDebugView.ts`:** Same pattern — drop the per-key `if (msg.beatGrid) ...` chain in favor of letting `DebugView`'s loop set everything; `BeatDebugView` only reads from the store.

**`src/params/WorkletBridge.ts`:** Hot-key forwarding becomes `worklet.postMessage({ type: "param", key, value })` and the worklet calls `dsp.set_param(key, value)` directly. The `applyHotKey` switch collapses (or moves entirely into a one-line forward).

### Tests

- Free functions (`acf::compute_gen_acf`, `acf::autocorrelate`, `acf::compute_harmonic_enhanced`, `beat::score_phase_for_tau`) become module-public — testable directly without constructing `Dsp`.
- `BeatState::process` is testable in isolation: construct a `BeatState`, allocate the output `Vec<f32>`s the right size, call `process` with a synthetic `onset` / `onset_acf_enhanced`. The `test_set_*` / `test_run_*` poke methods on `Dsp` are deleted.
- Existing integration tests in `lib.rs::tests` switch from typed getters (`dsp.spectrum()`, `dsp.beat_grid()`) to `dsp.get_buffer("spectrum")` etc. Numerical assertions unchanged.
- Add one test: `dsp.buffer_descriptors()` returns 15 entries with the expected names + lengths after construction.

## Migration / rollout

This is a single-PR refactor. Every consumer (worklet, `DebugView`, `BeatDebugView`, `WorkletBridge`, JS tests, Rust tests) updates in the same change because the wasm-bindgen surface is replaced wholesale — there is no ergonomic way to ship the Rust side independently. `npm run wasm` rebuilds the wasm-pkg; `npm test` and `cargo test -p dsp` both must pass before merge.

## Risks

- **Field name collisions with reserved Rust keywords.** None of the 15 keys collide (`type`, `match`, `let` etc. are not in the list).
- **`#[allow(non_snake_case)]` lint scope.** Applied to `Buffers` struct only. Other modules keep idiomatic snake_case.
- **`get_buffer(name) -> Vec<f32>` allocates.** Same allocation cost as today's typed getters — wasm-bindgen `Vec<f32>` returns already serialize as a JS array copy. No regression. (A future optimization could expose pointer + length and let JS read wasm memory directly, but that's deliberately out of scope.)
- **Order stability of `buffer_descriptors`.** Worklet caches the list once after `configured`; if the order changed mid-run we'd post the wrong sizes. The `descriptors()` impl returns a fixed `vec![...]` literal, so order is compile-time stable.

## Acceptance criteria

- `crates/dsp/src/lib.rs` is under 250 lines; the four other modules are each under 500 lines.
- The 15 typed buffer getters and 3 typed param setters on `Dsp` are deleted.
- `cargo test -p dsp` and `npm test` pass with all existing assertions unchanged in spirit (only API call sites updated).
- `npm run dev`: every existing visualization (waveform, spectrum, bands, RMS, ACF, beat grid, beat pulses, peak markers) renders identically, all camera presets work, all keyboard shortcuts work, ParamPanel hot keys still adjust live without a worklet rebuild.
- HMR still preserves DSP state across an App rebuild (the `sync` mechanism is unaffected).
- Adding a new buffer end-to-end (Rust struct field → 2 match arms → JS renders it) takes edits in only `buffers.rs` and the consumer's renderer file. No edits to `dsp-worklet.ts`, `DebugView.applyConfigured`, or `WorkletBridge`.
