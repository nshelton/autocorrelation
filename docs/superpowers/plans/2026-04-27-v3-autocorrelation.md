# v3 Autocorrelation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add buffer autocorrelation and RMS-envelope autocorrelation as two new line strips in the existing audio visualizer.

**Architecture:** Direct time-domain ACF computed in the Rust DSP crate inside `process()`. The worklet posts the two new arrays alongside existing features in one message. `App.ts` renders them via two new `LineRenderer` instances at fixed y-positions, with two new camera presets (keys 5 and 6).

**Tech Stack:** Rust (realfft already present, no new deps), TypeScript, Three.js WebGPURenderer, Vite, Vitest, wasm-pack.

**Spec:** `docs/superpowers/specs/2026-04-27-v3-autocorrelation-design.md`

---

## File Map

- **Modify:** `crates/dsp/src/lib.rs` — new free function, new fields, new getters, new computation in `process()`, new tests.
- **Modify:** `src/audio/dsp-worklet.ts` — call new getters, post two new fields with transferables.
- **Modify:** `src/App.ts` — prime store, adjust existing line layouts, two new `LineRenderer`s, two new camera presets, extend keybinds, extend `onmessage`, extend render loop.
- **Generated (gitignored):** `src/wasm-pkg/*` — regenerated via `npm run wasm` after Rust changes; not committed.

---

## Task 1: `autocorrelate` free function (TDD)

**Files:**
- Modify: `crates/dsp/src/lib.rs` (add free function inside the file, plus one test in the existing `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Add this test inside `mod tests` in `crates/dsp/src/lib.rs`, after the existing `loud_sine_produces_a_peak` test:

```rust
#[test]
fn autocorrelate_helper_correctness() {
    // Hand-computed for input [1, 2, 3, 4] with output length 3:
    //   raw[0] = 1*1 + 2*2 + 3*3 + 4*4 = 30
    //   raw[1] = 1*2 + 2*3 + 3*4       = 20
    //   raw[2] = 1*3 + 2*4             = 11
    // Normalized by raw[0]=30: [1.0, 20/30, 11/30].
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 3];
    autocorrelate(&input, &mut output);
    assert!((output[0] - 1.0).abs() < 1e-6, "got {}", output[0]);
    assert!((output[1] - 20.0 / 30.0).abs() < 1e-6, "got {}", output[1]);
    assert!((output[2] - 11.0 / 30.0).abs() < 1e-6, "got {}", output[2]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsp autocorrelate_helper_correctness`
Expected: FAIL — compilation error, "cannot find function `autocorrelate` in this scope".

- [ ] **Step 3: Implement the function**

Add this free function to `crates/dsp/src/lib.rs` at the bottom of the file (after `impl Dsp`, before `#[cfg(test)] mod tests`):

```rust
/// Direct time-domain autocorrelation, normalized so output[0] == 1.0
/// for any nonzero input. For all-zero input the output is filled with
/// zeros (no NaN from division by zero). The caller chooses how many
/// lags to compute via the length of `output`.
fn autocorrelate(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    for k in 0..output.len() {
        let mut sum = 0.0f32;
        if k < n {
            for i in 0..(n - k) {
                sum += input[i] * input[i + k];
            }
        }
        output[k] = sum;
    }
    let zero = output[0];
    if zero > 0.0 {
        for v in output.iter_mut() {
            *v /= zero;
        }
    } else {
        output.fill(0.0);
    }
}
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `cargo test -p dsp autocorrelate_helper_correctness`
Expected: PASS.

- [ ] **Step 5: Run full Rust test suite to verify no regression**

Run: `cargo test -p dsp`
Expected: 8 tests pass (7 existing + 1 new).

- [ ] **Step 6: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): add autocorrelate helper with normalization"
```

---

## Task 2: `buffer_acf` field, computation, getter

**Files:**
- Modify: `crates/dsp/src/lib.rs` (add field, update `Dsp::new`, update `process()`, add getter, add three tests)

- [ ] **Step 1: Write the first failing test (length)**

Add this test inside `mod tests` in `crates/dsp/src/lib.rs`, after the new `autocorrelate_helper_correctness` test:

```rust
#[test]
fn buffer_acf_has_correct_length() {
    let dsp = Dsp::new(2048);
    assert_eq!(dsp.buffer_acf().len(), 1024);
}
```

- [ ] **Step 2: Run, verify it fails**

Run: `cargo test -p dsp buffer_acf_has_correct_length`
Expected: FAIL — compilation error, "no method named `buffer_acf` found for struct `Dsp`".

- [ ] **Step 3: Add the field, getter, and initializer**

In `crates/dsp/src/lib.rs`, modify the `Dsp` struct definition to add a new field after `rms_history`:

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
    buffer_acf: Vec<f32>,
}
```

In `Dsp::new(window_size)`, add `buffer_acf` to the struct literal at the end (after `rms_history`):

```rust
        Dsp {
            waveform: vec![0.0; window_size],
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            spectrum,
            hann,
            mag_scale,
            rms_history: vec![0.0; RMS_HISTORY_LEN],
            buffer_acf: vec![0.0; window_size / 2],
        }
```

In the `#[wasm_bindgen] impl Dsp` block, add a getter after the existing `spectrum()` getter:

```rust
    pub fn buffer_acf(&self) -> Vec<f32> {
        self.buffer_acf.clone()
    }
```

- [ ] **Step 4: Run, verify the length test passes**

Run: `cargo test -p dsp buffer_acf_has_correct_length`
Expected: PASS.

- [ ] **Step 5: Write the second failing test (zero-lag = 1.0)**

Add this test inside `mod tests`:

```rust
#[test]
fn buffer_acf_zero_lag_is_one_for_nonzero_signal() {
    let mut dsp = Dsp::new(2048);
    let signal: Vec<f32> = (0..2048)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    dsp.process(&signal);
    let acf = dsp.buffer_acf();
    assert!((acf[0] - 1.0).abs() < 1e-6, "got {}", acf[0]);
}
```

- [ ] **Step 6: Run, verify it fails**

Run: `cargo test -p dsp buffer_acf_zero_lag_is_one_for_nonzero_signal`
Expected: FAIL — `acf[0]` is `0.0`, since nothing populates `buffer_acf` yet.

- [ ] **Step 7: Wire the computation in `process()`**

In `crates/dsp/src/lib.rs`, edit the `process` method. After the RMS-history shift block (the lines that copy `rms_history` and assign `rms_history[last] = rms`) and before the Hann window block, add this single line:

```rust
        autocorrelate(&self.waveform, &mut self.buffer_acf);
```

The block should now read:

```rust
        // Shift left and append newest at the end (oldest at index 0)
        self.rms_history.copy_within(1.., 0);
        let last = self.rms_history.len() - 1;
        self.rms_history[last] = rms;

        autocorrelate(&self.waveform, &mut self.buffer_acf);

        // Apply Hann window
        for i in 0..n {
            self.fft_buffer[i] = input[i] * self.hann[i];
        }
```

- [ ] **Step 8: Run, verify the zero-lag test passes**

Run: `cargo test -p dsp buffer_acf_zero_lag_is_one_for_nonzero_signal`
Expected: PASS.

- [ ] **Step 9: Write the third failing test (sine peak at period)**

Add this test inside `mod tests`:

```rust
#[test]
fn buffer_acf_of_sine_peaks_at_period() {
    let mut dsp = Dsp::new(2048);
    let sr = 48000.0_f32;
    let freq = 1000.0_f32;
    // Period at this sr/freq is exactly 48 samples.
    let signal: Vec<f32> = (0..2048)
        .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
        .collect();
    dsp.process(&signal);
    let acf = dsp.buffer_acf();
    // Skip lag 0 (always 1.0); find argmax in the rest.
    let (argmax, _) = acf[1..]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let lag = argmax + 1; // re-offset because we sliced acf[1..]
    assert!(
        (47..=49).contains(&lag),
        "expected peak near lag 48, got {}",
        lag
    );
}
```

- [ ] **Step 10: Run, verify it passes**

Run: `cargo test -p dsp buffer_acf_of_sine_peaks_at_period`
Expected: PASS.

- [ ] **Step 11: Run full Rust test suite to verify no regression**

Run: `cargo test -p dsp`
Expected: 11 tests pass (7 existing + 1 from Task 1 + 3 new).

- [ ] **Step 12: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): buffer autocorrelation field, getter, and computation"
```

---

## Task 3: `rms_acf` field with detrending

**Files:**
- Modify: `crates/dsp/src/lib.rs` (add `RMS_ACF_LEN` const, two new fields, update `Dsp::new`, update `process()`, add getter, add three tests)

- [ ] **Step 1: Write the first failing test (length)**

Add this test inside `mod tests` in `crates/dsp/src/lib.rs`, after the new `buffer_acf_of_sine_peaks_at_period` test:

```rust
#[test]
fn rms_acf_has_correct_length() {
    let dsp = Dsp::new(2048);
    assert_eq!(dsp.rms_acf().len(), 256);
}
```

- [ ] **Step 2: Run, verify it fails**

Run: `cargo test -p dsp rms_acf_has_correct_length`
Expected: FAIL — compilation error, "no method named `rms_acf` found for struct `Dsp`".

- [ ] **Step 3: Add constant, fields, getter, and initializer**

In `crates/dsp/src/lib.rs`, add the new constant after the existing `RMS_HISTORY_LEN: usize = 512;`:

```rust
const RMS_ACF_LEN: usize = RMS_HISTORY_LEN / 2;
```

Add two new fields to the `Dsp` struct, after `buffer_acf`:

```rust
    buffer_acf: Vec<f32>,
    rms_acf: Vec<f32>,
    rms_detrended: Vec<f32>,
```

In `Dsp::new(window_size)`, extend the struct literal:

```rust
        Dsp {
            waveform: vec![0.0; window_size],
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            spectrum,
            hann,
            mag_scale,
            rms_history: vec![0.0; RMS_HISTORY_LEN],
            buffer_acf: vec![0.0; window_size / 2],
            rms_acf: vec![0.0; RMS_ACF_LEN],
            rms_detrended: vec![0.0; RMS_HISTORY_LEN],
        }
```

In the `impl Dsp` block, add a getter after the new `buffer_acf()`:

```rust
    pub fn rms_acf(&self) -> Vec<f32> {
        self.rms_acf.clone()
    }
```

- [ ] **Step 4: Run, verify the length test passes**

Run: `cargo test -p dsp rms_acf_has_correct_length`
Expected: PASS.

- [ ] **Step 5: Write the second failing test (silence — both ACFs)**

Add this test inside `mod tests`. It covers both buffer and rms ACFs from one silent input, exercising the divide-by-zero guard for both:

```rust
#[test]
fn acf_of_silence_is_zero() {
    let mut dsp = Dsp::new(2048);
    dsp.process(&vec![0.0_f32; 2048]);
    for &v in dsp.buffer_acf().iter() {
        assert_eq!(v, 0.0);
    }
    for &v in dsp.rms_acf().iter() {
        assert_eq!(v, 0.0);
    }
}
```

- [ ] **Step 6: Run, verify it fails**

Run: `cargo test -p dsp acf_of_silence_is_zero`
Expected: FAIL — `rms_acf` is currently all zeros only because we never compute it; but the assertion will pass on `buffer_acf` (already wired in Task 2 with the silence guard) and pass on `rms_acf` (still all zeros from initialization). So this test may PASS even before computation is wired.

If the test passes at this step, that's fine — it locks in the silence behavior before adding the computation. Move to Step 7.

- [ ] **Step 7: Wire the detrending and ACF computation in `process()`**

In `crates/dsp/src/lib.rs`, edit the `process` method. The block added in Task 2 (`autocorrelate(&self.waveform, &mut self.buffer_acf);`) should be expanded to also handle the rms_acf. Replace that single line with this block:

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

- [ ] **Step 8: Re-run silence test to verify it still passes**

Run: `cargo test -p dsp acf_of_silence_is_zero`
Expected: PASS — silent input → mean=0 → detrended=zeros → ACF[0]=0 → guard fills with zeros.

- [ ] **Step 9: Write the third failing test (constant input → zero rms ACF)**

Add this test inside `mod tests`:

```rust
#[test]
fn rms_acf_constant_input_is_zero() {
    // Fill rms_history with a constant rms value, then verify the
    // detrended (mean-subtracted) ACF is zero everywhere.
    let mut dsp = Dsp::new(8);
    let constant = vec![0.5_f32; 8];
    // RMS of [0.5; 8] is 0.5; need >= RMS_HISTORY_LEN (512) calls to fully fill.
    for _ in 0..512 {
        dsp.process(&constant);
    }
    let acf = dsp.rms_acf();
    assert_eq!(acf.len(), 256);
    for &v in &acf {
        assert!(v.abs() < 1e-5, "expected near-zero ACF for constant rms, got {}", v);
    }
}
```

- [ ] **Step 10: Run, verify it passes**

Run: `cargo test -p dsp rms_acf_constant_input_is_zero`
Expected: PASS.

- [ ] **Step 11: Run full Rust test suite to verify no regression**

Run: `cargo test -p dsp`
Expected: 14 tests pass (7 existing + 1 from Task 1 + 3 from Task 2 + 3 new).

- [ ] **Step 12: Commit**

```bash
git add crates/dsp/src/lib.rs
git commit -m "feat(dsp): rms-envelope autocorrelation with detrending"
```

---

## Task 4: Worklet posts new fields

**Files:**
- Rebuild: `src/wasm-pkg/*` (regenerated, gitignored)
- Modify: `src/audio/dsp-worklet.ts:51-58`

- [ ] **Step 1: Rebuild the WASM artifact**

Run: `npm run wasm`
Expected: Output ends with `[INFO]: ✨ Done in N.NNs`. Verify the new bindings are in place:

Run: `grep -E "buffer_acf|rms_acf" src/wasm-pkg/dsp.d.ts`
Expected: Two lines, one for each new method on `class Dsp`.

- [ ] **Step 2: Edit the worklet to read and post the new fields**

In `src/audio/dsp-worklet.ts`, replace the message-posting block inside the `while` loop at lines 50-60. The current block reads:

```ts
    while (this.hopCounter >= HOP_SIZE) {
      this.dsp.process(this.window);
      const wf = new Float32Array(this.dsp.waveform());
      const sp = new Float32Array(this.dsp.spectrum());
      const rms = new Float32Array(this.dsp.rms_history());
      this.port.postMessage(
        { type: "features", waveform: wf, spectrum: sp, rms },
        [wf.buffer, sp.buffer, rms.buffer],
      );
      this.hopCounter -= HOP_SIZE;
    }
```

Replace with:

```ts
    while (this.hopCounter >= HOP_SIZE) {
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
      this.hopCounter -= HOP_SIZE;
    }
```

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Run JS tests to verify no regression**

Run: `npm run test`
Expected: All 20 existing tests pass.

- [ ] **Step 5: Build to verify the worker bundle**

Run: `npm run build`
Expected: Build completes without errors. The dist output should include a worklet bundle.

- [ ] **Step 6: Commit**

```bash
git add src/audio/dsp-worklet.ts
git commit -m "feat(audio): worklet posts buffer + rms autocorrelation"
```

---

## Task 5: App.ts integration — store, layouts, LineRenderers, render loop, onmessage

**Files:**
- Modify: `src/App.ts` (multiple locations — store priming, layout y-coords/heights, two new LineRenderer constructions, onmessage handler, render loop)

- [ ] **Step 1: Prime the store with two new zero-filled buffers**

In `src/App.ts`, find the store-priming block (currently lines 46-48):

```ts
    this.store.set("waveform", new Float32Array(2048));
    this.store.set("spectrum", new Float32Array(1024));
    this.store.set("rms", new Float32Array(512));
```

Replace with:

```ts
    this.store.set("waveform", new Float32Array(2048));
    this.store.set("spectrum", new Float32Array(1024));
    this.store.set("rms", new Float32Array(512));
    this.store.set("bufferAcf", new Float32Array(1024));
    this.store.set("rmsAcf", new Float32Array(256));
```

- [ ] **Step 2: Adjust existing line layouts to fit five lines vertically**

In `src/App.ts`, the three existing `LineRenderer` constructions are around lines 50-69. Update y-baselines and heights so all five lines fit between y=−1.0 and y=+1.0 with height 0.4 each.

Current:
```ts
    this.waveformLine = new LineRenderer({
      source: () => this.store.get("waveform"),
      layout: linearLayout(0.6, 0.5),
      color: 0x66ffcc,
    });
    scene.add(this.waveformLine.object3d);

    this.spectrumLine = new LineRenderer({
      source: () => this.store.get("spectrum"),
      layout: logSpectrumLayout(0.0, 0.5),
      color: 0xffaa66,
    });
    scene.add(this.spectrumLine.object3d);

    this.rmsLine = new LineRenderer({
      source: () => this.store.get("rms"),
      layout: linearLayout(-0.6, 0.5),
      color: 0xffffff,
    });
    scene.add(this.rmsLine.object3d);
```

Replace with:

```ts
    this.waveformLine = new LineRenderer({
      source: () => this.store.get("waveform"),
      layout: linearLayout(1.0, 0.4),
      color: 0x66ffcc,
    });
    scene.add(this.waveformLine.object3d);

    this.bufferAcfLine = new LineRenderer({
      source: () => this.store.get("bufferAcf"),
      layout: linearLayout(0.5, 0.4),
      color: 0xcc99ff,
    });
    scene.add(this.bufferAcfLine.object3d);

    this.spectrumLine = new LineRenderer({
      source: () => this.store.get("spectrum"),
      layout: logSpectrumLayout(0.0, 0.4),
      color: 0xffaa66,
    });
    scene.add(this.spectrumLine.object3d);

    this.rmsLine = new LineRenderer({
      source: () => this.store.get("rms"),
      layout: linearLayout(-0.5, 0.4),
      color: 0xffffff,
    });
    scene.add(this.rmsLine.object3d);

    this.rmsAcfLine = new LineRenderer({
      source: () => this.store.get("rmsAcf"),
      layout: linearLayout(-1.0, 0.4),
      color: 0xff99cc,
    });
    scene.add(this.rmsAcfLine.object3d);
```

- [ ] **Step 3: Add the two new fields to the App class**

In `src/App.ts`, find the class field declarations (currently around lines 13-19):

```ts
export class App {
  private rig!: CameraRig;
  private waveformLine!: LineRenderer;
  private spectrumLine!: LineRenderer;
  private rmsLine!: LineRenderer;
  private store = new FeatureStore();
  private last = 0;
  private fps = new FpsOverlay();
```

Add two new line fields:

```ts
export class App {
  private rig!: CameraRig;
  private waveformLine!: LineRenderer;
  private spectrumLine!: LineRenderer;
  private rmsLine!: LineRenderer;
  private bufferAcfLine!: LineRenderer;
  private rmsAcfLine!: LineRenderer;
  private store = new FeatureStore();
  private last = 0;
  private fps = new FpsOverlay();
```

- [ ] **Step 4: Extend the `onmessage` handler**

In `src/App.ts`, find the `onmessage` handler (currently around lines 132-143):

```ts
    node.port.onmessage = (e) => {
      const msg = e.data as {
        type: string;
        waveform?: Float32Array;
        spectrum?: Float32Array;
        rms?: Float32Array;
      };
      if (msg.type !== "features") return;
      if (msg.waveform) this.store.set("waveform", msg.waveform);
      if (msg.spectrum) this.store.set("spectrum", msg.spectrum);
      if (msg.rms) this.store.set("rms", msg.rms);
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
      };
      if (msg.type !== "features") return;
      if (msg.waveform) this.store.set("waveform", msg.waveform);
      if (msg.spectrum) this.store.set("spectrum", msg.spectrum);
      if (msg.rms) this.store.set("rms", msg.rms);
      if (msg.bufferAcf) this.store.set("bufferAcf", msg.bufferAcf);
      if (msg.rmsAcf) this.store.set("rmsAcf", msg.rmsAcf);
    };
```

- [ ] **Step 5: Extend the render loop**

In `src/App.ts`, find the render loop (currently around lines 145-156):

```ts
    const loop = (now: number) => {
      this.fps.begin();
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;
      this.rig.update(dt);
      this.waveformLine.update();
      this.spectrumLine.update();
      this.rmsLine.update();
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
      this.rmsLine.update();
      this.rmsAcfLine.update();
      renderer.render(scene, camera);
      this.fps.end();
      requestAnimationFrame(loop);
    };
```

- [ ] **Step 6: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 7: Run JS tests to verify no regression**

Run: `npm run test`
Expected: All 20 existing tests pass.

- [ ] **Step 8: Build to verify**

Run: `npm run build`
Expected: Build completes without errors.

- [ ] **Step 9: Commit**

```bash
git add src/App.ts
git commit -m "feat(app): five-line layout with buffer + rms ACF renderers"
```

---

## Task 6: Camera presets and keybinds

**Files:**
- Modify: `src/App.ts` (`addPreset` calls in `start()`, `presetKeys` map)

- [ ] **Step 1: Update existing presets and add two new ones**

In `src/App.ts`, find the camera-preset block (currently around lines 28-44):

```ts
    this.rig.addPreset("front", {
      position: new Vector3(0, 0, 3),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("side", {
      position: new Vector3(3, 1, 0),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("spectrum", {
      position: new Vector3(0, 0, 1.4),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("rms", {
      position: new Vector3(0, -0.6, 1.4),
      target: new Vector3(0, -0.6, 0),
    });
    await this.rig.goTo("front", { duration: 0 });
```

Replace with:

```ts
    this.rig.addPreset("front", {
      position: new Vector3(0, 0, 4),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("side", {
      position: new Vector3(4, 0, 0),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("spectrum", {
      position: new Vector3(0, 0, 1.4),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("rms", {
      position: new Vector3(0, -0.5, 1.4),
      target: new Vector3(0, -0.5, 0),
    });
    this.rig.addPreset("buffer-acf", {
      position: new Vector3(0, 0.5, 1.4),
      target: new Vector3(0, 0.5, 0),
    });
    this.rig.addPreset("rms-acf", {
      position: new Vector3(0, -1.0, 1.4),
      target: new Vector3(0, -1.0, 0),
    });
    await this.rig.goTo("front", { duration: 0 });
```

- [ ] **Step 2: Extend `presetKeys`**

In `src/App.ts`, find the `presetKeys` map (currently around lines 74-79):

```ts
    const presetKeys: Record<string, string> = {
      "1": "front",
      "2": "side",
      "3": "spectrum",
      "4": "rms",
    };
```

Replace with:

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

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Build**

Run: `npm run build`
Expected: Build completes without errors.

- [ ] **Step 5: Commit**

```bash
git add src/App.ts
git commit -m "feat(camera): buffer-acf and rms-acf presets (keys 5,6)"
```

---

## Task 7: Final acceptance and v3.0.0 tag

**Files:** No code changes; verification only.

- [ ] **Step 1: Run full Rust test suite**

Run: `cargo test -p dsp`
Expected: 14 tests pass, 0 failed.

- [ ] **Step 2: Run full JS test suite**

Run: `npm run test`
Expected: 20 tests pass.

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Production build**

Run: `npm run build`
Expected: Build completes without errors. No new console warnings beyond the existing pre-v3 warnings.

- [ ] **Step 5: Manual visual acceptance**

Run: `npm run dev` and open the printed URL.

Click **Test 440Hz**. Verify each:

- Press `1` (front): all five lines visible, top-to-bottom: cyan waveform, purple buffer-ACF, orange spectrum, white rms, pink rms-ACF. None overlapping.
- Press `5` (buffer-ACF): line is centered. With a 440 Hz tone at 48 kHz sample rate, expect a clear positive peak near lag 109 (= 48000 / 440 ≈ 109.1).
- Press `6` (rms-ACF): for a steady tone, expect ACF near zero everywhere (no rhythmic content). Line should look mostly flat at the baseline.
- Press `1` again: lines stack as expected.
- Press `2` (side), `3` (spectrum), `4` (rms): no regression — each shows the expected view.
- FPS overlay reports >60 throughout.

If a tab-audio source with a steady beat (e.g., a 4-on-the-floor track) is convenient, click **Tab Audio**, share a tab, and verify rms-ACF (key 6) shows a clear peak at the beat lag.

- [ ] **Step 6: Update ROADMAP.md to move v3 from "In progress" to "Shipped"**

Edit `/Users/nshelton/autocorrelation/ROADMAP.md`. Move the `### v3 — Autocorrelation` section from under `## In progress` to under `## Shipped` as the first bullet (above v2). Format:

```markdown
- **v3** (tag `v3.0.0`): buffer autocorrelation + RMS-envelope autocorrelation as two new line strips, two new camera presets (keys 5 and 6)
```

The `## In progress` section should now be empty (or contain whatever the user decides next — leave it empty for now).

- [ ] **Step 7: Commit the ROADMAP update**

```bash
git add ROADMAP.md
git commit -m "docs: mark v3 autocorrelation as shipped"
```

- [ ] **Step 8: Tag v3.0.0**

```bash
git tag -a v3.0.0 -m "v3.0.0 — buffer + RMS-envelope autocorrelation"
```

- [ ] **Step 9: Verify the tag**

Run: `git tag --list 'v*'`
Expected: `v1.0.0`, `v2.0.0`, `v3.0.0` listed.

Run: `git log --oneline -5`
Expected: The most recent commits include the seven new ones from Tasks 1–7.
