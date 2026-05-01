# RMS Event Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-render-frame audio event queue that fires when any of the four RMS channels (`rms`, `rmsLow`, `rmsMid`, `rmsHigh`) crosses 0.9 on a rising edge. Events flow worklet → main thread on the existing `features` message; main thread accumulates them in `FeatureStore` and clears once per RAF tick.

**Architecture:** New Rust module `crates/dsp/src/events.rs` owns four independent rising-edge state machines (one per channel), each with a refractory window of one `tau_smoothed`-worth of frames (fixed-ms fallback when tau is NaN). `Dsp::process` invokes the detector after `self.beat.process(...)`; the worklet drains `dsp.take_events()` each hop and ships a flat stride-4 Float32Array `[channel, frameLo, frameHi, value, ...]` alongside the existing buffers. Main thread decodes into `AudioEvent[]` and pushes into `FeatureStore`; render loop clears after `view.update()`.

**Tech Stack:** Rust + wasm-bindgen (existing crate, no new deps), TypeScript AudioWorklet (existing pattern), Vitest + happy-dom for TS unit tests, `cargo test -p dsp` for Rust. Spec at `docs/superpowers/specs/2026-05-01-rms-event-queue-design.md`.

---

## Task 1: Rust — `EventDetector` module with state machine

Build the detector in isolation, fully unit-tested, before wiring it into `Dsp`. Six tests cover the six state-machine behaviors. The module exports `Event`, `EventDetector`, `EVENT_THRESHOLD`, and four channel-ID constants — nothing else escapes.

**Files:**
- Create: `crates/dsp/src/events.rs`

- [ ] **Step 1: Create the file with the failing tests**

Create `crates/dsp/src/events.rs` with this exact content:

```rust
//! Rising-edge RMS event detector. Per-frame events fire when a channel's
//! latest RMS sample crosses `EVENT_THRESHOLD` from below; subsequent fires
//! on the same channel are blocked until (a) the channel falls back below
//! threshold AND (b) `refractory` frames have elapsed since the last event
//! on that channel. `refractory` = `tau_smoothed` rounded to frames, with a
//! fixed-ms fallback when tau is NaN.

/// RMS value (post-autogain, normalized [0,1]) at and above which a rising-
/// edge event fires. Compile-time constant by design — events are a coarse
/// "loud peak" notification, not a tunable signal.
pub const EVENT_THRESHOLD: f32 = 0.9;

/// Fallback refractory in milliseconds when `tau_smoothed` is NaN. Converted
/// to frames in `EventDetector::new` from the actual sr/hop. ~100 ms at
/// default sr=48000, hop=1024 → ≈5 frames.
const REFRACTORY_FALLBACK_MS: f32 = 100.0;

/// Maximum events buffered in `pending` between `take_pending` calls. Worst
/// realistic case is 4 (one per channel per hop), so 64 is paranoid but cheap.
const PENDING_CAP: usize = 64;

pub const CHAN_RMS: u8 = 0;
pub const CHAN_RMS_LOW: u8 = 1;
pub const CHAN_RMS_MID: u8 = 2;
pub const CHAN_RMS_HIGH: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    pub channel: u8,
    pub frame: u64,
    pub value: f32,
}

pub struct EventDetector {
    above: [bool; 4],
    last_event_frame: [Option<u64>; 4],
    frame: u64,
    refractory_fallback_frames: usize,
    pending: Vec<Event>,
}

impl EventDetector {
    pub fn new(sample_rate: f32, hop_size: usize) -> Self {
        let hop = hop_size.max(1) as f32;
        let frames = (REFRACTORY_FALLBACK_MS * sample_rate / 1000.0 / hop).round();
        let refractory_fallback_frames = frames.max(1.0) as usize;
        Self {
            above: [false; 4],
            last_event_frame: [None; 4],
            frame: 0,
            refractory_fallback_frames,
            pending: Vec::with_capacity(PENDING_CAP),
        }
    }

    pub fn process(
        &mut self,
        rms: f32,
        rms_low: f32,
        rms_mid: f32,
        rms_high: f32,
        tau_smoothed: f32,
    ) {
        let values = [rms, rms_low, rms_mid, rms_high];
        let refractory: u64 = if tau_smoothed.is_nan() {
            self.refractory_fallback_frames as u64
        } else {
            tau_smoothed.round().max(1.0) as u64
        };

        for c in 0..4 {
            let v = values[c];
            let now_above = v >= EVENT_THRESHOLD;
            let refractory_ok = match self.last_event_frame[c] {
                None => true,
                Some(t) => self.frame.saturating_sub(t) > refractory,
            };
            if now_above && !self.above[c] && refractory_ok {
                if self.pending.len() < PENDING_CAP {
                    self.pending.push(Event {
                        channel: c as u8,
                        frame: self.frame,
                        value: v,
                    });
                }
                self.last_event_frame[c] = Some(self.frame);
            }
            self.above[c] = now_above;
        }
        self.frame += 1;
    }

    pub fn take_pending(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: drive only channel 0 (`CHAN_RMS`); other three quiet.
    fn step_ch0(d: &mut EventDetector, v: f32, tau: f32) {
        d.process(v, 0.0, 0.0, 0.0, tau);
    }

    #[test]
    fn rising_edge_fires_once() {
        let mut d = EventDetector::new(48000.0, 1024);
        step_ch0(&mut d, 0.0, f32::NAN);
        step_ch0(&mut d, 0.95, f32::NAN);
        let evs = d.take_pending();
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].channel, CHAN_RMS);
        assert_eq!(evs[0].frame, 1);
        assert!((evs[0].value - 0.95).abs() < 1e-6);
    }

    #[test]
    fn sustained_does_not_refire() {
        let mut d = EventDetector::new(48000.0, 1024);
        step_ch0(&mut d, 0.0, f32::NAN);
        for _ in 0..100 {
            step_ch0(&mut d, 0.95, f32::NAN);
        }
        assert_eq!(d.take_pending().len(), 1);
    }

    #[test]
    fn falling_then_rising_after_refractory_fires_again() {
        let mut d = EventDetector::new(48000.0, 1024);
        // Fallback at sr=48000, hop=1024 ≈ 5 frames.
        step_ch0(&mut d, 0.0, f32::NAN);
        step_ch0(&mut d, 0.95, f32::NAN); // event #1 at frame 1
        step_ch0(&mut d, 0.0, f32::NAN);
        for _ in 0..6 {
            step_ch0(&mut d, 0.0, f32::NAN);
        }
        step_ch0(&mut d, 0.95, f32::NAN); // event #2, well past refractory
        assert_eq!(d.take_pending().len(), 2);
    }

    #[test]
    fn refractory_blocks_refire_within_window() {
        let mut d = EventDetector::new(48000.0, 1024);
        step_ch0(&mut d, 0.0, f32::NAN);
        step_ch0(&mut d, 0.95, f32::NAN); // event #1
        step_ch0(&mut d, 0.0, f32::NAN); // dip
        step_ch0(&mut d, 0.95, f32::NAN); // attempt #2 — within fallback (~5 frames)
        assert_eq!(d.take_pending().len(), 1);
    }

    #[test]
    fn nan_tau_uses_fallback_frames() {
        // At sr=48000, hop=1024: fallback = round(100 * 48000 / 1000 / 1024) = 5.
        let d = EventDetector::new(48000.0, 1024);
        assert_eq!(d.refractory_fallback_frames, 5);
    }

    #[test]
    fn channels_are_independent() {
        let mut d = EventDetector::new(48000.0, 1024);
        // Frame 0: all below.
        d.process(0.0, 0.0, 0.0, 0.0, f32::NAN);
        // Frame 1: only mid crosses.
        d.process(0.0, 0.0, 0.95, 0.0, f32::NAN);
        let evs = d.take_pending();
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].channel, CHAN_RMS_MID);
        assert_eq!(evs[0].frame, 1);
    }

    #[test]
    fn take_pending_empties_buffer() {
        let mut d = EventDetector::new(48000.0, 1024);
        step_ch0(&mut d, 0.0, f32::NAN);
        step_ch0(&mut d, 0.95, f32::NAN);
        assert_eq!(d.take_pending().len(), 1);
        // No new processing — second call is empty.
        assert_eq!(d.take_pending().len(), 0);
    }

    #[test]
    fn tau_drives_refractory_when_finite() {
        // tau_smoothed = 20 frames → refractory = 20.
        let mut d = EventDetector::new(48000.0, 1024);
        d.process(0.0, 0.0, 0.0, 0.0, 20.0);
        d.process(0.95, 0.0, 0.0, 0.0, 20.0); // event #1 at frame 1
        d.process(0.0, 0.0, 0.0, 0.0, 20.0); // dip
        // 15 frames at 0.0 — still inside the 20-frame refractory.
        for _ in 0..15 {
            d.process(0.0, 0.0, 0.0, 0.0, 20.0);
        }
        d.process(0.95, 0.0, 0.0, 0.0, 20.0); // attempt — should be blocked
        assert_eq!(d.take_pending().len(), 1);
    }
}
```

- [ ] **Step 2: Add the module declaration to `lib.rs`**

In `crates/dsp/src/lib.rs`, just below the existing `mod buffers;` line in the module declarations block (currently lines 3–8), add:

```rust
mod events;
```

The block should now read:

```rust
mod acf;
mod autogain;
mod beat;
mod buffers;
mod events;
mod perf;
mod spectrum;
```

- [ ] **Step 3: Run tests — verify they pass**

Run: `cargo test -p dsp events::tests`
Expected: 8 tests pass.

If any fail, the module's logic is wrong — fix `events.rs`, not the tests. (The tests encode the spec's behavioral requirements verbatim.)

- [ ] **Step 4: Run the full DSP test suite — verify no regressions**

Run: `cargo test -p dsp`
Expected: all existing tests still pass plus the 8 new ones.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/events.rs crates/dsp/src/lib.rs
git commit -m "feat(dsp): add EventDetector with rising-edge + refractory state machines"
```

---

## Task 2: Wire `EventDetector` into `Dsp` + add `take_events` wasm export

The detector now exists and is unit-tested. Plug it into `Dsp::process` so it sees the latest RMS values and `tau_smoothed`, and expose a `take_events` accessor that the worklet will call. One Rust integration test pins the flat-Float32Array encoding.

**Files:**
- Modify: `crates/dsp/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add this test to the existing `mod tests` block at the bottom of `crates/dsp/src/lib.rs` (right before the closing `}` of `mod tests`):

```rust
    #[test]
    fn take_events_encodes_stride_4_with_u24_split_frame() {
        use crate::events::{Event, CHAN_RMS, CHAN_RMS_HIGH};
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        // Inject events directly into the detector — bypasses the state
        // machine so we test only the encoding.
        dsp.events.pending_for_test().push(Event {
            channel: CHAN_RMS,
            frame: 5,
            value: 0.95,
        });
        dsp.events.pending_for_test().push(Event {
            channel: CHAN_RMS_HIGH,
            // (1 << 24) + 7 — exercises both halves of the u24/u24 split.
            frame: (1u64 << 24) + 7,
            value: 0.91,
        });

        let flat = dsp.take_events();
        assert_eq!(flat.len(), 8); // 2 events × stride 4

        // Event 0
        assert_eq!(flat[0] as u8, CHAN_RMS);
        let lo0 = flat[1] as u64;
        let hi0 = flat[2] as u64;
        assert_eq!((hi0 << 24) | lo0, 5);
        assert!((flat[3] - 0.95).abs() < 1e-6);

        // Event 1
        assert_eq!(flat[4] as u8, CHAN_RMS_HIGH);
        let lo1 = flat[5] as u64;
        let hi1 = flat[6] as u64;
        assert_eq!((hi1 << 24) | lo1, (1u64 << 24) + 7);
        assert!((flat[7] - 0.91).abs() < 1e-6);

        // Drain semantics: second call empty.
        assert_eq!(dsp.take_events().len(), 0);
    }
```

Also add a small test-only accessor on `EventDetector`. Append this to `crates/dsp/src/events.rs` after the closing `}` of `impl EventDetector`:

```rust
#[cfg(test)]
impl EventDetector {
    /// Test-only direct access to the pending buffer for round-trip tests
    /// that bypass the state machine.
    pub fn pending_for_test(&mut self) -> &mut Vec<Event> {
        &mut self.pending
    }
}
```

- [ ] **Step 2: Run the test — verify compile failure**

Run: `cargo test -p dsp take_events_encodes_stride_4`
Expected: compile error — `no method named 'take_events' found for struct 'Dsp'`, `no field 'events' on type 'Dsp'`.

- [ ] **Step 3: Add the `events: EventDetector` field to `Dsp`**

In `crates/dsp/src/lib.rs`, find the `Dsp` struct (currently around line 19) and add `events: EventDetector` to the field list. Also add the use import.

After the existing `use crate::spectrum::SpectrumState;` line, add:

```rust
use crate::events::EventDetector;
```

In the `Dsp` struct, append `events: EventDetector,` to the field list (after `auto_onset: AutoGain,`):

```rust
pub struct Dsp {
    buffers: Buffers,
    spectrum: SpectrumState,
    db_floor: f32,
    /// dt = hop_size / sample_rate, captured at construction. Used by
    /// `set_smoothing_tau` to recompute `smoothing_alpha`.
    dt: f32,
    acf: AcfState,
    beat: BeatState,
    perf: PerfState,
    auto_rms: AutoGain,
    auto_rms_low: AutoGain,
    auto_rms_mid: AutoGain,
    auto_rms_high: AutoGain,
    auto_onset: AutoGain,
    events: EventDetector,
}
```

- [ ] **Step 4: Construct the detector in `Dsp::new`**

In `Dsp::new` (currently around lines 40–61), append `events: EventDetector::new(sample_rate, hop_size),` to the struct literal — last field, after `auto_onset: AutoGain::new(...)`:

```rust
            auto_onset: AutoGain::new(dt, AUTOGAIN_DEFAULT_TAU_SECS),
            events: EventDetector::new(sample_rate, hop_size),
        }
```

- [ ] **Step 5: Add `take_events` to the wasm-bindgen impl block**

In the existing `#[wasm_bindgen] impl Dsp { ... }` block (around lines 37–145), add this method right after `set_param` (just before the closing `}` of that impl block):

```rust
    /// Drain the event detector. Returns a flat stride-4 Float32Array on the
    /// JS side: `[channel, frameLo, frameHi, value, ...]`. Frame is split
    /// u24/u24 so an f32 round-trip is exact past 16M frames (~5 days at
    /// hop=1024). Empty (length 0) when nothing fired this hop.
    pub fn take_events(&mut self) -> Vec<f32> {
        let events = self.events.take_pending();
        let mut out = Vec::with_capacity(events.len() * 4);
        for e in events {
            out.push(e.channel as f32);
            let lo = (e.frame & 0xFF_FFFF) as f32;
            let hi = ((e.frame >> 24) & 0xFF_FFFF) as f32;
            out.push(lo);
            out.push(hi);
            out.push(e.value);
        }
        out
    }
```

- [ ] **Step 6: Hook the detector into `Dsp::process`**

In `Dsp::process` (currently lines 63–100), insert the detector call after `self.beat.process(...)` (currently lines 86–95) and before `crate::acf::autocorrelate(...)`. The end of `process` should now read:

```rust
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

        let rms_now      = self.buffers.rms.last().copied().unwrap_or(0.0);
        let rms_low_now  = self.buffers.rmsLow.last().copied().unwrap_or(0.0);
        let rms_mid_now  = self.buffers.rmsMid.last().copied().unwrap_or(0.0);
        let rms_high_now = self.buffers.rmsHigh.last().copied().unwrap_or(0.0);
        let tau          = self.beat.tau_smoothed();
        self.events.process(rms_now, rms_low_now, rms_mid_now, rms_high_now, tau);

        crate::acf::autocorrelate(&self.buffers.waveform, &mut self.buffers.bufferAcf);

        self.perf.frame_end(t_start, &mut self.buffers.dspPerf);
    }
```

- [ ] **Step 7: Run the new test — verify it passes**

Run: `cargo test -p dsp take_events_encodes_stride_4`
Expected: 1 test passes.

- [ ] **Step 8: Run the full DSP test suite — verify no regressions**

Run: `cargo test -p dsp`
Expected: all tests pass.

- [ ] **Step 9: Build the wasm package — verify it compiles for the JS target**

Run: `npm run wasm`
Expected: `wasm-pack build` succeeds; `src/wasm-pkg/dsp.d.ts` now includes `take_events(): Float32Array`.

- [ ] **Step 10: Commit**

```bash
git add crates/dsp/src/lib.rs crates/dsp/src/events.rs
git commit -m "feat(dsp): wire EventDetector into Dsp::process + take_events export"
```

(`src/wasm-pkg/` is gitignored — the wasm rebuild output is regenerated by `npm run wasm` and not committed.)

---

## Task 3: Worklet — drain events into the `features` message

The worklet calls `dsp.take_events()` each hop and includes the result in the `features` postMessage. There's no Vitest harness for `AudioWorkletProcessor`, so verification is manual: build the project and check the message shape in the browser. The shape is small enough that a TS-side decoder test in Task 5 covers the protocol contract.

**Files:**
- Modify: `src/audio/dsp-worklet.ts`

- [ ] **Step 1: Add `take_events` call and include in postMessage**

In `src/audio/dsp-worklet.ts`, the existing `process` method (lines 124–155) builds a `features` message inside the `while` loop. Modify the loop body so it reads:

```ts
    this.hopCounter += len;
    while (this.hopCounter >= this.hopSize) {
      this.dsp.process(this.window);
      const buffers: Record<string, Float32Array> = {};
      const transferList: ArrayBuffer[] = [];
      for (const name of this.bufferNames) {
        const data = this.dsp.get_buffer(name);
        buffers[name] = data;
        transferList.push(data.buffer as ArrayBuffer);
      }
      const events = this.dsp.take_events();
      transferList.push(events.buffer as ArrayBuffer);
      this.port.postMessage({ type: "features", buffers, events }, transferList);
      this.hopCounter -= this.hopSize;
    }
```

The `events` Float32Array is length 0 most hops. Including it unconditionally is simpler than branching and saves one allocation per hop (no need to special-case zero-length).

- [ ] **Step 2: Build the project — verify it compiles**

Run: `npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add src/audio/dsp-worklet.ts
git commit -m "feat(audio): drain DSP events into features message"
```

---

## Task 4: `FeatureStore` — add event API + tests

Extend the store with `pushEvents`/`getEvents`/`clearEvents` and export the `AudioEvent` type plus `CHAN_*` constants. New unit-test file pins the contract.

**Files:**
- Modify: `src/store/FeatureStore.ts`
- Create: `tests/store/FeatureStore.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `tests/store/FeatureStore.test.ts` with this content:

```ts
import { describe, it, expect } from "vitest";
import {
  FeatureStore,
  CHAN_RMS,
  CHAN_RMS_LOW,
  CHAN_RMS_MID,
  CHAN_RMS_HIGH,
  type AudioEvent,
} from "../../src/store/FeatureStore";

describe("FeatureStore — buffer API (existing behavior)", () => {
  it("set then get returns the same buffer", () => {
    const s = new FeatureStore();
    const buf = new Float32Array([1, 2, 3]);
    s.set("k", buf);
    expect(s.get("k")).toBe(buf);
  });

  it("get of unknown key returns a shared empty Float32Array", () => {
    const s = new FeatureStore();
    expect(s.get("nope").length).toBe(0);
  });
});

describe("FeatureStore — event API", () => {
  it("pushEvents then getEvents returns appended events in order", () => {
    const s = new FeatureStore();
    const a: AudioEvent = { channel: CHAN_RMS, frame: 1, value: 0.95 };
    const b: AudioEvent = { channel: CHAN_RMS_HIGH, frame: 2, value: 0.91 };
    s.pushEvents([a, b]);
    const got = s.getEvents();
    expect(got.length).toBe(2);
    expect(got[0]).toEqual(a);
    expect(got[1]).toEqual(b);
  });

  it("multiple pushEvents calls accumulate", () => {
    const s = new FeatureStore();
    s.pushEvents([{ channel: CHAN_RMS_LOW, frame: 1, value: 0.95 }]);
    s.pushEvents([{ channel: CHAN_RMS_MID, frame: 2, value: 0.92 }]);
    expect(s.getEvents().length).toBe(2);
    expect(s.getEvents()[0].channel).toBe(CHAN_RMS_LOW);
    expect(s.getEvents()[1].channel).toBe(CHAN_RMS_MID);
  });

  it("clearEvents empties the queue", () => {
    const s = new FeatureStore();
    s.pushEvents([{ channel: CHAN_RMS, frame: 1, value: 0.95 }]);
    s.clearEvents();
    expect(s.getEvents().length).toBe(0);
  });

  it("clearEvents on an empty queue is a no-op", () => {
    const s = new FeatureStore();
    s.clearEvents();
    expect(s.getEvents().length).toBe(0);
  });

  it("getEvents returns the same content across reads when no clear", () => {
    const s = new FeatureStore();
    const ev: AudioEvent = { channel: CHAN_RMS, frame: 1, value: 0.95 };
    s.pushEvents([ev]);
    const first = s.getEvents();
    const second = s.getEvents();
    expect(first.length).toBe(1);
    expect(second.length).toBe(1);
    expect(first[0]).toEqual(second[0]);
  });
});
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `npx vitest run tests/store/FeatureStore.test.ts`
Expected: failures — `pushEvents`, `getEvents`, `clearEvents`, `CHAN_RMS`, `AudioEvent` are not exported. Module-resolution / type errors.

- [ ] **Step 3: Extend `FeatureStore` with the event API**

Replace the entire contents of `src/store/FeatureStore.ts` with:

```ts
const EMPTY = new Float32Array(0);

export const CHAN_RMS = 0 as const;
export const CHAN_RMS_LOW = 1 as const;
export const CHAN_RMS_MID = 2 as const;
export const CHAN_RMS_HIGH = 3 as const;

export type AudioEvent = {
  channel: 0 | 1 | 2 | 3;
  frame: number;
  value: number;
};

export class FeatureStore {
  private buffers = new Map<string, Float32Array>();
  private events: AudioEvent[] = [];

  set(key: string, buf: Float32Array): void {
    this.buffers.set(key, buf);
  }

  get(key: string): Float32Array {
    return this.buffers.get(key) ?? EMPTY;
  }

  pushEvents(events: readonly AudioEvent[]): void {
    for (const e of events) this.events.push(e);
  }

  getEvents(): readonly AudioEvent[] {
    return this.events;
  }

  clearEvents(): void {
    this.events.length = 0;
  }
}
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `npx vitest run tests/store/FeatureStore.test.ts`
Expected: 7 tests pass.

- [ ] **Step 5: Run the full TS test suite — verify no regressions**

Run: `npm test`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/store/FeatureStore.ts tests/store/FeatureStore.test.ts
git commit -m "feat(store): FeatureStore audio event queue + AudioEvent type"
```

---

## Task 5: App — decode events and clear queue per render frame

Wire the worklet message into the store: decode the flat `events` Float32Array into `AudioEvent[]`, push to store; in the render loop, call `clearEvents()` after `view.update()`. A small unit test for the decoder logic — extracted into a pure helper — pins the wire-format contract.

**Files:**
- Modify: `src/App.ts`
- Create: `tests/app/decodeEvents.test.ts`

- [ ] **Step 1: Write the failing decoder test**

Create `tests/app/decodeEvents.test.ts` with:

```ts
import { describe, it, expect } from "vitest";
import { decodeEvents } from "../../src/App";

describe("decodeEvents", () => {
  it("returns [] for empty input", () => {
    expect(decodeEvents(new Float32Array(0))).toEqual([]);
  });

  it("decodes a stride-4 Float32Array into AudioEvents", () => {
    // Two events: (channel=0, frame=5, value=0.95) and
    // (channel=3, frame=(1<<24)+7, value=0.91).
    const flat = new Float32Array([
      0, 5, 0, 0.95,
      3, 7, 1, 0.9100000262260437, // value chosen f32-clean enough
    ]);
    const events = decodeEvents(flat);
    expect(events.length).toBe(2);

    expect(events[0].channel).toBe(0);
    expect(events[0].frame).toBe(5);
    expect(events[0].value).toBeCloseTo(0.95, 5);

    expect(events[1].channel).toBe(3);
    expect(events[1].frame).toBe((1 << 24) + 7);
    expect(events[1].value).toBeCloseTo(0.91, 5);
  });

  it("handles large frame values past 24-bit boundary", () => {
    // frame = (5 << 24) + 12345
    const flat = new Float32Array([2, 12345, 5, 0.92]);
    const events = decodeEvents(flat);
    expect(events.length).toBe(1);
    expect(events[0].frame).toBe(5 * 0x1000000 + 12345);
  });
});
```

- [ ] **Step 2: Run the test — verify it fails**

Run: `npx vitest run tests/app/decodeEvents.test.ts`
Expected: failure — `decodeEvents` is not exported from `src/App`.

- [ ] **Step 3: Add the `decodeEvents` helper and wire events into App**

In `src/App.ts`:

a. Add an import for `AudioEvent`. The existing imports include the `FeatureStore`. Add `AudioEvent` to its named imports. Find the existing line:

```ts
import { FeatureStore } from "./store/FeatureStore";
```

Change it to:

```ts
import { FeatureStore, type AudioEvent } from "./store/FeatureStore";
```

b. Update the `WorkletMsg` type (currently lines 18–21) to include `events`:

```ts
type WorkletMsg = {
  type: "features";
  buffers: Record<string, Float32Array>;
  events: Float32Array;
};
```

c. Export the `decodeEvents` helper. Add this function at the **module top level**, after the imports and before `interface AppDeps` / `type WorkletMsg`:

```ts
/// Decode the worklet's flat stride-4 events array into typed AudioEvents.
/// Wire format: [channel, frameLo, frameHi, value, ...] where frame is split
/// u24/u24 (lo + hi*2^24) so the round-trip through f32 is exact past 16M.
export function decodeEvents(flat: Float32Array): AudioEvent[] {
  const out: AudioEvent[] = [];
  for (let i = 0; i + 3 < flat.length; i += 4) {
    const channel = (flat[i] | 0) as 0 | 1 | 2 | 3;
    const lo = flat[i + 1];
    const hi = flat[i + 2];
    const frame = hi * 0x1000000 + lo;
    const value = flat[i + 3];
    out.push({ channel, frame, value });
  }
  return out;
}
```

d. Update the worklet message handler in `App.start` (currently lines 85–91) to push decoded events into the store:

```ts
    workletNode.port.onmessage = (e) => {
      const msg = e.data as WorkletMsg;
      if (msg.type !== "features") return;
      for (const [name, buf] of Object.entries(msg.buffers)) {
        this.store.set(name, buf);
      }
      if (msg.events.length > 0) {
        this.store.pushEvents(decodeEvents(msg.events));
      }
    };
```

e. Update the render loop (currently lines 93–103) to clear the queue after `view.update()`:

```ts
    const loop = (now: number) => {
      this.fps.begin();
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;
      this.rig.update(dt);
      this.view.update();
      this.store.clearEvents();
      renderer.render(scene, camera);
      this.fps.end();
      this.rafHandle = requestAnimationFrame(loop);
    };
```

`clearEvents` runs after `view.update()` — visualizers that read `store.getEvents()` during their update see this frame's events; the next frame's `view.update()` only sees events that arrived after this clear. `renderer.render` doesn't read events, so clearing before vs after it is moot — clearing right after `view.update()` keeps the lifetime tight.

- [ ] **Step 4: Run the decoder test — verify it passes**

Run: `npx vitest run tests/app/decodeEvents.test.ts`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full TS test suite — verify no regressions**

Run: `npm test`
Expected: all tests pass (existing + 7 new FeatureStore + 3 new decodeEvents).

- [ ] **Step 6: Build the project — verify TS compiles**

Run: `npm run build`
Expected: build succeeds.

- [ ] **Step 7: Commit**

```bash
git add src/App.ts tests/app/decodeEvents.test.ts
git commit -m "feat(app): decode worklet events and drain queue per render frame"
```

---

## Task 6: Update CLAUDE.md

Three small edits to keep the architecture doc in sync with the new event flow.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the FeatureStore description**

In `CLAUDE.md`, find the existing line in the Rendering path section:

```
- `FeatureStore` is intentionally a thin `Map<string, Float32Array>` — buffers in, buffers out; no events. Missing keys return a shared empty `Float32Array` so renderers can no-op safely before the first features message arrives.
```

Replace it with:

```
- `FeatureStore` is a thin `Map<string, Float32Array>` plus a per-render-frame audio event queue. Buffers in, buffers out; events appended via `pushEvents`, read via `getEvents`, cleared by App after `view.update()` each render frame. Missing buffer keys return a shared empty `Float32Array` so renderers can no-op safely before the first features message arrives.
```

- [ ] **Step 2: Update the worklet → main message protocol**

Find the `**features**` bullet in the "Worklet ↔ main message protocol" section:

```
- **`features`** (worklet → main, ~47 Hz): `{ type, buffers: { [name]: Float32Array } }`. The buffer name set comes from `dsp.buffer_names()`, cached at boot/configure. Each frame the worklet builds the dict by calling `dsp.get_buffer(name)` for every cached name, posting them all in one transfer-list batch.
```

Replace it with:

```
- **`features`** (worklet → main, ~47 Hz): `{ type, buffers: { [name]: Float32Array }, events: Float32Array }`. The buffer name set comes from `dsp.buffer_names()`, cached at boot/configure. Each frame the worklet builds the dict by calling `dsp.get_buffer(name)` for every cached name, drains events via `dsp.take_events()`, and posts them in one transfer-list batch. `events` is a flat stride-4 array `[channel, frameLo, frameHi, value, ...]` — usually length 0; non-zero when an RMS channel crossed `EVENT_THRESHOLD = 0.9` on a rising edge during the hop. Frame is split u24/u24 to round-trip through f32 exactly past 16M frames. App decodes into `AudioEvent[]` and pushes into `FeatureStore`.
```

- [ ] **Step 3: Update the wasm-bindgen surface**

Find the `### Wasm-bindgen surface (`Dsp`)` section. The current text says:

```
Five methods total:
- `new(window_size, sample_rate, hop_size, rms_history_len) -> Dsp`
- `process(input: &[f32])`
- `get_buffer(name: &str) -> Vec<f32>` — string-keyed buffer accessor (Float32Array on the JS side).
- `buffer_names() -> Vec<String>` — list all 16 buffer keys; called once per configure to populate the worklet's name cache.
- `set_param(key: &str, value: f32)` — recognized keys: ...
```

Change "Five methods total:" to "Six methods total:" and add this bullet after `set_param`:

```
- `take_events() -> Vec<f32>` — drain the event detector's pending list. Returns a flat stride-4 Float32Array on the JS side: `[channel, frameLo, frameHi, value, ...]`. Empty when nothing fired this hop. Worklet calls every hop; main thread decodes via `decodeEvents` in `App.ts`.
```

- [ ] **Step 4: Add a Pitfalls entry**

In the `## Pitfalls / non-obvious invariants` section, append a new bullet:

```
- **Events are per-render-frame, not per-DSP-frame.** Visualizers read `store.getEvents()` during their `update()`; App clears the queue after `view.update()` every RAF tick. Anything that misses a render frame loses those events permanently — the queue is not retained. If a visualizer needs decay/fade behavior, it tracks that state itself; don't try to keep events alive in the queue. Rising-edge + refractory means the queue is only ever non-empty when an RMS channel actually crossed `EVENT_THRESHOLD = 0.9` from below, and refractory blocks re-fires within one `tau_smoothed`-worth of frames (with ~100 ms fallback when tau is NaN).
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): document FeatureStore event queue + features.events field"
```

---

## Task 7: End-to-end verification

Final sanity check: full Rust suite, full TS suite, full build, and a quick browser smoke test to confirm an actual rising edge produces an event.

**Files:** none modified.

- [ ] **Step 1: Run the full Rust test suite**

Run: `cargo test -p dsp`
Expected: all tests pass.

- [ ] **Step 2: Rebuild wasm**

Run: `npm run wasm`
Expected: success.

- [ ] **Step 3: Run the full TS test suite**

Run: `npm test`
Expected: all tests pass.

- [ ] **Step 4: Run the production build**

Run: `npm run build`
Expected: build succeeds with no errors or warnings new to this PR.

- [ ] **Step 5: Browser smoke test (manual)**

Run: `npm run dev`

In the browser at the dev URL:

1. Open DevTools → Console.
2. Add a temporary one-liner log in `src/App.ts`'s features handler, immediately after `if (msg.events.length > 0)`:
   ```ts
   console.log("[events]", decodeEvents(msg.events));
   ```
   (Don't commit this.)
3. Press `T` to start the test source (in-process oscillator, no permissions needed).
4. The oscillator's RMS will be steady — should see ONE event log per channel as the autogain settles, then silence (refractory + sustained-above-threshold blocks re-fires).
5. Reload, switch to mic, clap loudly. Expect a burst of 1–4 events per clap depending on how the bands respond, then silence until the next clap.
6. Remove the temporary `console.log` line. Confirm `git diff src/App.ts` shows the line gone.

Expected: events fire on rising edges, do not refire while sustained, and respect refractory between bursts.

- [ ] **Step 6: Final commit (if any cleanup needed)**

If the smoke test left any temporary changes, revert them. Otherwise, this task has no commit. Run `git status` to confirm a clean tree.

---

## File Structure Summary

**Created:**
- `crates/dsp/src/events.rs` — detector module + tests (Task 1; test-only accessor added in Task 2)
- `tests/store/FeatureStore.test.ts` — TS unit tests for the event API (Task 4)
- `tests/app/decodeEvents.test.ts` — wire-format decoder tests (Task 5)

**Modified:**
- `crates/dsp/src/lib.rs` — `mod events;` declaration (Task 1); `events: EventDetector` field, construction, `Dsp::process` hook, `take_events` wasm export, integration test (Task 2)
- `src/audio/dsp-worklet.ts` — `dsp.take_events()` + `events` in `features` postMessage (Task 3)
- `src/store/FeatureStore.ts` — event API + `AudioEvent` type + `CHAN_*` constants (Task 4)
- `src/App.ts` — `decodeEvents` helper, message handler updates, `clearEvents` in render loop (Task 5)
- `CLAUDE.md` — three section updates + new pitfalls entry (Task 6)
