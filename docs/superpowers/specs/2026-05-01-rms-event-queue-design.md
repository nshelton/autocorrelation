# RMS Event Queue Design

**Date:** 2026-05-01
**Status:** Design approved, awaiting plan

## Goal

Add a per-frame audio event queue that fires when any of the four RMS channels (`rms`, `rmsLow`, `rmsMid`, `rmsHigh`) crosses a fixed threshold on a rising edge. Events flow worklet → main thread without loss across the worklet/render rate mismatch (worklet ~47 Hz at hop=1024, can go higher; render fixed at 60 Hz). Visualizers consume events per render frame; event payloads are self-contained and discardable after one frame.

Events are conceptually distinct from the existing buffer signals: a buffer is a continuously-overwritten time series, an event is an instantaneous, lossless notification. They need a different transport, but they share the same `features` message and the same `FeatureStore` to keep the architecture flat.

## Architecture

The detector lives next to the existing pipeline stages in the Rust DSP crate. After `Dsp::process` writes RMS and beat outputs, a new `EventDetector` reads the latest sample of each of the four RMS channels plus `tau_smoothed`, runs four independent rising-edge state machines with refractory, and pushes any new events into a pending list owned by the detector. The worklet drains the list once per hop via a new `dsp.take_events()` accessor, transfers the result on the existing `features` message, and the main thread's `FeatureStore` accumulates events into a per-render-frame queue that the render loop clears after `view.update()`.

No new message types, no new top-level main-thread state object — the event queue is folded into `FeatureStore` and the existing `features` message gains one field. CLAUDE.md is updated to reflect both changes.

The threshold (`0.9`) is a compile-time constant in the new module, not a runtime param. The refractory length is one `tau_smoothed`-worth of DSP frames; if `tau_smoothed` is NaN (silence / no rhythmic content / boot transient) a fixed fallback (~100 ms in DSP frames) is used.

## Tech Stack

- **Rust** — new module `crates/dsp/src/events.rs`. No new crate dependencies.
- **TypeScript AudioWorklet** — extend the existing `features` postMessage payload. No new APIs.
- **TypeScript main thread** — extend `FeatureStore` with three small methods. No new classes, no new files in `src/store/`.
- **wasm-bindgen** — one new exported method on `Dsp` (`take_events`). Returns `Vec<f32>` (Float32Array on the JS side), matching the existing `get_buffer` style.

## File Structure

**New files:**

- `crates/dsp/src/events.rs` — `EventDetector`, `Event`, `EVENT_THRESHOLD`, `REFRACTORY_FALLBACK_FRAMES`, plus `#[cfg(test)] mod tests`.
- `tests/store/FeatureStore.test.ts` — TS unit tests for the new event API on `FeatureStore`.

**Modified files:**

- `crates/dsp/src/lib.rs` — `mod events;`. `Dsp` gains an `EventDetector` field. `Dsp::process` calls `events.process(...)` after `beat.update_beat_pulses` (i.e. after `tau_smoothed` is current). `Dsp` exposes `take_events()` (wasm-bindgen) which delegates to the detector and returns a flat `Vec<f32>` stride-4 encoding.
- `src/audio/dsp-worklet.ts` — after `dsp.process(...)`, call `dsp.take_events()` and include it in the `features` postMessage as `events`. Add the buffer to the transfer list.
- `src/store/FeatureStore.ts` — add `pushEvents`, `getEvents`, `clearEvents` and an internal `AudioEvent[]` field. Export the `AudioEvent` type and the channel-ID constants.
- `src/App.ts` — message handler decodes `msg.events` (Float32Array, stride 4) into `AudioEvent[]` and calls `store.pushEvents(...)`. The render loop calls `store.clearEvents()` after `view.update()`.
- `CLAUDE.md` — three edits (FeatureStore section, worklet protocol section, pitfalls section). See the CLAUDE.md updates section below.

## Components

### `crates/dsp/src/events.rs`

**Constants:**

```rust
/// RMS value (post-autogain, normalized [0,1]) at and above which a
/// rising-edge event fires. Compile-time constant by design — events
/// are a coarse "loud peak" notification, not a tunable signal.
pub const EVENT_THRESHOLD: f32 = 0.9;

/// Fallback refractory in DSP frames when `tau_smoothed` is NaN
/// (silence / boot transient / no rhythmic content). ~100 ms at
/// default sr=48000, hop=1024 → 100 ms × 48000 / 1024 ≈ 5 frames.
/// Computed in `EventDetector::new` from the actual sr/hop.
const REFRACTORY_FALLBACK_MS: f32 = 100.0;
```

**Channel ID convention** (constants exported from the module):

```rust
pub const CHAN_RMS: u8 = 0;
pub const CHAN_RMS_LOW: u8 = 1;
pub const CHAN_RMS_MID: u8 = 2;
pub const CHAN_RMS_HIGH: u8 = 3;
```

**Types:**

```rust
#[derive(Clone, Copy)]
pub struct Event {
    pub channel: u8,
    pub frame: u64,   // DSP-frame index, monotonic since last reconfigure
    pub value: f32,   // RMS value at trigger time
}

pub struct EventDetector {
    above: [bool; 4],
    last_event_frame: [Option<u64>; 4],
    frame: u64,
    refractory_fallback_frames: usize,
    pending: Vec<Event>,
}
```

**Method surface:**

```rust
impl EventDetector {
    pub fn new(sample_rate: f32, hop_size: usize) -> Self;

    /// Called once per hop, after RMS and beat stages have updated
    /// their latest values. `tau_smoothed` may be NaN.
    pub fn process(
        &mut self,
        rms: f32,
        rms_low: f32,
        rms_mid: f32,
        rms_high: f32,
        tau_smoothed: f32,
    );

    /// Drain pending events. Returned vec is fresh; internal buffer
    /// is left empty for the next hop. Worklet calls this every hop.
    pub fn take_pending(&mut self) -> Vec<Event>;
}
```

(No `reset` method — `Dsp` reconfigure constructs a fresh `Dsp`, which constructs a fresh `EventDetector`. Tests do the same.)
```

**Per-channel state machine** (inside `process`, executed for each `c in 0..4`):

```text
let v = values[c];
let refractory = if tau_smoothed.is_nan() {
    self.refractory_fallback_frames
} else {
    tau_smoothed.round() as usize
};

let now_above = v >= EVENT_THRESHOLD;

let refractory_ok = match self.last_event_frame[c] {
    None => true,
    Some(t) => self.frame - t > refractory as u64,
};

if now_above && !self.above[c] && refractory_ok {
    self.pending.push(Event { channel: c as u8, frame: self.frame, value: v });
    self.last_event_frame[c] = Some(self.frame);
}

self.above[c] = now_above;
```

After the four-channel loop: `self.frame += 1`.

**Pending cap:** `self.pending` is bounded at 64. If the loop would push past 64, drop the new event silently. Realistic worst case is 4 events per hop (one per channel max, refractory enforced), so 64 is paranoid but cheap.

### `crates/dsp/src/lib.rs`

Two changes.

**`Dsp` struct:** new field `events: EventDetector`. Constructed in `Dsp::new` with the same `sample_rate` and `hop_size` already passed to `Dsp::new`. No reset path needed — reconfigure rebuilds the whole `Dsp` (worklet calls `dsp.free()` then `new Dsp(...)`), which constructs a fresh detector.

**`Dsp::process`:** after `self.beat.process(...)` (i.e. `tau_smoothed` is current) and before `self.perf.frame_end(...)`, add:

```rust
let rms_now      = self.buffers.rms.last().copied().unwrap_or(0.0);
let rms_low_now  = self.buffers.rmsLow.last().copied().unwrap_or(0.0);
let rms_mid_now  = self.buffers.rmsMid.last().copied().unwrap_or(0.0);
let rms_high_now = self.buffers.rmsHigh.last().copied().unwrap_or(0.0);
let tau          = self.beat.tau_smoothed();  // method on BeatState, may be NaN

self.events.process(rms_now, rms_low_now, rms_mid_now, rms_high_now, tau);
```

`BeatState::tau_smoothed` is private; the existing `pub fn tau_smoothed(&self) -> f32` accessor in `beat.rs` is the entry point.

**New wasm-bindgen export:**

```rust
#[wasm_bindgen]
impl Dsp {
    pub fn take_events(&mut self) -> Vec<f32> {
        let events = self.events.take_pending();
        let mut out = Vec::with_capacity(events.len() * 4);
        for e in events {
            out.push(e.channel as f32);
            // u64 frame split into two f32s preserves precision past 2^24.
            // Lo = low 24 bits as f32, hi = high bits as f32. JS recombines.
            let lo = (e.frame & 0xFFFFFF) as f32;
            let hi = ((e.frame >> 24) & 0xFFFFFF) as f32;
            out.push(lo);
            out.push(hi);
            out.push(e.value);
        }
        out
    }
}
```

Encoding rationale: matches the existing flat-Float32Array pattern (`candidates`, `beatGrid`, etc.). 24/24 split keeps `frame` exact past 16M frames (~5 days at hop=1024, ~1 day at hop=256) — well past any session length we care about. A single f32 frame would lose precision after ~6h at hop=1024 and is risky enough to avoid.

### `src/audio/dsp-worklet.ts`

After `this.dsp.process(this.window)` in the existing `process` loop, before the postMessage:

```ts
const events = this.dsp.take_events();  // Float32Array, stride 4
// ...build buffers as today...
this.port.postMessage(
  { type: "features", buffers, events },
  [...transferList, events.buffer as ArrayBuffer]
);
```

`events.length` is `4 * N`; `N === 0` is the common case.

### `src/store/FeatureStore.ts`

```ts
const EMPTY = new Float32Array(0);

export const CHAN_RMS      = 0 as const;
export const CHAN_RMS_LOW  = 1 as const;
export const CHAN_RMS_MID  = 2 as const;
export const CHAN_RMS_HIGH = 3 as const;

export type AudioEvent = {
  channel: 0 | 1 | 2 | 3;
  frame: number;   // u48 reassembled from the wire u24/u24 split
  value: number;
};

export class FeatureStore {
  private buffers = new Map<string, Float32Array>();
  private events: AudioEvent[] = [];

  set(key: string, buf: Float32Array): void { this.buffers.set(key, buf); }
  get(key: string): Float32Array { return this.buffers.get(key) ?? EMPTY; }

  pushEvents(events: readonly AudioEvent[]): void {
    for (const e of events) this.events.push(e);
  }
  getEvents(): readonly AudioEvent[] { return this.events; }
  clearEvents(): void { this.events.length = 0; }
}
```

### `src/App.ts`

The existing `features` handler:

```ts
workletNode.port.onmessage = (e) => {
  const msg = e.data as WorkletMsg;
  if (msg.type !== "features") return;
  for (const [name, buf] of Object.entries(msg.buffers)) {
    this.store.set(name, buf);
  }
  // NEW:
  if (msg.events && msg.events.length > 0) {
    const decoded: AudioEvent[] = [];
    for (let i = 0; i < msg.events.length; i += 4) {
      const channel = msg.events[i] | 0;
      const lo = msg.events[i + 1];
      const hi = msg.events[i + 2];
      const value = msg.events[i + 3];
      // u24/u24 reassembly: hi << 24 | lo. JS numbers are 53-bit safe.
      const frame = hi * 0x1000000 + lo;
      decoded.push({ channel: channel as 0|1|2|3, frame, value });
    }
    this.store.pushEvents(decoded);
  }
};
```

`WorkletMsg` type updates to include `events: Float32Array`.

The render loop:

```ts
const loop = (now: number) => {
  // ...existing...
  this.view.update();
  renderer.render(scene, camera);
  this.store.clearEvents();   // NEW: drain after consumers had their look
  this.fps.end();
  this.rafHandle = requestAnimationFrame(loop);
};
```

`clearEvents` is called *after* `view.update()` so any current or future visualizer gets one read per render frame.

## Data Flow

```
DSP frame N (worklet, ~47Hz at default):
  Dsp::process
    └─ FFT, RMS, beat, ...                       (existing)
    └─ EventDetector::process(rms, rmsL, rmsM, rmsH, tau)
         └─ for each channel: rising edge + refractory check
              └─ push Event { channel, frame=N, value }   if fires
  worklet
    └─ dsp.take_events() -> flat Float32Array [c, lo, hi, v, ...]
    └─ postMessage({ type: "features", buffers, events }, transferList)

Main thread:
  port.onmessage
    └─ buffers → FeatureStore.set(name, buf)             (existing)
    └─ events  → decode → FeatureStore.pushEvents(...)   (new)

RAF tick (60Hz):
  view.update()                                          (visualizers read getEvents())
  renderer.render(...)
  store.clearEvents()                                    (new)
```

Rate mismatch is handled by accumulation: events from any number of `features` messages between two RAF ticks all land in `FeatureStore.events` and are visible together to the next `view.update()`.

## Error Handling and Edge Cases

- **`tau_smoothed` NaN at boot or during silence.** Detector falls back to `refractory_fallback_frames` (~100 ms in frames). Events still fire; they're just spaced by a fixed minimum.
- **First frame after `Dsp::new` (or reconfigure).** `above = [false; 4]`, `last_event_frame = [None; 4]`, `frame = 0`. If a channel is already above threshold on frame 0, it fires — this is correct rising-edge-from-unknown-state behavior. `last_event_frame[c] = None` makes refractory always-pass on first event, also correct.
- **HMR teardown.** `App.dispose()` clears `port.onmessage` (existing). The new App constructs a new `FeatureStore`. Worklet keeps producing events; first features message after rewire seeds the new queue. No special handling, no events lost beyond what's normally lost during the dispose/rewire microsecond gap.
- **Pending overflow in Rust.** Cap at 64 events per hop; silently drop beyond. Logging is overkill — this is a "should never happen" path.
- **No visualizers consume events yet.** Events accumulate, get cleared each frame. Cost: one `Array` push per event per frame, one `length = 0` per frame. Negligible.

## Testing

### Rust (`crates/dsp/src/events.rs`, `#[cfg(test)] mod tests`)

1. **Rising edge fires once.** Feed value sequence `[0.0, 0.95, 0.95, 0.95]` into channel 0; expect exactly one event at frame 1.
2. **Sustained does not refire.** Same channel held at 0.95 for 100 frames after firing; expect 1 event total.
3. **Falling then rising fires after refractory.** Sequence `[0.95, 0.0, 0.95]` separated by enough frames; expect 2 events.
4. **Refractory blocks re-fire.** Sequence `[0.95, 0.0, 0.95]` within refractory window; expect 1 event.
5. **NaN tau uses fallback.** Pass `f32::NAN` for `tau_smoothed`; verify refractory equals `refractory_fallback_frames` by timing the gap between two events.
6. **Per-channel independence.** Channel 1 fires; channels 0/2/3 quiet on the same frame; verify only one event with `channel == 1`.
7. **`take_pending` empties the buffer.** Push 3 events, call `take_pending`, call again with no new processing → second call returns empty.
8. **`Dsp::take_events` encoding round-trip.** Push events with known `(channel, frame, value)` (e.g. `frame = (1 << 24) + 5` to exercise both halves), call `take_events`, decode the stride-4 Float32Array; verify exact equality. This pins the u24/u24 frame split.

### TypeScript (`tests/store/FeatureStore.test.ts`, new)

1. `pushEvents` then `getEvents` returns the same events in order.
2. `clearEvents` empties the queue; `getEvents` returns `[]`.
3. Multiple `pushEvents` calls accumulate.
4. `getEvents` returns a reference (or readonly view) — the existing visualizer pattern reads without copying. Verify the second read returns the same content if no clear happened.

### Existing test suites

- Run full Rust test suite (`cargo test -p dsp`) — verify no regressions.
- Run TS test suite (`npm test`) — verify FeatureStore changes don't break existing consumers (`set`/`get` should be byte-identical).

## CLAUDE.md updates

Three changes, each small.

**1. FeatureStore description** (currently in the "Rendering path" section):

> `FeatureStore` is intentionally a thin `Map<string, Float32Array>` plus a per-render-frame audio-event queue. Buffers in, buffers out; events appended via `pushEvents`, read via `getEvents`, cleared by App after `view.update()` each render frame. Missing buffer keys return a shared empty `Float32Array`.

**2. Worklet → main message protocol** (the `features` bullet):

> **`features`** (worklet → main, ~47 Hz): `{ type, buffers: { [name]: Float32Array }, events: Float32Array }`. Buffer set is from `dsp.buffer_names()`. `events` is a flat stride-4 array `[channel, frameLo, frameHi, value, ...]` — usually length 0; non-zero when an RMS channel crossed `EVENT_THRESHOLD` on a rising edge during the hop. App decodes into `AudioEvent[]` and pushes into `FeatureStore`.

**3. Pitfalls / non-obvious invariants** (new entry):

> **Events are per-render-frame, not per-DSP-frame.** Visualizers read `store.getEvents()` during their `update()`; App clears the queue after `view.update()` every RAF tick. Anything that misses a render frame loses those events permanently — the queue is not retained. If a visualizer needs decay/fade behavior, it tracks that state itself; don't try to keep events alive in the queue.

## Open Questions

None as of approval. Threshold-as-constant, refractory-tied-to-tau-with-fallback, payload shape, transport, and main-thread API are all locked.

## Out of Scope

- **Visualizers consuming events.** This spec sets up the pipeline; first consumer is a separate piece of work.
- **Per-channel thresholds.** Single shared `EVENT_THRESHOLD` constant. If per-channel becomes useful, it's a small follow-up — promote to `[f32; 4]` and either keep as constant or expose as param.
- **Event types other than RMS-crossing.** Beat events, onset events, etc. — same infrastructure could carry them, but they're not part of this spec.
- **Sub-frame timestamping.** Events are quantized to the DSP hop. If millisecond-accurate event timing matters later, that's a separate design.
- **Persisting events past one render frame.** Visualizer-side state, not queue concern.
