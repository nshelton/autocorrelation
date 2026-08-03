# Latency: audio in → geometry on screen

Reference numbers are at the defaults: 48 kHz, `dsp.windowSize` 2048,
`dsp.hopSize` 1024 (→ analysis every 21.33 ms, 46.9 Hz), 60 Hz display.

## The chain

```
mic/tab ──► OS + Chrome capture ──► AudioWorkletNode.process()  [128-sample quanta]
                                          │  sliding window, right-aligned
                                          ▼  fires every hopSize samples
                                    Dsp::process()  ── spectral flux → onset
                                          │
                                          ▼  postMessage (transfer list)
                                    App.onmessage ──► FeatureStore.set()
                                          │
                                          ▼  next RAF frame
                                    Modulator.tick()  ── EMA, power curve, threshold
                                          │  trigger → spawnQueue.request()
                                          ▼
                                    Spawner.update()  ── spawn(), setMatrixAt()
                                          │
                                          ▼
                                    PostStack.renderAsync()  ── encode + submit
                                          │
                                          ▼
                                    compositor ──► panel ──► photons
```

## Budget

| # | Stage | Typical | Range | Visible to JS |
|---|---|---|---|---|
| 1 | Mic/OS capture → worklet input | ~15 ms | 5–40 | no |
| 2 | Window fill + spectral-flux rise | ~16 ms | 11–32 | **yes** (probe) |
| 3 | Onset release EMA (`onsetSmoothingTauSecs`) | 0 ms | 0 | — |
| 4 | Worklet → main thread `postMessage` | ~2 ms | 0–16 | **yes** (`LAT deliver`) |
| 5 | RAF wait (receipt → `modulator.tick`) | ~8 ms | 0–17 | **yes** (`LAT age`) |
| 6 | Modulator source EMA (`smoothing`) | 0 ms | 0–150+ | no (config-dependent) |
| 7 | tick → spawn → `setMatrixAt` | <1 ms | — | **yes** |
| 8 | Render encode + GPU submit | ~3 ms | 1–10 | **yes** (`CPU rnd`) |
| 9 | Compositor + panel | ~25 ms | 16–50 | no |
| | **mic → photons** | **~70 ms** | **35–150+** | |
| | **software path only (2+4+5+7+8)** | **~30 ms** | **12–65** | measurable |

### Stage 2 is structural, and it's the biggest one you control

The analysis window is **right-aligned**: `dsp-worklet.ts` keeps the newest
sample at `window[windowSize-1]`, where the Hann taper is ~0. A transient that
just arrived contributes nothing to this frame's FFT; it only reaches full
weight once it has slid to the window center, one hop later.

Working it through for a transient landing at sample offset `s` inside the hop
that is about to fire (hop = 1024, window = 2048):

| arrival within the hop | Hann weight at the firing frame | detect lag |
|---|---|---|
| start of the hop (`s=0`) | ~1.0 | 21.3 ms, this frame |
| middle (`s=512`) | ~0.5 | 10.7 ms, this frame (half amplitude) |
| end (`s=1023`) | ~0.0 | 21.3 ms, **next** frame |

So detection lands **0.5–1.0 hops** after the transient, and the *amplitude*
swings ~2× with alignment. That amplitude swing is what makes a fixed trigger
threshold sometimes miss the first frame and fire on the next — which is the
occasional extra ~21 ms you're seeing. It is not jitter in the transport; it is
the window taper interacting with the threshold.

Lever: **halve `dsp.hopSize` to 512.** Detect lag halves to 5–11 ms and the
analysis rate goes to 94 Hz, which also kills most of stage 5's beat against
the 60 Hz RAF. Cost is 2× the DSP call rate (check `DSP ms` in the HUD).
Shrinking `windowSize` to 1024 halves the taper penalty too, at the cost of
frequency resolution for the low bands and the beat tracker.

### Stage 5 has a beat you can see

The DSP publishes at 46.9 Hz; RAF consumes at 60 Hz. ~22 % of frames get **no
new analysis** and re-read the previous buffer. The scene's data age therefore
oscillates between ~2 ms and ~23 ms on a ~3.7 Hz envelope. That is a real,
visible pulsing of responsiveness even when nothing is wrong. `LAT age` shows
it directly.

### Stage 6 is the one that surprises people

`ModBinding.smoothing` is a per-RAF-frame EMA with `alpha = 1 - smoothing`. At
`smoothing = 0.8` that's `tau ≈ 5 frames ≈ 83 ms` to 63 % — on top of
everything else. `power > 1` compounds it: it crushes small values, so a rising
edge takes longer to cross a trigger threshold. **Check the smoothing and power
on the binding driving the spawner before blaming the pipeline.**

Autogain (`dsp.autoGain`, τ = 1 s) doesn't delay the signal — the gain divides
the current sample — but after a loud passage the peak is still high, so a
fixed threshold takes longer to be crossed. Second-order, but real.

### What does *not* cost anything

- `Modulator.tick()` runs at `t2`, `components.update()` at `t3–t4`, same RAF
  frame. A trigger fires and its geometry is written in the same frame — no
  extra frame of latency there.
- `stepPhysics()` runs *before* the spawn, so a new body isn't simulated until
  the next frame, but it is drawn at its spawn position immediately.
- Onset release smoothing has instant attack; only the falling edge is filtered.

## Measuring it

### 1. Continuous — the `LAT` plot in the perf HUD (`P` to toggle)

- `deliver` — worklet publish → main-thread receipt (stage 4).
- `age` — how old the newest analysis was when the scene consumed it
  (stages 4 + 5, plus a whole hop on frames that got no new data).

Free, always on, needs no setup. Every `features` message now carries the
worklet's `performance.now()` at publish; the AudioWorklet global shares the
document's time origin, so the main thread diffs it directly.

### 2. One-shot software-path probe — press `L`

Schedules a 5 ms full-scale noise burst straight into the worklet's input at an
exact `AudioContext` time. The worklet node has no outputs, so the burst is
**inaudible** and never leaves the graph — what's left is purely the software
detection path. Prints to the console:

```
[latency] onset floor before impulse: 0.031
[latency] analysis frames after the impulse:
    +   4.2 ms  onset 0.033
    +  25.5 ms  onset 0.981   ← detected
[latency] ─────────────────────────────────────────────
[latency]   impulse → onset rise        25.5 ms (1.20 hops)
[latency]   worklet → main thread        1.8 ms
[latency]   receipt → scene consume      9.4 ms
[latency]   consume → GPU submit         3.1 ms
[latency] ─────────────────────────────────────────────
[latency]   software path total         39.8 ms
```

Because the impulse time and the frame time are both `AudioContext` times, the
first leg is exact to the sample — no cross-clock mapping. Fire it ~10 times
and look at the spread: that spread *is* the hop-alignment variance from
stage 2, and it should be roughly one hop wide.

Caveats: run it on a quiet input (loud music raises the onset floor and the
probe aborts); autogain latches its peak to the burst and re-normalizes over
~1 s afterward; the `consume → GPU submit` leg is ±1 frame because a stale
`renderAsync` promise can land first.

### 3. Capture-side latency (stage 1) — acoustic loopback

Not measurable in-process. Play a click out the speakers while the mic is the
source, so the pipeline hears its own output:

```
detect_time − schedule_time = outputLatency + acoustic flight + INPUT latency + software path
```

`AudioContext.outputLatency` gives the first term, flight time is
`distance / 343 m·s⁻¹` (put the mic against the speaker → ~0), and the probe
above gives the software path. What's left is the input path.

### 4. Ground truth (stages 1 + 9 included) — phone slo-mo

The only way to get photons. 240 fps ≈ 4.2 ms per frame:

1. Bind a spawner trigger to `onset.flux`, set spawn rate high enough that one
   hit is unmistakable.
2. Frame the screen and your hands in one shot. Clap.
3. Count frames from the clap contact to the first frame showing new geometry.

Do it ~10 times and take the median. Compare against the software-path total
from the `L` probe — the difference is stages 1 + 9, which is where the
remaining ~40 ms lives and which no code change in this repo can touch (short
of a lower-latency capture path).

## If it needs to be faster

Stage 2 is really **two** independent knobs that happen to be the same number at
the current 50 % overlap, which is why they read as one 21 ms effect:

- **`windowSize / 2`** — samples for a transient to slide from the right edge
  (Hann ≈ 0) to the center (Hann = 1). The *taper ramp*. 21.3 ms at 2048.
- **`hopSize`** — how often you get to look at all. The *quantization*, and the
  earliest point you can catch a transient at partial weight. 21.3 ms at 1024.

In order of payoff per unit of pain:

1. **`dsp.windowFall` 2048 → 128.** Kills the taper ramp and, more importantly,
   the 9.3× flux swing that makes a fixed threshold fire a hop late at random.
   **~5 ms of mean, ~11 ms of worst case, and nearly all of the variance.**
   See below.
2. **Zero out `smoothing`, pull `power` toward 1** on the spawner's binding.
   **0–150 ms**, entirely config.
3. **`dsp.hopSize` 1024 → 512.** Halves the quantization and smooths stage 5.
   **~5–10 ms.** But it also halves the onset history's time span (512 frames ×
   10.67 ms = 5.5 s) while doubling `tau_max` to ~140 lags, so the ACF at slow
   tempos gets noticeably noisier — pair it with `rmsHistoryLen` 512 → 1024,
   which doubles ACF cost.
4. **A dedicated short-window onset detector** — the real fix. See below.
5. **`dsp.windowSize` 2048 → 1024 is a bad trade on its own.** It does halve the
   taper ramp, and it does *not* touch the beat tracker (that runs on the
   hop-rate onset history, so it only cares about `hopSize`). But: the low band
   drops from 6 bins under 150 Hz to 3, and `bufferAcf` is `windowSize / 2` lags
   so its lowest resolvable pitch goes 46.9 Hz → 93.8 Hz — above the fundamental
   of a bass guitar and most male voices. You'd be gutting the kick-band
   resolution that drives the spawner in order to speed up the spawner.

### The window shape — `dsp.windowFall`

Hann is symmetric because it's built for *spectral* analysis, where time
alignment doesn't matter. Onset detection has the opposite need: the newest
samples should count **most**, not least. `dsp.windowFall` sets the length of
the window's falling edge in samples; `windowSize/2` (the default, and the max
slider value at every `windowSize`) is the ordinary symmetric Hann, and shorter
values move the peak next to the newest sample.

Swapping Hann for Hamming / Blackman / Kaiser buys **nothing** here — they're
all symmetric, so the ramp is still `windowSize/2`. Blackman is actively worse,
being more concentrated. Only asymmetry helps.

Measured flux for a 256-sample broadband burst at n=2048, by how long ago it
arrived (`age`) and by `fall`:

| burst age | 1024 (sym) | 512 | 256 | 128 | 64 |
|---|---|---|---|---|---|
| 0.0 ms (just arrived) | **0.63** | 2.17 | 4.73 | 5.13 | 5.30 |
| 5.3 ms | 2.38 | 5.27 | 6.06 | 5.90 | 5.79 |
| 10.7 ms | 4.50 | 6.02 | 5.53 | 5.20 | 5.04 |
| 21.3 ms | **5.86** | 4.05 | 3.25 | 2.93 | 2.78 |

Read the first column top to bottom: with the symmetric window the *same burst*
produces flux from 0.63 to 5.86 depending purely on where it landed in the hop
— a **9.3× swing**. That is the whole problem. A threshold tuned for 5.86 needs
a full extra hop before a freshly-arrived burst reaches it; a threshold tuned
for 0.63 fires on noise.

At `fall = 128` the same burst reads 5.13 / 5.90 / 5.20 over its first 10.7 ms
— a **1.15× swing**, essentially flat. And the peak (5.90) is within 1 % of the
symmetric window's peak (5.86), so **existing thresholds carry over unchanged**;
only the timing moves. `fall = 128` (2.7 ms) is the recommended starting point:
64 is no flatter, and 256 still has a 1.3× swing.

Amplitude calibration is unaffected by construction — both half-Hanns have mean
0.5 and mean-square 3/8 whatever their length, so `sum(w)` and `sum(w²)` — and
therefore `mag_scale` and `parseval_band_scale` — are identical for every
`fall`. There's a test pinning that.

The cost is leakage. The window stays C¹ at both ends and at the peak, so the
asymptotic sidelobe rolloff is preserved, but near sidelobes rise as `fall`
shrinks. It shows up as a smearier displayed spectrum with a higher floor
between peaks, and slightly blurrier band edges. Band RMS also becomes
recency-weighted, which for a visualizer is arguably an improvement.
`bufferAcf` reads the raw `waveform`, not the windowed buffer, so it's
untouched. **Sweep the slider against the spectrum display and decide by eye.**

### If leakage is too high: stop making one window serve two masters

The 2048-point Hann FFT exists for **frequency resolution** — low-band
separation, the displayed spectrum, `bufferAcf`. The `onset` flux signal rides
along on it and pays that window's latency, but flux needs almost no frequency
resolution: 256 or 512 bins is plenty to see "energy rose broadband."

Split them. Keep the 2048/Hann FFT exactly as-is for `spectrum`, the three
bands, and `bufferAcf`. Add a small (256-point) FFT purely for flux. A 256-point
transform is ~1/16 the work of a 2048-point one, so even running it every
128-sample quantum costs about half of what the current FFT costs per hop. The
taper ramp collapses from 21.3 ms to 2.7 ms.

The change is contained: `spectrum.rs::process` already returns `flux` as one of
four outputs, so it becomes its own small state struct fed from the tail of the
same input slice. The `onset` buffer keeps its exact semantics and rate, so the
ACF, the beat tracker, and every `MOD_SOURCES` consumer are untouched.

**The catch, and why this is #4 and not #1:** the transport is hop-rate.
`postMessage` only fires inside the hop branch of `process()`, so a flux spike
detected 3 ms after the transient still waits until the next hop boundary to
leave the worklet. Detecting faster only pays off if you also publish faster —
either a smaller `hopSize`, or a separate lightweight scalar message (a few
floats, no transfer list) posted every 2–4 quanta while the full buffer batch
stays at hop rate. And past ~8 ms the 60 Hz RAF becomes the floor anyway, so
there's a hard limit to how much of this is worth chasing.
