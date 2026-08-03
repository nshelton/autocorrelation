// Latency instrumentation for the audio → analysis → scene chain.
//
// Two measurements, deliberately separate because they live on different clocks:
//
//  1. Continuous, no setup. Every features message carries the worklet's
//     wall-clock at publish, so `deliverMs` (worklet post → main-thread
//     receipt) and `ageMs` (how stale the newest analysis is at the instant the
//     scene reads it) come for free, every frame.
//
//  2. One-shot impulse — `fire()`. Schedules a full-scale noise burst straight
//     into the worklet's input at an exact AudioContext time. The worklet node
//     has no outputs, so the burst is inaudible and never leaves the graph:
//     what's left is purely the software detection path (window fill +
//     spectral-flux rise + hop quantization), measured entirely in the sample
//     clock so there is no cross-clock mapping to get wrong.
//
// What NEITHER can see, and why the on-screen delay is always bigger than the
// number printed here:
//   - mic/OS capture latency, upstream of the worklet (5-40 ms typical)
//   - compositor + panel latency, downstream of the GPU submit (1-2 display
//     frames + the panel's own pipeline)
// Both need a camera pointed at the screen to measure. See docs/latency.md.

// Burst length. Long enough to put real energy in every FFT bin, short enough
// that its own duration doesn't smear the measurement.
const BURST_SECS = 0.005;
// Lead-in before the impulse: gives the probe several analysis frames to
// measure the onset floor it has to clear, and covers the audio thread's
// render-ahead so the scheduled time isn't already in the past.
const LEAD_SECS = 0.2;
// Give up after this many analysis frames past the impulse (~0.6 s at 47 Hz).
const MAX_TRACE = 30;
// Detection threshold over the pre-impulse onset floor. onset is autogained to
// ~[0,1], and the burst is the loudest thing in the window, so it lands at 1.0.
const DETECT_MARGIN = 0.25;
const DETECT_FLOOR = 0.35;

interface Pending {
  clickT: number;
  baseline: number;
  trace: { lagMs: number; onset: number }[];
  detectLagMs: number;
  detectRecvMs: number;
  detectHopMs: number;
  consumeMs: number;
}

export class LatencyProbe {
  // Worklet publish → main-thread receipt, ms. NaN until the first message.
  deliverMs = NaN;
  // Age of the newest analysis frame when the scene consumed it, ms. Bigger
  // than deliverMs by the RAF wait; frames that get no new audio data (the DSP
  // runs at ~47 Hz, RAF at 60) show a full hop more.
  ageMs = NaN;

  private hopMs = NaN;
  private lastAudioT = NaN;
  // Observed analysis period (hopSize / sampleRate), derived from the message
  // stream rather than passed in, so it tracks a live hopSize change.
  private hopSecs = NaN;
  private p: Pending | null = null;

  constructor(
    private ctx: AudioContext,
    private dest: AudioNode,
  ) {}

  fire(): void {
    if (this.p) return;
    const sr = this.ctx.sampleRate;
    const buf = this.ctx.createBuffer(1, Math.round(sr * BURST_SECS), sr);
    const d = buf.getChannelData(0);
    // Broadband and full scale: spectral flux sums per-bin rises, so lighting
    // every bin at once is the sharpest edge the detector can possibly see.
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.dest);
    src.onended = () => src.disconnect();
    const clickT = this.ctx.currentTime + LEAD_SECS;
    src.start(clickT);
    this.p = {
      clickT,
      baseline: 0,
      trace: [],
      detectLagMs: NaN,
      detectRecvMs: NaN,
      detectHopMs: NaN,
      consumeMs: NaN,
    };
    console.log(
      `[latency] armed — impulse at ctx ${clickT.toFixed(4)}s ` +
        `(in ${(LEAD_SECS * 1000).toFixed(0)} ms). Autogain will re-normalize over ~1 s after.`,
    );
  }

  // From the worklet message handler. `audioT` is the context time of the
  // newest sample in that frame's analysis window; `hopMs` the worklet's
  // performance clock at publish (same time origin as the document's).
  onFeatures(audioT: number, hopMs: number, onset: Float32Array): void {
    const recvMs = performance.now();
    this.deliverMs = recvMs - hopMs;
    this.hopMs = hopMs;
    if (Number.isFinite(this.lastAudioT)) this.hopSecs = audioT - this.lastAudioT;
    this.lastAudioT = audioT;

    const p = this.p;
    if (!p) return;
    const v = onset.length > 0 ? onset[onset.length - 1] : 0;
    if (audioT < p.clickT) {
      // Lead-in frames set the floor the detector has to clear.
      p.baseline = Math.max(p.baseline, v);
      return;
    }
    if (Number.isFinite(p.detectLagMs)) return;

    const lagMs = (audioT - p.clickT) * 1000;
    p.trace.push({ lagMs, onset: v });
    if (v >= Math.max(DETECT_FLOOR, p.baseline + DETECT_MARGIN)) {
      p.detectLagMs = lagMs;
      p.detectRecvMs = recvMs;
      p.detectHopMs = hopMs;
    } else if (p.trace.length >= MAX_TRACE) {
      console.warn(
        `[latency] no onset rise within ${lagMs.toFixed(0)} ms ` +
          `(floor ${p.baseline.toFixed(2)}) — aborted. Try again with a quieter input.`,
      );
      this.p = null;
    }
  }

  // From the RAF loop, right after the components have written their geometry.
  onConsume(nowMs: number): void {
    this.ageMs = nowMs - this.hopMs;
    const p = this.p;
    if (!p || !Number.isFinite(p.detectLagMs) || Number.isFinite(p.consumeMs)) return;
    p.consumeMs = nowMs;
  }

  // From the renderAsync continuation — the frame's GPU work is encoded and
  // submitted. ±1 frame: a stale promise from the previous frame can land here
  // first, which under-reports the submit leg by one frame's encode.
  onRendered(nowMs: number): void {
    const p = this.p;
    if (!p || !Number.isFinite(p.consumeMs)) return;
    this.p = null;
    this.report(p, nowMs);
  }

  private report(p: Pending, renderedMs: number): void {
    const deliver = p.detectRecvMs - p.detectHopMs;
    const wait = p.consumeMs - p.detectRecvMs;
    const submit = renderedMs - p.consumeMs;
    const total = p.detectLagMs + deliver + wait + submit;
    const hops = Number.isFinite(this.hopSecs)
      ? ` (${(p.detectLagMs / (this.hopSecs * 1000)).toFixed(2)} hops)`
      : "";
    const ms = (v: number) => `${v.toFixed(1).padStart(6)} ms`;

    const trace = p.trace
      .map((t) => `    +${t.lagMs.toFixed(1).padStart(6)} ms  onset ${t.onset.toFixed(3)}`)
      .join("\n");

    console.log(
      [
        `[latency] onset floor before impulse: ${p.baseline.toFixed(3)}`,
        `[latency] analysis frames after the impulse:`,
        trace + "   ← detected",
        `[latency] ─────────────────────────────────────────────`,
        `[latency]   impulse → onset rise      ${ms(p.detectLagMs)}${hops}`,
        `[latency]   worklet → main thread     ${ms(deliver)}`,
        `[latency]   receipt → scene consume   ${ms(wait)}`,
        `[latency]   consume → GPU submit      ${ms(submit)}`,
        `[latency] ─────────────────────────────────────────────`,
        `[latency]   software path total       ${ms(total)}`,
        `[latency] not included: mic/OS capture upstream (~5-40 ms),`,
        `[latency]               compositor + panel downstream (~1-2 frames + panel).`,
      ].join("\n"),
    );
  }
}
