//! Output scope — a floating Canvas2D panel showing the synth's live **output**
//! waveform (time domain) and spectrum (FFT). It taps the post-mix signal with a
//! Web Audio `AnalyserNode` branched off the worklet node (a node can fan out to
//! several destinations without affecting what reaches `context.destination`), so
//! it reflects everything you hear regardless of which track is selected.
//!
//! Self-contained DOM/Canvas2D + rAF, consistent with the rest of
//! `src/sequencer/`. It's a passive HUD: `pointer-events:none` so it never
//! intercepts editing, and it docks just above the params pane via the
//! `--synth-panel-h` var that `SequencerLayout` publishes — no layout changes.

const ACCENT = "#7fd1ff"; // waveform (matches the app accent)
const SPECTRUM = "#ffcc55"; // spectrum fill (matches the gold loop accent)
const LABEL = "rgba(204, 204, 204, 0.45)";
const AXIS = "rgba(204, 204, 204, 0.12)";

export class AnalysisPanel {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private analyser: AnalyserNode;
  private timeBuf: Float32Array<ArrayBuffer>; // [-1, 1] samples
  private freqBuf: Float32Array<ArrayBuffer>; // dBFS per bin
  private raf = 0;
  private dpr = 1;
  private resizeObserver: ResizeObserver;

  /** `source` is the synth worklet node; we tap it (it keeps reaching the
   *  speakers unchanged). */
  constructor(context: AudioContext, source: AudioNode) {
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048; // 2048 time samples → 1024 freq bins
    analyser.smoothingTimeConstant = 0.7; // steady spectrum; waveform is instantaneous
    analyser.minDecibels = -90;
    analyser.maxDecibels = -20;
    source.connect(analyser); // tap only — not connected onward, so audio is untouched
    this.analyser = analyser;
    this.timeBuf = new Float32Array(analyser.fftSize);
    this.freqBuf = new Float32Array(analyser.frequencyBinCount);

    const canvas = document.createElement("canvas");
    canvas.className = "gui-el";
    Object.assign(canvas.style, {
      position: "fixed",
      right: "8px",
      // Sit just above the params pane (whose height SequencerLayout publishes).
      bottom: "calc(var(--synth-panel-h, 0px) + 8px)",
      width: "360px",
      height: "150px",
      zIndex: "6",
      borderRadius: "6px",
      border: "1px solid rgba(204, 204, 204, 0.15)",
      background: "rgba(12, 12, 14, 0.82)",
      pointerEvents: "none", // passive HUD — never intercept editing underneath
    } satisfies Partial<CSSStyleDeclaration>);
    document.body.appendChild(canvas);
    this.canvas = canvas;

    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("[AnalysisPanel] 2D context unavailable");
    this.ctx = ctx;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();

    const loop = () => {
      this.draw();
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  private resize(): void {
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * this.dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * this.dpr);
  }

  private draw(): void {
    const { ctx, dpr } = this;
    const W = this.canvas.width;
    const H = this.canvas.height;
    if (W <= 0 || H <= 0) return;
    ctx.clearRect(0, 0, W, H); // CSS background shows through

    const pad = 6 * dpr;
    const gap = 5 * dpr;
    const plotH = H - 2 * pad - gap;
    const waveH = plotH * 0.45;
    const specH = plotH - waveH;
    const x0 = pad;
    const plotW = W - 2 * pad;

    // --- Waveform (time domain) ----------------------------------------
    const waveMid = pad + waveH / 2;
    ctx.strokeStyle = AXIS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, waveMid);
    ctx.lineTo(x0 + plotW, waveMid);
    ctx.stroke();

    this.analyser.getFloatTimeDomainData(this.timeBuf);
    const n = this.timeBuf.length;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.25 * dpr;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = x0 + (i / (n - 1)) * plotW;
      const y = waveMid - this.timeBuf[i] * (waveH / 2) * 0.92;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // --- Spectrum (FFT magnitude, log-frequency x) ---------------------
    const specTop = pad + waveH + gap;
    const specBottom = specTop + specH;
    this.analyser.getFloatFrequencyData(this.freqBuf);
    const bins = this.freqBuf.length;
    const { minDecibels, maxDecibels } = this.analyser;
    const range = maxDecibels - minDecibels;
    const logDen = Math.log2(bins);
    // Filled area from the baseline up to the (normalized) magnitude curve.
    ctx.beginPath();
    ctx.moveTo(x0, specBottom);
    for (let i = 1; i < bins; i++) {
      const x = x0 + (Math.log2(i + 1) / logDen) * plotW;
      const db = this.freqBuf[i];
      const norm = Math.max(0, Math.min(1, (db - minDecibels) / range));
      ctx.lineTo(x, specBottom - norm * specH);
    }
    ctx.lineTo(x0 + plotW, specBottom);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, specTop, 0, specBottom);
    grad.addColorStop(0, "rgba(255, 204, 85, 0.55)");
    grad.addColorStop(1, "rgba(255, 204, 85, 0.06)");
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = SPECTRUM;
    ctx.lineWidth = 1 * dpr;
    ctx.stroke();

    // --- Labels --------------------------------------------------------
    ctx.fillStyle = LABEL;
    ctx.font = `${9 * dpr}px ui-sans-serif, system-ui`;
    ctx.textBaseline = "top";
    ctx.fillText("WAVE", x0 + 2 * dpr, pad + 1 * dpr);
    ctx.fillText("FFT", x0 + 2 * dpr, specTop + 1 * dpr);
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.resizeObserver.disconnect();
    try {
      this.analyser.disconnect();
    } catch {
      /* already torn down */
    }
    this.canvas.remove();
  }
}
