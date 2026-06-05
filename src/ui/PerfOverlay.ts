// Lightweight perf HUD: GPU frame time (WebGPU timestamp queries), main-thread
// CPU section breakdown, and DSP analysis time. All values EMA-smoothed; the DOM
// text is rewritten at ~10 Hz so the panel doesn't thrash layout every frame.
//
// GPU time comes from renderer.info.render.timestamp, which accumulates every
// render pass in a frame (scene pass + each post pass) — so it's a single total,
// not a per-pass split. It also resolves a frame or two late (async GPU readback),
// hence the smoothing. Falls back to "—" when timestamp queries are unsupported
// (renderer keeps info.render.timestamp at 0).

const EMA_ALPHA = 0.1;
const DOM_INTERVAL_MS = 100;

function ema(prev: number, x: number): number {
  return Number.isNaN(prev) ? x : prev + EMA_ALPHA * (x - prev);
}

export interface FrameSample {
  cameraMs: number;
  componentsMs: number;
  submitMs: number;
  analysisMs: number; // NaN when no dspPerf yet
  analysisHz: number;
  now: number; // RAF timestamp, drives DOM throttle
}

export class PerfOverlay {
  private el: HTMLPreElement;
  private lastDom = 0;

  private cameraMs = NaN;
  private componentsMs = NaN;
  private submitMs = NaN;
  private analysisMs = NaN;
  private analysisHz = NaN;
  private gpuMs = NaN;
  private drawCalls = NaN;
  private triangles = NaN;

  constructor() {
    const el = document.createElement("pre");
    el.style.cssText = [
      "position:fixed",
      "top:4rem",
      "right:1rem",
      "margin:0",
      "padding:4px 8px",
      "font:11px/1.4 ui-monospace,Menlo,Consolas,monospace",
      "color:#cfe",
      // Transparent background; text-shadow keeps it legible over the canvas.
      "background:transparent",
      "text-shadow:0 0 3px #000,0 0 3px #000",
      "text-align:right",
      "z-index:10",
      "pointer-events:none",
      "white-space:pre",
    ].join(";");
    el.classList.add("gui-el");
    this.el = el;
  }

  mount(parent: HTMLElement = document.body): void {
    parent.appendChild(this.el);
  }

  // Per-frame, synchronous. Refreshes the DOM text on the throttle interval.
  sample(s: FrameSample): void {
    this.cameraMs = ema(this.cameraMs, s.cameraMs);
    this.componentsMs = ema(this.componentsMs, s.componentsMs);
    this.submitMs = ema(this.submitMs, s.submitMs);
    if (Number.isFinite(s.analysisMs)) this.analysisMs = ema(this.analysisMs, s.analysisMs);
    if (Number.isFinite(s.analysisHz)) this.analysisHz = ema(this.analysisHz, s.analysisHz);

    if (s.now - this.lastDom >= DOM_INTERVAL_MS) {
      this.lastDom = s.now;
      this.render();
    }
  }

  // Read once per frame from renderer.info.render — GPU timestamp lags a frame
  // or two (async readback) and stays 0 when timestamp queries are unsupported.
  sampleGpu(ms: number, drawCalls: number, triangles: number): void {
    if (ms > 0) this.gpuMs = ema(this.gpuMs, ms);
    this.drawCalls = ema(this.drawCalls, drawCalls);
    this.triangles = ema(this.triangles, triangles);
  }

  toggle(): void {
    this.el.style.display = this.el.style.display === "none" ? "" : "none";
  }

  unmount(): void {
    this.el.remove();
  }

  private render(): void {
    const ms = (v: number) => (Number.isNaN(v) ? "  — " : v.toFixed(2).padStart(5));
    const draws = Number.isNaN(this.drawCalls) ? "—" : Math.round(this.drawCalls).toString();
    const tris = Number.isNaN(this.triangles) ? "—" : `${(this.triangles / 1000).toFixed(1)}k`;
    const cpuTotal = this.cameraMs + this.componentsMs + this.submitMs;
    const hz = Number.isNaN(this.analysisHz) ? "—" : this.analysisHz.toFixed(1);

    this.el.textContent =
      `GPU ${ms(this.gpuMs)} ms   draws ${draws}  tris ${tris}\n` +
      `CPU ${ms(cpuTotal)} ms   cam ${ms(this.cameraMs)} cmp ${ms(this.componentsMs)} sub ${ms(this.submitMs)}\n` +
      `DSP ${ms(this.analysisMs)} ms @ ${hz} Hz`;
  }
}
