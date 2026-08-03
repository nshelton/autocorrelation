// Perf HUD: scrolling sparklines (LinePlot) for GPU frame time, main-thread CPU
// breakdown, physics body counts, and DSP analysis time, plus a one-line text
// footer for the non-timeseries counters (draws, tris, DSP rate).
//
// GPU time comes from renderer.info.render.timestamp, which accumulates every
// render pass in a frame (scene pass + each post pass) — so it's a single total,
// not a per-pass split. It also resolves a frame or two late (async GPU readback).
// Stays NaN (gap in the plot) when timestamp queries are unsupported (renderer
// keeps info.render.timestamp at 0).

import { LinePlot } from "./LinePlot";

const EMA_ALPHA = 0.1;
const FOOTER_INTERVAL_MS = 100;

function ema(prev: number, x: number): number {
  return Number.isNaN(prev) ? x : prev + EMA_ALPHA * (x - prev);
}

export interface FrameSample {
  fps: number; // NaN on the first frame (no dt yet)
  cameraMs: number;
  physicsMs: number;
  componentsMs: number;
  submitMs: number;
  analysisMs: number; // NaN when no dspPerf yet
  analysisHz: number;
  bodies: number;
  colliders: number;
  now: number; // RAF timestamp, drives the footer throttle
}

const COUNT = (v: number) => Math.round(v).toString();

export class PerfOverlay {
  private el: HTMLDivElement;
  private fps: LinePlot;
  private gpu: LinePlot;
  private cpu: LinePlot;
  private phys: LinePlot;
  private dsp: LinePlot;
  private footer: HTMLDivElement;
  private lastFooter = 0;

  private analysisHz = NaN;
  private drawCalls = NaN;
  private triangles = NaN;

  constructor() {
    const el = document.createElement("div");
    el.style.cssText = [
      "position:fixed",
      "top:1rem",
      "right:1rem",
      "display:flex",
      "flex-direction:column",
      "gap:6px",
      "z-index:10",
      "pointer-events:none",
    ].join(";");
    el.classList.add("gui-el");
    this.el = el;

    // Fixed categorical order (validated CVD-safe on a dark surface):
    // blue, orange, aqua, yellow.
    // autoFloor 60 keeps the fps scale pinned at 100 through hitches instead
    // of the axis collapsing to the dip.
    this.fps = new LinePlot({
      title: "FPS",
      series: [{ label: "fps", color: "#3987e5" }],
      autoFloor: 60,
      format: COUNT,
    });
    this.gpu = new LinePlot({
      title: "GPU",
      series: [{ label: "ms", color: "#3987e5" }],
    });
    this.cpu = new LinePlot({
      title: "CPU",
      series: [
        { label: "cam", color: "#3987e5" },
        { label: "phys", color: "#d95926" },
        { label: "cmp", color: "#199e70" },
        { label: "sub", color: "#c98500" },
      ],
    });
    this.phys = new LinePlot({
      title: "PHYS",
      series: [
        { label: "bodies", color: "#3987e5" },
        { label: "colliders", color: "#d95926" },
      ],
      autoFloor: 10,
      format: COUNT,
    });
    this.dsp = new LinePlot({
      title: "DSP",
      series: [{ label: "ms", color: "#3987e5" }],
      autoFloor: 0.1,
    });
    for (const p of [this.fps, this.gpu, this.cpu, this.phys, this.dsp]) el.appendChild(p.el);

    this.footer = document.createElement("div");
    this.footer.style.cssText =
      "font:10px/1.4 ui-monospace,Menlo,Consolas,monospace;color:#8aa;" +
      "text-align:right;text-shadow:0 0 3px #000,0 0 3px #000";
    el.appendChild(this.footer);
  }

  mount(parent: HTMLElement = document.body): void {
    parent.appendChild(this.el);
  }

  // Per-frame, synchronous. Plots get raw samples (jitter is the point of a
  // timeseries); the plot headers EMA their own readouts.
  sample(s: FrameSample): void {
    this.fps.push(s.fps);
    this.cpu.push(s.cameraMs, s.physicsMs, s.componentsMs, s.submitMs);
    this.phys.push(s.bodies, s.colliders);
    this.dsp.push(s.analysisMs);
    if (Number.isFinite(s.analysisHz)) this.analysisHz = ema(this.analysisHz, s.analysisHz);

    if (s.now - this.lastFooter >= FOOTER_INTERVAL_MS) {
      this.lastFooter = s.now;
      const draws = Number.isNaN(this.drawCalls) ? "—" : COUNT(this.drawCalls);
      const tris = Number.isNaN(this.triangles) ? "—" : `${(this.triangles / 1000).toFixed(1)}k`;
      const hz = Number.isNaN(this.analysisHz) ? "—" : this.analysisHz.toFixed(1);
      this.footer.textContent = `draws ${draws}  tris ${tris}  dsp ${hz} Hz`;
    }
  }

  // Read once per frame from renderer.info.render — GPU timestamp lags a frame
  // or two (async readback) and stays 0 when timestamp queries are unsupported.
  sampleGpu(ms: number, drawCalls: number, triangles: number): void {
    this.gpu.push(ms > 0 ? ms : NaN);
    this.drawCalls = ema(this.drawCalls, drawCalls);
    this.triangles = ema(this.triangles, triangles);
  }

  toggle(): void {
    this.el.style.display = this.el.style.display === "none" ? "" : "none";
  }

  unmount(): void {
    this.el.remove();
  }
}
