// Reusable HUD sparkline: one small 2D canvas, N series over fixed-length ring
// buffers, autoscaled to a 1-2-5 "nice" max so the scale doesn't jitter every
// frame. The canvas redraws on every push (it's tiny); the DOM header (legend +
// EMA readout) refreshes at ~10 Hz so text changes don't thrash layout.
// NaN is the "no value" sentinel throughout: it gaps the line, is skipped by
// autoscale, and reads as "—" in the header.

const HEADER_INTERVAL_MS = 100;
const EMA_ALPHA = 0.1;
const FONT = "ui-monospace,Menlo,Consolas,monospace";

export interface LinePlotOpts {
  title: string;
  series: { label: string; color: string }[];
  width?: number; // css px
  height?: number; // css px
  history?: number; // samples (~4 s at 60 fps by default)
  autoFloor?: number; // lower bound on the autoscaled max
  format?: (v: number) => string;
}

function niceCeil(v: number): number {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

export class LinePlot {
  readonly el: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  // Null in DOM environments without 2D canvas (tests) — push still keeps
  // buffers and header state, draw just no-ops.
  private ctx: CanvasRenderingContext2D | null;
  private valueEls: HTMLSpanElement[] = [];
  private data: Float32Array[];
  private head = 0;
  private filled = false;
  private ema: number[];
  private lastHeader = 0;
  private w: number;
  private h: number;
  private series: { label: string; color: string }[];
  private autoFloor: number;
  private format: (v: number) => string;

  constructor(opts: LinePlotOpts) {
    this.series = opts.series;
    this.w = opts.width ?? 240;
    this.h = opts.height ?? 36;
    this.autoFloor = opts.autoFloor ?? 1;
    this.format = opts.format ?? ((v) => v.toFixed(2));
    const history = opts.history ?? 240;
    this.data = opts.series.map(() => new Float32Array(history).fill(NaN));
    this.ema = opts.series.map(() => NaN);

    this.el = document.createElement("div");
    const header = document.createElement("div");
    header.style.cssText = `display:flex;flex-wrap:wrap;gap:0 8px;font:10px/1.4 ${FONT};color:#cfe;text-shadow:0 0 3px #000,0 0 3px #000`;
    const title = document.createElement("span");
    title.textContent = opts.title;
    title.style.color = "#8aa";
    header.appendChild(title);
    for (const s of opts.series) {
      const item = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.style.cssText = `display:inline-block;width:6px;height:6px;border-radius:2px;margin-right:3px;background:${s.color}`;
      const value = document.createElement("span");
      value.textContent = `${s.label} —`;
      item.append(swatch, value);
      header.appendChild(item);
      this.valueEls.push(value);
    }

    this.canvas = document.createElement("canvas");
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.w * dpr;
    this.canvas.height = this.h * dpr;
    this.canvas.style.cssText = `width:${this.w}px;height:${this.h}px;display:block`;
    this.ctx = this.canvas.getContext("2d");
    this.ctx?.scale(dpr, dpr);

    this.el.append(header, this.canvas);
  }

  // One value per series, in constructor order.
  push(...values: number[]): void {
    for (let s = 0; s < this.data.length; s++) {
      const v = values[s] ?? NaN;
      this.data[s][this.head] = v;
      if (Number.isFinite(v)) {
        this.ema[s] = Number.isNaN(this.ema[s]) ? v : this.ema[s] + EMA_ALPHA * (v - this.ema[s]);
      }
    }
    this.head = (this.head + 1) % this.data[0].length;
    if (this.head === 0) this.filled = true;

    const now = performance.now();
    if (now - this.lastHeader >= HEADER_INTERVAL_MS) {
      this.lastHeader = now;
      for (let s = 0; s < this.valueEls.length; s++) {
        const v = this.ema[s];
        this.valueEls[s].textContent =
          `${this.series[s].label} ${Number.isNaN(v) ? "—" : this.format(v)}`;
      }
    }
    this.draw();
  }

  private draw(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const { w, h } = this;
    const len = this.data[0].length;
    const n = this.filled ? len : this.head;

    let dataMax = 0;
    for (const buf of this.data) {
      for (let i = 0; i < n; i++) {
        const v = buf[this.filled ? (this.head + i) % len : i];
        if (v > dataMax) dataMax = v;
      }
    }
    const max = niceCeil(Math.max(this.autoFloor, dataMax));

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    for (let s = 0; s < this.data.length; s++) {
      const buf = this.data[s];
      ctx.strokeStyle = this.series[s].color;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < n; i++) {
        const v = buf[this.filled ? (this.head + i) % len : i];
        if (!Number.isFinite(v)) {
          pen = false;
          continue;
        }
        const x = (i / (len - 1)) * w;
        // 1px inset so a pegged line isn't clipped by the canvas edge.
        const y = h - 1 - (Math.min(v, max) / max) * (h - 2);
        if (pen) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        pen = true;
      }
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.font = `9px ${FONT}`;
    ctx.fillText(this.format(max), 3, 9);
  }
}
