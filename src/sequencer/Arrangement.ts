//! Arrangement view — the top pane of the sequencer. A Canvas2D timeline (same
//! DPR + rAF + ResizeObserver pattern as `PianoRoll`) showing every track as a
//! horizontal lane and each clip as a block with a **mini thumbnail** of its
//! contents (note dots for synth tracks, step dots for drum tracks). Clicking a
//! clip (or a track's header / empty lane) selects it; the host then populates
//! the editor + params panes. Replaces the old `TrackBar`.
//!
//! Mounts into a container the layout provides (fills it 100%); it does not
//! position itself, so the resizable arrangement pane owns its geometry.

import { DRUM_LANES } from "./drumkit";
import type { Clip, Project, Track } from "./model";

export interface ArrangementCallbacks {
  /** A clip (or its track) was clicked — open it in the editor + params. */
  onSelectClip?: (track: number, clip: number) => void;
  /** Track volume fader dragged — `value` 0..1 (drives the instrument gain). */
  onVolume?: (track: number, value: number) => void;
}

const BG = "rgba(12, 12, 14, 0.0)"; // pane already paints the bg; keep canvas clear
const HEADER_W = 120; // left track-header column width (CSS px)
const RULER_H = 16; // top bar-number ruler (CSS px)
const LANE_PAD = 3; // vertical padding inside each lane (CSS px)
const GRID = "rgba(204, 204, 204, 0.10)";
const BAR_LINE = "rgba(204, 204, 204, 0.28)";
const LANE_ALT = "rgba(255, 255, 255, 0.02)";
const HEADER_BG = "rgba(204, 204, 204, 0.05)";
const HEADER_SEL = "rgba(127, 209, 255, 0.16)";
const PLAYHEAD = "#ff5566";
const OUTLINE = "rgba(255, 255, 255, 0.9)";
const TEXT = "rgba(204, 204, 204, 0.85)";

const BEATS_PER_BAR = 4;
const VOL_FILL = "#7fd1ff"; // selected-track fader fill (matches the accent)

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16) || 127,
    parseInt(h.slice(2, 4), 16) || 127,
    parseInt(h.slice(4, 6), 16) || 127,
  ];
}

export class Arrangement {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private project: Project;
  private callbacks: ArrangementCallbacks;
  private playhead = 0;
  private selTrack = 0;
  private selClip = 0;
  private raf = 0;
  private dpr = 1;
  private resizeObserver: ResizeObserver;

  // Active volume-fader drag (header column), else null.
  private drag: { track: number } | null = null;
  private onDown = (e: PointerEvent) => this.pointerDown(e);
  private onMove = (e: PointerEvent) => this.pointerMove(e);
  private onUp = (e: PointerEvent) => this.pointerUp(e);

  constructor(project: Project, mount: HTMLElement, callbacks: ArrangementCallbacks = {}) {
    this.project = project;
    this.callbacks = callbacks;

    const canvas = document.createElement("canvas");
    canvas.className = "gui-el";
    Object.assign(canvas.style, {
      position: "absolute",
      inset: "0",
      width: "100%",
      height: "100%",
      touchAction: "none",
      cursor: "pointer",
    } satisfies Partial<CSSStyleDeclaration>);
    mount.appendChild(canvas);
    this.canvas = canvas;

    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("[Arrangement] 2D context unavailable");
    this.ctx = ctx;

    canvas.addEventListener("pointerdown", this.onDown);
    canvas.addEventListener("pointermove", this.onMove);
    canvas.addEventListener("pointerup", this.onUp);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();

    const loop = () => {
      this.draw();
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  // --- Public API ------------------------------------------------------

  setProject(project: Project): void {
    this.project = project;
  }

  setSelected(track: number, clip: number): void {
    this.selTrack = track;
    this.selClip = clip;
  }

  setPlayhead(beat: number): void {
    this.playhead = beat;
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener("pointerdown", this.onDown);
    this.canvas.removeEventListener("pointermove", this.onMove);
    this.canvas.removeEventListener("pointerup", this.onUp);
    this.canvas.remove();
  }

  // --- Geometry --------------------------------------------------------

  /** End of the visible timeline (beats): the loop window, stretched to fit the
   *  furthest clip so off-window content is still reachable. */
  private viewEndBeat(): number {
    let end = Math.max(this.project.loopEnd, 4);
    for (const t of this.project.tracks) {
      for (const c of t.clips) end = Math.max(end, c.start + c.length);
    }
    return end;
  }

  private beatToX(beat: number, padLeft: number, plotW: number, viewEnd: number): number {
    return padLeft + (beat / viewEnd) * plotW;
  }

  private xToBeat(x: number, padLeft: number, plotW: number, viewEnd: number): number {
    return ((x - padLeft) / plotW) * viewEnd;
  }

  private resize(): void {
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * this.dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * this.dpr);
  }

  private eventXY(e: PointerEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    return { x: (e.clientX - rect.left) * this.dpr, y: (e.clientY - rect.top) * this.dpr };
  }

  // --- Interaction -----------------------------------------------------

  private pointerDown(e: PointerEvent): void {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const dpr = this.dpr;
    const padLeft = HEADER_W * dpr;
    const padTop = RULER_H * dpr;
    const plotW = W - padLeft;
    const plotH = H - padTop;
    const n = this.project.tracks.length;
    if (n === 0 || plotW <= 0 || plotH <= 0) return;

    const { x, y } = this.eventXY(e);
    const laneH = plotH / n;
    const ti = Math.floor((y - padTop) / laneH);
    if (ti < 0 || ti >= n) return;
    const track = this.project.tracks[ti];
    const top = padTop + ti * laneH;

    if (x < padLeft) {
      // Header column. Grabbing the volume fader band starts a drag; otherwise
      // a click selects the track (its first clip).
      const vr = this.volumeRect(top, laneH, padLeft);
      if (vr && y >= vr.y - 6 * dpr && y <= vr.y + vr.h + 6 * dpr) {
        this.drag = { track: ti };
        this.canvas.setPointerCapture(e.pointerId);
        this.applyVolume(ti, x, vr);
        return;
      }
      this.selTrack = ti;
      this.selClip = 0;
      this.callbacks.onSelectClip?.(ti, 0);
      return;
    }

    // Clip area: select the clip under the cursor, else the track's first clip.
    const beat = this.xToBeat(x, padLeft, plotW, this.viewEndBeat());
    const hit = track.clips.findIndex((c) => beat >= c.start && beat <= c.start + c.length);
    this.selTrack = ti;
    this.selClip = hit >= 0 ? hit : 0;
    this.callbacks.onSelectClip?.(ti, this.selClip);
  }

  private pointerMove(e: PointerEvent): void {
    if (!this.drag) return;
    const dpr = this.dpr;
    const padTop = RULER_H * dpr;
    const padLeft = HEADER_W * dpr;
    const plotH = this.canvas.height - padTop;
    const n = this.project.tracks.length;
    if (n === 0 || plotH <= 0) return;
    const laneH = plotH / n;
    const top = padTop + this.drag.track * laneH;
    const vr = this.volumeRect(top, laneH, padLeft);
    if (vr) this.applyVolume(this.drag.track, this.eventXY(e).x, vr);
  }

  private pointerUp(e: PointerEvent): void {
    if (!this.drag) return;
    this.drag = null;
    this.canvas.releasePointerCapture(e.pointerId);
  }

  /** The volume fader's visual track rect in a lane header, or null if the lane
   *  is too short to show it. */
  private volumeRect(
    top: number,
    laneH: number,
    padLeft: number,
  ): { x: number; y: number; w: number; h: number } | null {
    const dpr = this.dpr;
    if (laneH <= 30 * dpr) return null;
    const x = 12 * dpr;
    return { x, y: top + laneH * 0.7 - 2 * dpr, w: padLeft - 24 * dpr, h: 4 * dpr };
  }

  private applyVolume(track: number, x: number, vr: { x: number; w: number }): void {
    this.callbacks.onVolume?.(track, clamp01((x - vr.x) / vr.w));
  }

  // --- Rendering -------------------------------------------------------

  private draw(): void {
    const { ctx } = this;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const dpr = this.dpr;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    const padLeft = HEADER_W * dpr;
    const padTop = RULER_H * dpr;
    const plotW = W - padLeft;
    const plotH = H - padTop;
    const n = this.project.tracks.length;
    if (n === 0 || plotW <= 0 || plotH <= 0) return;

    const viewEnd = this.viewEndBeat();
    const laneH = plotH / n;

    // Bar gridlines + numbers in the ruler.
    ctx.lineWidth = 1;
    ctx.font = `${10 * dpr}px ui-sans-serif, system-ui`;
    ctx.textBaseline = "middle";
    for (let b = 0; b <= viewEnd; b += BEATS_PER_BAR) {
      const x = this.beatToX(b, padLeft, plotW, viewEnd);
      ctx.strokeStyle = BAR_LINE;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.fillStyle = TEXT;
      ctx.fillText(`${b / BEATS_PER_BAR + 1}`, x + 3 * dpr, padTop / 2);
    }
    // Faint in-bar beat lines.
    ctx.strokeStyle = GRID;
    for (let b = 0; b <= viewEnd; b++) {
      if (b % BEATS_PER_BAR === 0) continue;
      const x = this.beatToX(b, padLeft, plotW, viewEnd);
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, H);
      ctx.stroke();
    }

    // Lanes.
    this.project.tracks.forEach((track, ti) => {
      const top = padTop + ti * laneH;
      if (ti % 2 === 1) {
        ctx.fillStyle = LANE_ALT;
        ctx.fillRect(padLeft, top, plotW, laneH);
      }
      // Clips with thumbnails.
      for (let ci = 0; ci < track.clips.length; ci++) {
        this.drawClip(track, ti, track.clips[ci], ci, top, laneH, padLeft, plotW, viewEnd);
      }
      // Header cell (drawn last per lane so it sits above clip overflow).
      this.drawHeader(track, ti, top, laneH, padLeft);
    });

    // Lane separators.
    ctx.strokeStyle = GRID;
    for (let i = 0; i <= n; i++) {
      const y = padTop + i * laneH;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // Playhead.
    if (this.playhead >= 0 && this.playhead <= viewEnd) {
      const x = this.beatToX(this.playhead, padLeft, plotW, viewEnd);
      ctx.strokeStyle = PLAYHEAD;
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  private drawHeader(track: Track, ti: number, top: number, laneH: number, padLeft: number): void {
    const { ctx } = this;
    const dpr = this.dpr;
    const selected = ti === this.selTrack;
    ctx.fillStyle = selected ? HEADER_SEL : HEADER_BG;
    ctx.fillRect(0, top, padLeft, laneH);

    const vr = this.volumeRect(top, laneH, padLeft);
    // When the fader fits, the name sits in the upper half; otherwise it centers.
    const nameY = vr ? top + laneH * 0.34 : top + laneH / 2;

    // Color swatch.
    ctx.fillStyle = track.color;
    ctx.beginPath();
    ctx.arc(12 * dpr, nameY, 4 * dpr, 0, Math.PI * 2);
    ctx.fill();
    // Name.
    ctx.fillStyle = selected ? "#ffffff" : TEXT;
    ctx.font = `${11 * dpr}px ui-sans-serif, system-ui`;
    ctx.textBaseline = "middle";
    ctx.save();
    ctx.beginPath();
    ctx.rect(22 * dpr, top, padLeft - 24 * dpr, laneH);
    ctx.clip();
    ctx.fillText(track.name, 22 * dpr, nameY);
    ctx.restore();

    // Volume fader — reads/drives the track's instrument gain (0..1).
    if (vr) {
      const gain = clamp01(track.instrument.params.gain ?? 0.25);
      ctx.fillStyle = "rgba(204, 204, 204, 0.18)";
      ctx.fillRect(vr.x, vr.y, vr.w, vr.h);
      ctx.fillStyle = selected ? VOL_FILL : "rgba(204, 204, 204, 0.6)";
      ctx.fillRect(vr.x, vr.y, vr.w * gain, vr.h);
    }
  }

  private drawClip(
    track: Track,
    ti: number,
    clip: Clip,
    ci: number,
    laneTop: number,
    laneH: number,
    padLeft: number,
    plotW: number,
    viewEnd: number,
  ): void {
    const { ctx } = this;
    const dpr = this.dpr;
    const x0 = this.beatToX(clip.start, padLeft, plotW, viewEnd);
    const x1 = this.beatToX(clip.start + clip.length, padLeft, plotW, viewEnd);
    const pad = LANE_PAD * dpr;
    const top = laneTop + pad;
    const h = laneH - 2 * pad;
    const w = Math.max(2 * dpr, x1 - x0);
    const selected = ti === this.selTrack && ci === this.selClip;

    const [r, g, b] = hexToRgb(track.color);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${selected ? 0.34 : 0.2})`;
    ctx.fillRect(x0, top, w, h);
    if (selected) {
      ctx.strokeStyle = OUTLINE;
      ctx.lineWidth = 1.5 * dpr;
      ctx.strokeRect(x0, top, w, h);
      ctx.lineWidth = 1;
    }

    // Thumbnail — clipped to the block.
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, top, w, h);
    ctx.clip();
    if (track.instrument.kind === "drums") {
      this.drawDrumThumb(clip, x0, top, w, h, viewEnd, plotW, padLeft);
    } else {
      this.drawSynthThumb(clip, x0, top, w, h, viewEnd, plotW, padLeft, track.color);
    }
    ctx.restore();
  }

  /** Synth clip: tiny dots at (note time, pitch) over the clip's own pitch range. */
  private drawSynthThumb(
    clip: Clip,
    x0: number,
    top: number,
    w: number,
    h: number,
    viewEnd: number,
    plotW: number,
    padLeft: number,
    color: string,
  ): void {
    const { ctx } = this;
    const dpr = this.dpr;
    if (clip.notes.length === 0) return;
    let lo = Infinity;
    let hi = -Infinity;
    for (const nt of clip.notes) {
      lo = Math.min(lo, nt.midi);
      hi = Math.max(hi, nt.midi);
    }
    const span = Math.max(1, hi - lo);
    const dotH = Math.max(1.5 * dpr, h / (span + 1));
    ctx.fillStyle = color;
    for (const nt of clip.notes) {
      const nx = this.beatToX(clip.start + nt.start, padLeft, plotW, viewEnd);
      const nw = Math.max(1.5 * dpr, this.beatToX(clip.start + nt.start + nt.duration, padLeft, plotW, viewEnd) - nx);
      const ny = top + (1 - (nt.midi - lo) / span) * (h - dotH);
      ctx.fillRect(nx, ny, Math.min(nw, x0 + w - nx), dotH);
    }
  }

  /** Drum clip: a dot per hit, row = lane index, x = beat. */
  private drawDrumThumb(
    clip: Clip,
    _x0: number,
    top: number,
    _w: number,
    h: number,
    viewEnd: number,
    plotW: number,
    padLeft: number,
  ): void {
    const { ctx } = this;
    const dpr = this.dpr;
    const rows = DRUM_LANES.length;
    const rowH = h / rows;
    const size = Math.max(1.5 * dpr, Math.min(3 * dpr, rowH * 0.7));
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    for (const nt of clip.notes) {
      const row = DRUM_LANES.findIndex((l) => l.midi === Math.round(nt.midi));
      if (row < 0) continue;
      const nx = this.beatToX(clip.start + nt.start, padLeft, plotW, viewEnd);
      const ny = top + row * rowH + (rowH - size) / 2;
      ctx.fillRect(nx, ny, size, size);
    }
  }
}
