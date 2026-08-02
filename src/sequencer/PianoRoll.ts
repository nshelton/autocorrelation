//! Canvas2D piano-roll — the classical timeline view, now an editor.
//!
//! Owns its own `<canvas>` overlay and rAF redraw loop, fully self-contained so
//! synth mode stays decoupled from the analysis App / WebGPU scene. The
//! component holds the mutable `Project` and emits `onChange` on each edit; the
//! caller re-serializes that project to the Sequencer so edits are heard live.
//!
//! Editing:
//!   - click empty space → paint a note, drag right to set its length
//!   - drag a note body → move (pitch + time); drag its right edge → resize
//!   - right-click a note → delete
//!   - times snap to a 1/16-note grid
//!
//! Selection (current track): Shift+drag empty space rubber-bands a marquee;
//! Shift+click toggles a note in/out. The marquee also grabs loose bend control
//! points — a note fully inside the box is selected whole (moves as a note),
//! otherwise just the breakpoints inside the box are selected (move on their
//! own). Dragging any selected thing moves the whole selection together.
//! Cmd/Ctrl+C / +V copy/paste notes (the block anchors at the playhead and
//! becomes the new selection); Delete/Backspace removes the selection.
//!
//! Pitch bend (synth tracks only — a drum's midi is a lane, not a pitch): each
//! note carries an optional piecewise-linear pitch envelope (`Note.bend`,
//! offsets from the base pitch). Double-click a note to drop a control point on
//! its curve; drag a point to bend (pitch snaps to a semitone, time to the grid,
//! Shift = free); right-click a point to delete it. Dragging the last point past
//! the note's end lengthens the note to reach it. A bent note renders as a
//! ribbon following the curve instead of a flat block. See `flattenBends` /
//! Rust `Sequencer.set_bends`.
//!
//! Navigation (Ableton-style scroll): a plain wheel/trackpad pans — vertical
//! over pitch, horizontal over time; Shift+wheel pans time (a vertical mouse
//! wheel scrolls sideways); Cmd/Ctrl+wheel (and the trackpad pinch, which
//! arrives as a ctrl-wheel) zooms the time axis around the cursor. The pitch
//! view is a center + octave-count and the time view is an independent
//! [start, end] beat window; both start at the loop on load and the toolbar
//! buttons still pan/zoom pitch by octaves.

import type { BendPoint, Clip, Note, PianoRollView, Project } from "./model";

/** Absolute pitch (MIDI, fractional) of a note's pitch-bend curve at `tRel`
 *  beats from the note's start. The head is implicit at `(t: 0, offset: 0)`;
 *  breakpoints are sorted by `t` and the final value is held to the note's end.
 *  No bend → flat at the base pitch. Mirrors Rust `interp_bend` (in MIDI space). */
function bendPitch(note: Note, tRel: number): number {
  const pts = note.bend;
  if (!pts || pts.length === 0 || tRel <= 0) return note.midi;
  let pt = 0; // previous breakpoint time (head)
  let po = 0; // previous breakpoint offset (head)
  for (const p of pts) {
    if (tRel <= p.t) {
      const f = p.t > pt ? (tRel - pt) / (p.t - pt) : 0;
      return note.midi + po + f * (p.offset - po);
    }
    pt = p.t;
    po = p.offset;
  }
  return note.midi + pts[pts.length - 1].offset; // held past the last point
}

const BG = "rgba(12, 12, 14, 0.92)";
const GRID = "rgba(204, 204, 204, 0.08)";
const GRID_STRONG = "rgba(204, 204, 204, 0.20)";
const BAR_LINE = "rgba(204, 204, 204, 0.34)"; // bar boundaries — bright + thick
const BEAT_LINE = "rgba(204, 204, 204, 0.12)"; // in-bar beats — faint + thin
const PLAYHEAD = "#ff5566";
const OUTLINE = "rgba(255, 255, 255, 0.9)";
const GOLD = "#ffcc55"; // loop region
const SELECT = "#7fd1ff"; // multi-selection outline + marquee

// Key-structure row tints: black keys (accidentals) sit recessed; rows that
// already contain notes get a green wash so the used pitches read at a glance.
const ROW_WHITE = "rgba(255, 255, 255, 0.03)";
const ROW_BLACK = "rgba(0, 0, 0, 0.22)";
const ROW_USED = "rgba(130, 220, 170, 0.13)";
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]); // C#, D#, F#, G#, A#

const BEATS_PER_BAR = 4; // 4/4 assumed until the model carries a time signature

// Scientific pitch notation: MIDI 60 = C4 (octave = floor(midi/12) - 1).
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const midiToName = (m: number): string =>
  `${NOTE_NAMES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`;
// Scale a #rrggbb color toward black (keep = fraction of brightness retained).
// Used to label a note in a darker shade of its own track color so the text
// reads as part of the block rather than jumping out.
const darken = (hex: string, keep: number): string => {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  const c = (shift: number) => Math.round(((n >> shift) & 0xff) * keep);
  return `rgb(${c(16)}, ${c(8)}, ${c(0)})`;
};

const SNAP = 0.25; // 1/16-note grid (beats)
const EDGE_PX = 7; // right-edge grab zone for resize (CSS px)
const BEND_DOT_PX = 7; // grab radius for a bend control point (CSS px)
const BEND_DOT_R = 3.5; // draw radius for a bend control point (CSS px)
const SCRUB_PX = 6; // grab zone around the playhead line for scrubbing (CSS px)
const LOOP_HANDLE_PX = 7; // grab zone around each loop-region edge (CSS px)
const RULER_H = 20; // top ruler / scrub band height (CSS px)
const DEFAULT_VELOCITY = 0.8;

const MAX_OCTAVES = 6;
const DEFAULT_OCTAVES = 2;
// Most-zoomed-in pitch view (1 octave = 12 semitones across the canvas). Auto-fit
// won't go tighter than DEFAULT_OCTAVES, but a user's explicitly-saved view can.
const MIN_OCTAVES = 1;
const MIN_CENTER = 24; // clamp pitch-view center so it can't pan off the keyboard
const MAX_CENTER = 96;
const MIN_VIEW_BEATS = 1; // most-zoomed-in time window (beats across the canvas)
const MAX_VIEW_BEATS = 128; // most-zoomed-out time window
const VIEW_TAIL_BEATS = 4; // empty beats you can scroll/zoom past the content end

export interface PianoRollCallbacks {
  /** Notes added / moved / resized / deleted — re-serialize the schedule. */
  onNotesChange?: (project: Project) => void;
  /** Loop length changed — re-send the transport loop bounds. */
  onLoopChange?: (project: Project) => void;
  /** Transport play/pause toggled from the toolbar. */
  onPlayPause?: () => void;
  /** Stop pressed (return to start). */
  onStop?: () => void;
  /** Playhead scrubbed to a beat (ruler click or dragging the line). */
  onSeek?: (beat: number) => void;
  /** Tempo changed from the BPM control. */
  onTempo?: (bpm: number) => void;
  /** Pitch/time zoom or scroll changed (user navigation) — persist the view. */
  onViewChange?: () => void;
}

interface Transform {
  padX: number;
  padTop: number;
  padBottom: number;
  plotW: number;
  plotH: number;
  W: number;
  H: number;
  viewStart: number; // beats
  viewEnd: number;
  viewLo: number; // midi
  viewHi: number;
  rowH: number; // device px per semitone
}

type Drag =
  // Move the selection: whole notes (boxed entirely) and/or individual bend
  // control points (boxed loose). Each entry snapshots its start; the one snapped
  // (beat, pitch) delta from the grab applies to all, preserving relative spacing.
  // (note.start is clip-relative, converted via the clip on apply; a point's t/
  // offset are relative to its note.)
  | {
      kind: "move";
      notes: Array<{ note: Note; clip: Clip; startAbs: number; startMidi: number }>;
      points: Array<{ note: Note; point: BendPoint; startT: number; startOffset: number }>;
      grabBeat: number;
      grabMidi: number;
    }
  | { kind: "resize"; note: Note; clip: Clip }
  // Dragging one pitch-bend breakpoint (time + pitch) on a note's envelope.
  | { kind: "bendPoint"; note: Note; clip: Clip; index: number }
  // Shift+drag rubber-band selection; coords are device px.
  | { kind: "marquee"; x0: number; y0: number; x1: number; y1: number }
  | { kind: "scrub" }
  | { kind: "loopEdge"; side: "start" | "end" };

const beatToX = (t: Transform, b: number) =>
  t.padX + ((b - t.viewStart) / (t.viewEnd - t.viewStart)) * t.plotW;
const xToBeat = (t: Transform, x: number) =>
  t.viewStart + ((x - t.padX) / t.plotW) * (t.viewEnd - t.viewStart);
const midiToY = (t: Transform, m: number) =>
  t.padTop + ((t.viewHi - m) / (t.viewHi - t.viewLo)) * t.plotH;
const yToMidi = (t: Transform, y: number) =>
  t.viewHi - ((y - t.padTop) / t.plotH) * (t.viewHi - t.viewLo);

const snap = (b: number) => Math.round(b / SNAP) * SNAP;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export class PianoRoll {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private project!: Project;
  private callbacks: PianoRollCallbacks;
  private playhead = 0;
  private raf = 0;
  private dpr = 1;

  // Time view follows the project loop; pitch view is a center + octave count
  // (so it's stable while editing, re-centered on content only on setProject).
  private viewStartBeat = 0;
  private viewEndBeat = 1;
  private viewCenterMidi = 60;
  private viewOctaves = DEFAULT_OCTAVES;

  /** Which track the piano-roll edits; other tracks render dimmed. */
  private selectedTrack = 0;

  // Interaction state.
  private drag: Drag | null = null;
  private hovered: Note | null = null;
  /** Multi-selected notes (selected track only). Drag/copy/delete act on these. */
  private selection = new Set<Note>();
  /** Multi-selected bend control points (of partially-boxed notes) — moved/deleted
   *  on their own without moving the whole note. */
  private pointSelection = new Set<BendPoint>();
  /** Copied notes, with `start` repurposed as the offset (in beats) from the
   *  earliest copied note, so paste can anchor the group at the playhead. */
  private clipboard: Note[] = [];

  private resizeObserver!: ResizeObserver;
  private onDown = (e: PointerEvent) => this.pointerDown(e);
  private onMove = (e: PointerEvent) => this.pointerMove(e);
  private onUp = (e: PointerEvent) => this.pointerUp(e);
  private onContext = (e: MouseEvent) => this.contextMenu(e);
  private onDblClick = (e: MouseEvent) => this.doubleClick(e);
  private onWheel = (e: WheelEvent) => this.wheel(e);
  private onKeyDown = (e: KeyboardEvent) => this.keyDown(e);

  constructor(project: Project, callbacks: PianoRollCallbacks = {}) {
    this.callbacks = callbacks;

    const canvas = document.createElement("canvas");
    canvas.className = "gui-el";
    Object.assign(canvas.style, {
      position: "fixed",
      left: "0",
      // Sandwiched between the track bar (top) and the synth bar (bottom),
      // each of which publishes its height as a CSS var.
      top: "var(--track-bar-h, 0px)",
      width: "100vw",
      height: "calc(100dvh - var(--track-bar-h, 0px) - var(--synth-panel-h, 0px))",
      zIndex: "5",
      touchAction: "none", // we handle pointer drags ourselves
      cursor: "crosshair",
    } satisfies Partial<CSSStyleDeclaration>);
    document.body.appendChild(canvas);
    this.canvas = canvas;

    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("[PianoRoll] 2D context unavailable");
    this.ctx = ctx;

    this.setProject(project);

    // Observe the canvas itself: this fires on window resize (it's 100vw) AND
    // when --synth-panel-h changes the computed height, so the backing store
    // always re-syncs without manual coordination with the panel.
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    canvas.addEventListener("pointerdown", this.onDown);
    canvas.addEventListener("pointermove", this.onMove);
    canvas.addEventListener("pointerup", this.onUp);
    canvas.addEventListener("contextmenu", this.onContext);
    canvas.addEventListener("dblclick", this.onDblClick);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    // Copy/paste/delete are global shortcuts (the canvas isn't focusable); the
    // handler no-ops when the roll is hidden (a drum track is the active editor).
    window.addEventListener("keydown", this.onKeyDown);
    this.resize();

    const loop = () => {
      this.draw();
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  setProject(project: Project): void {
    this.project = project;
    this.drag = null;
    this.hovered = null;
    this.selection.clear();
    this.pointSelection.clear();
    this.selectedTrack = 0;

    this.viewStartBeat = project.loopStart;
    this.viewEndBeat = Math.max(project.loopEnd, project.loopStart + 1);

    let lo = Infinity;
    let hi = -Infinity;
    for (const t of project.tracks) {
      for (const clip of t.clips) {
        for (const n of clip.notes) {
          lo = Math.min(lo, n.midi);
          hi = Math.max(hi, n.midi);
        }
      }
    }
    if (!Number.isFinite(lo)) {
      lo = 54;
      hi = 66;
    }
    // Center on the content and auto-fit the zoom so the note range fits with a
    // little padding; the user can pan/zoom from there.
    this.viewCenterMidi = clamp(Math.round((lo + hi) / 2), MIN_CENTER, MAX_CENTER);
    const neededOctaves = Math.ceil((hi - lo + 4) / 12);
    this.viewOctaves = clamp(neededOctaves, DEFAULT_OCTAVES, MAX_OCTAVES);

    // Keep the loop region sane / within the view.
    if (!(project.loopRegionEnd > project.loopRegionStart)) {
      project.loopRegionStart = project.loopStart;
      project.loopRegionEnd = project.loopEnd;
    }
  }

  setPlayhead(beat: number): void {
    // While scrubbing, the drag owns the playhead — ignore worklet echoes.
    if (this.drag?.kind === "scrub") return;
    this.playhead = beat;
  }

  /** Choose which track the roll edits (others render dimmed). */
  setSelectedTrack(index: number): void {
    this.selectedTrack = index;
    this.drag = null;
    this.hovered = null;
    this.selection.clear(); // selection is per-track
    this.pointSelection.clear();
  }

  /** Current zoom/scroll, for persistence (see `onViewChange`). */
  getView(): PianoRollView {
    return {
      start: this.viewStartBeat,
      end: this.viewEndBeat,
      center: this.viewCenterMidi,
      octaves: this.viewOctaves,
    };
  }

  /** Restore a saved view (clamped to the same bounds the gestures use), or —
   *  when there's none — auto-fit to the selected track's content. */
  setView(view: PianoRollView | undefined): void {
    if (!view) {
      this.fitToSelectedTrack();
      return;
    }
    const width = clamp(view.end - view.start, MIN_VIEW_BEATS, MAX_VIEW_BEATS);
    this.viewStartBeat = Math.max(0, view.start);
    this.viewEndBeat = this.viewStartBeat + width;
    this.viewCenterMidi = clamp(view.center, MIN_CENTER, MAX_CENTER);
    this.viewOctaves = clamp(view.octaves, MIN_OCTAVES, MAX_OCTAVES);
  }

  /** Frame the selected track's note range (time window = loop) — the default
   *  view for a track that has no saved one. Same fit math as `setProject`. */
  private fitToSelectedTrack(): void {
    this.viewStartBeat = this.project.loopStart;
    this.viewEndBeat = Math.max(this.project.loopEnd, this.project.loopStart + 1);
    let lo = Infinity;
    let hi = -Infinity;
    const track = this.project.tracks[this.selectedTrack];
    if (track) {
      for (const clip of track.clips) {
        for (const n of clip.notes) {
          lo = Math.min(lo, n.midi);
          hi = Math.max(hi, n.midi);
        }
      }
    }
    if (!Number.isFinite(lo)) {
      lo = 54;
      hi = 66;
    }
    this.viewCenterMidi = clamp(Math.round((lo + hi) / 2), MIN_CENTER, MAX_CENTER);
    this.viewOctaves = clamp(Math.ceil((hi - lo + 4) / 12), DEFAULT_OCTAVES, MAX_OCTAVES);
  }

  /** Show/hide the roll — a drum track swaps in the DrumMachine instead. Hidden
   *  canvases cost ~nothing (zero-size draw no-ops). */
  setVisible(visible: boolean): void {
    this.canvas.style.display = visible ? "" : "none";
  }

  /** Move the playhead to the beat under `x` and report the seek. */
  private scrubTo(t: Transform, x: number): void {
    const beat = clamp(xToBeat(t, x), this.viewStartBeat, this.viewEndBeat);
    this.playhead = beat;
    this.callbacks.onSeek?.(beat);
  }

  /** Which loop-region edge (if any) is under `x`. */
  private loopEdgeAt(t: Transform, x: number): "start" | "end" | null {
    const grab = LOOP_HANDLE_PX * this.dpr;
    if (Math.abs(x - beatToX(t, this.project.loopRegionStart)) <= grab) return "start";
    if (Math.abs(x - beatToX(t, this.project.loopRegionEnd)) <= grab) return "end";
    return null;
  }

  /** Drag a loop-region edge, snapped to whole beats, kept inside the view. */
  private dragLoopEdge(t: Transform, x: number, side: "start" | "end"): void {
    const p = this.project;
    const beat = clamp(Math.round(xToBeat(t, x)), this.viewStartBeat, this.viewEndBeat);
    if (side === "start") {
      p.loopRegionStart = Math.max(this.viewStartBeat, Math.min(beat, p.loopRegionEnd - 1));
    } else {
      p.loopRegionEnd = Math.min(this.viewEndBeat, Math.max(beat, p.loopRegionStart + 1));
    }
    this.callbacks.onLoopChange?.(p);
  }

  /** Drag one bend control point. Time is snapped to the grid (unless Shift)
   *  and kept strictly between its neighbours so points can't reorder; pitch is
   *  snapped to a semitone (unless Shift) and stored as an offset from the note's
   *  base. Mutates the breakpoint in place — the caller commits on pointer-up. */
  private dragBendPoint(
    d: Extract<Drag, { kind: "bendPoint" }>,
    beat: number,
    midiF: number,
    free: boolean,
    t: Transform,
  ): void {
    const pts = d.note.bend;
    if (!pts || !pts[d.index]) return;
    const absStart = d.clip.start + d.note.start;
    let tRel = beat - absStart;
    if (!free) tRel = Math.round(tRel / SNAP) * SNAP;
    const lo = d.index > 0 ? pts[d.index - 1].t : 0;
    // Interior points stay before their next neighbour; the last point may be
    // dragged out to the view edge, growing the note to reach it (below).
    const isLast = d.index === pts.length - 1;
    const hi = isLast ? t.viewEnd - absStart : pts[d.index + 1].t;
    tRel = clamp(tRel, lo + 1e-3, hi); // strictly after the previous point
    const pitch = clamp(midiF, t.viewLo, t.viewHi);
    const offset = (free ? pitch : Math.round(pitch)) - d.note.midi;
    pts[d.index] = { t: tRel, offset };
    // Dragging the last point past the note's end extends the note to reach it
    // (so you can bend out into new territory and lengthen the note in one drag).
    if (isLast && tRel > d.note.duration) d.note.duration = tRel;
  }


  // Scroll model (Ableton-style):
  //   - plain wheel/trackpad: vertical → pitch pan, horizontal (deltaX) → time pan
  //   - Shift+wheel:          time pan (a vertical mouse wheel scrolls sideways)
  //   - Cmd/Ctrl+wheel:       zoom the time axis around the cursor — this also
  //                           catches the trackpad pinch gesture, which the
  //                           browser delivers as a wheel event with ctrlKey set
  private wheel(e: WheelEvent): void {
    e.preventDefault();
    const t = this.transform();
    if (t.plotW <= 0 || t.plotH <= 0) return;
    const pxPerUnit = e.deltaMode === 1 ? 16 : 1; // line-mode deltas → ~px
    const dxPx = e.deltaX * pxPerUnit;
    const dyPx = e.deltaY * pxPerUnit;

    if (e.metaKey || e.ctrlKey) {
      const anchor = xToBeat(t, this.eventXY(e).x);
      this.zoomTimeAt(anchor, Math.pow(1.0015, dyPx)); // >1 zooms out, <1 zooms in
      return;
    }

    // Convert pixel deltas to view units so content tracks the gesture 1:1.
    const beatsPerPx = (t.viewEnd - t.viewStart) / (t.plotW / this.dpr);
    if (e.shiftKey) {
      this.panTime(dyPx * beatsPerPx);
      return;
    }
    if (dxPx !== 0) this.panTime(dxPx * beatsPerPx);
    if (dyPx !== 0) {
      const semisPerPx = (t.viewHi - t.viewLo) / (t.plotH / this.dpr);
      this.viewCenterMidi = clamp(this.viewCenterMidi - dyPx * semisPerPx, MIN_CENTER, MAX_CENTER);
      this.callbacks.onViewChange?.();
    }
  }

  /** Furthest beat any content reaches (across all tracks) — the right limit for
   *  time panning/zooming, plus a small tail of empty space. */
  private contentEndBeat(): number {
    let end = this.project.loopEnd;
    for (const tr of this.project.tracks) {
      for (const clip of tr.clips) {
        for (const n of clip.notes) end = Math.max(end, clip.start + n.start + n.duration);
      }
    }
    return end;
  }

  /** Slide the visible time window by `dBeats`, keeping its width and staying
   *  within `[0, contentEnd + tail]`. */
  private panTime(dBeats: number): void {
    const width = this.viewEndBeat - this.viewStartBeat;
    const maxStart = Math.max(0, this.contentEndBeat() + VIEW_TAIL_BEATS - width);
    this.viewStartBeat = clamp(this.viewStartBeat + dBeats, 0, maxStart);
    this.viewEndBeat = this.viewStartBeat + width;
    this.callbacks.onViewChange?.();
  }

  /** Scale the time window by `factor` about `anchorBeat` (the beat under the
   *  cursor stays fixed), clamped to the zoom + content bounds. */
  private zoomTimeAt(anchorBeat: number, factor: number): void {
    const width = this.viewEndBeat - this.viewStartBeat;
    const newWidth = clamp(width * factor, MIN_VIEW_BEATS, MAX_VIEW_BEATS);
    const ratio = newWidth / width;
    const start = anchorBeat - (anchorBeat - this.viewStartBeat) * ratio;
    const maxStart = Math.max(0, this.contentEndBeat() + VIEW_TAIL_BEATS - newWidth);
    this.viewStartBeat = clamp(start, 0, maxStart);
    this.viewEndBeat = this.viewStartBeat + newWidth;
    this.callbacks.onViewChange?.();
  }

  // --- Geometry --------------------------------------------------------

  private transform(): Transform {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const dpr = this.dpr;
    const padX = 16 * dpr;
    const padTop = RULER_H * dpr; // top band doubles as the scrub ruler
    const padBottom = 24 * dpr;
    const plotH = H - padTop - padBottom;
    const half = this.viewOctaves * 6; // semitones above/below the center
    const viewLo = this.viewCenterMidi - half;
    const viewHi = this.viewCenterMidi + half;
    return {
      padX,
      padTop,
      padBottom,
      plotW: W - 2 * padX,
      plotH,
      W,
      H,
      viewStart: this.viewStartBeat,
      viewEnd: this.viewEndBeat,
      viewLo,
      viewHi,
      rowH: plotH / (viewHi - viewLo),
    };
  }

  private eventXY(e: PointerEvent | MouseEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    // Canvas backing store is device px (clientSize × dpr); map CSS → device.
    return {
      x: (e.clientX - rect.left) * this.dpr,
      y: (e.clientY - rect.top) * this.dpr,
    };
  }

  /** Topmost editable note (in the selected track) under a (beat, midi) point.
   *  `beat` is absolute, `midi` continuous; note positions are clip-relative.
   *  Proximity is tested against the note's pitch-bend curve, so a bent note's
   *  whole ribbon is grabbable — not just its base row. */
  private noteAt(beat: number, midi: number): { clip: Clip; note: Note } | null {
    const track = this.project.tracks[this.selectedTrack];
    if (!track) return null;
    for (let ci = track.clips.length - 1; ci >= 0; ci--) {
      const clip = track.clips[ci];
      for (let ni = clip.notes.length - 1; ni >= 0; ni--) {
        const n = clip.notes[ni];
        const start = clip.start + n.start;
        if (
          beat >= start &&
          beat <= start + n.duration &&
          Math.abs(midi - bendPitch(n, beat - start)) < 0.5
        ) {
          return { clip, note: n };
        }
      }
    }
    return null;
  }

  /** Whether the selected track edits pitch (a synth, not a drum kit). Bend
   *  envelopes only apply to pitched tracks. */
  private isSynthSelected(): boolean {
    return this.project.tracks[this.selectedTrack]?.instrument.kind !== "drums";
  }

  /** Topmost bend control point (selected synth track) within grab range of the
   *  device-px point `(px, py)`. */
  private bendDotAt(t: Transform, px: number, py: number): { clip: Clip; note: Note; index: number } | null {
    if (!this.isSynthSelected()) return null;
    const track = this.project.tracks[this.selectedTrack];
    if (!track) return null;
    const grab = BEND_DOT_PX * this.dpr;
    for (let ci = track.clips.length - 1; ci >= 0; ci--) {
      const clip = track.clips[ci];
      for (let ni = clip.notes.length - 1; ni >= 0; ni--) {
        const n = clip.notes[ni];
        const pts = n.bend;
        if (!pts) continue;
        const absStart = clip.start + n.start;
        for (let i = pts.length - 1; i >= 0; i--) {
          const dx = beatToX(t, absStart + pts[i].t) - px;
          const dy = midiToY(t, n.midi + pts[i].offset) - py;
          if (dx * dx + dy * dy <= grab * grab) return { clip, note: n, index: i };
        }
      }
    }
    return null;
  }

  /** Clip new notes land in — the selected track's first clip, created if none.
   *  Null when there is no selected track. */
  private editClip(): Clip | null {
    const track = this.project.tracks[this.selectedTrack];
    if (!track) return null;
    if (track.clips.length === 0) {
      const length = Math.max(this.project.loopEnd, BEATS_PER_BAR);
      track.clips.push({ id: `${track.id}-clip`, start: 0, length, notes: [] });
    }
    return track.clips[0];
  }

  private commit(): void {
    this.callbacks.onNotesChange?.(this.project);
  }

  // --- Selection -------------------------------------------------------

  /** The selected track's clip that contains `note` (selection is per-track). */
  private clipOf(note: Note): Clip | null {
    const track = this.project.tracks[this.selectedTrack];
    if (!track) return null;
    for (const clip of track.clips) if (clip.notes.includes(note)) return clip;
    return null;
  }

  /** Absolute arrangement-beat start of a note (clip start + clip-relative start). */
  private absStartOf(note: Note): number {
    return (this.clipOf(note)?.start ?? 0) + note.start;
  }

  /** Rebuild the marquee selection. A flat note is selected when the box touches
   *  it; a bent note is selected (as a whole) only when the box *contains* it,
   *  otherwise its individual control points inside the box are point-selected —
   *  so you can box just some breakpoints and move them without the note. */
  private updateMarqueeSelection(t: Transform): void {
    if (this.drag?.kind !== "marquee") return;
    const d = this.drag;
    const loX = Math.min(d.x0, d.x1);
    const hiX = Math.max(d.x0, d.x1);
    const loY = Math.min(d.y0, d.y1);
    const hiY = Math.max(d.y0, d.y1);
    const track = this.project.tracks[this.selectedTrack];
    this.selection.clear();
    this.pointSelection.clear();
    if (!track) return;
    for (const clip of track.clips) {
      for (const n of clip.notes) {
        const absStart = clip.start + n.start;
        const x = beatToX(t, absStart);
        const x2 = beatToX(t, absStart + n.duration);
        let pLo = n.midi;
        let pHi = n.midi;
        if (n.bend)
          for (const p of n.bend) {
            pLo = Math.min(pLo, n.midi + p.offset);
            pHi = Math.max(pHi, n.midi + p.offset);
          }
        const yTop = midiToY(t, pHi) - t.rowH * 0.45;
        const yBot = midiToY(t, pLo) + t.rowH * 0.45;
        const intersects = x <= hiX && x2 >= loX && yTop <= hiY && yBot >= loY;
        const contained = x >= loX && x2 <= hiX && yTop >= loY && yBot <= hiY;
        if (!n.bend || n.bend.length === 0) {
          if (intersects) this.selection.add(n);
        } else if (contained) {
          this.selection.add(n); // whole bent note boxed → moves as a note
        } else {
          for (const p of n.bend) {
            const px = beatToX(t, absStart + p.t);
            const py = midiToY(t, n.midi + p.offset);
            if (px >= loX && px <= hiX && py >= loY && py <= hiY) this.pointSelection.add(p);
          }
        }
      }
    }
  }

  /** The selected control points paired with their owning notes (scans the
   *  selected track, since `pointSelection` stores bare point objects). */
  private selectedPointItems(): Array<{ note: Note; point: BendPoint }> {
    const out: Array<{ note: Note; point: BendPoint }> = [];
    if (this.pointSelection.size === 0) return out;
    const track = this.project.tracks[this.selectedTrack];
    if (!track) return out;
    for (const clip of track.clips) {
      for (const n of clip.notes) {
        if (!n.bend) continue;
        for (const p of n.bend) if (this.pointSelection.has(p)) out.push({ note: n, point: p });
      }
    }
    return out;
  }

  /** Snapshot the current selection (notes + points) into a move drag. */
  private beginMove(grabBeat: number, grabMidi: number): Drag {
    const notes: Array<{ note: Note; clip: Clip; startAbs: number; startMidi: number }> = [];
    for (const note of this.selection) {
      const clip = this.clipOf(note);
      if (clip) notes.push({ note, clip, startAbs: clip.start + note.start, startMidi: note.midi });
    }
    const points = this.selectedPointItems().map(({ note, point }) => ({
      note,
      point,
      startT: point.t,
      startOffset: point.offset,
    }));
    return { kind: "move", notes, points, grabBeat, grabMidi };
  }

  /** Cmd/Ctrl+C — snapshot the selection (deep, incl. bend) with each note's
   *  start re-expressed as an offset from the earliest, so paste can anchor it. */
  private copySelection(): boolean {
    if (this.selection.size === 0) return false;
    const notes = [...this.selection];
    const base = Math.min(...notes.map((n) => this.absStartOf(n)));
    this.clipboard = notes.map((n) => ({
      start: this.absStartOf(n) - base,
      duration: n.duration,
      midi: n.midi,
      velocity: n.velocity,
      bend: n.bend?.map((p) => ({ ...p })),
    }));
    return true;
  }

  /** Cmd/Ctrl+V — paste the clipboard into the edit clip with the earliest note
   *  anchored at the playhead (snapped); the pasted notes become the selection. */
  private pasteClipboard(): boolean {
    if (this.clipboard.length === 0) return false;
    const clip = this.editClip();
    if (!clip) return false;
    const anchor = Math.max(0, snap(this.playhead));
    this.selection.clear();
    for (const c of this.clipboard) {
      const note: Note = {
        start: Math.max(0, anchor + c.start - clip.start),
        duration: c.duration,
        midi: c.midi,
        velocity: c.velocity,
        bend: c.bend?.map((p) => ({ ...p })),
      };
      clip.notes.push(note);
      this.selection.add(note);
    }
    this.commit();
    return true;
  }

  /** Delete/Backspace — remove the selected notes and any selected bend points. */
  private deleteSelection(): boolean {
    if (this.selection.size === 0 && this.pointSelection.size === 0) return false;
    const track = this.project.tracks[this.selectedTrack];
    if (!track) return false;
    for (const clip of track.clips) {
      clip.notes = clip.notes.filter((n) => !this.selection.has(n));
      if (this.pointSelection.size) {
        for (const n of clip.notes) {
          if (n.bend) n.bend = n.bend.filter((p) => !this.pointSelection.has(p));
        }
      }
    }
    this.selection.clear();
    this.pointSelection.clear();
    this.hovered = null;
    this.commit();
    return true;
  }

  private keyDown(e: KeyboardEvent): void {
    // No-op when the roll is hidden (a drum track owns the editor) or when the
    // user is typing into a field.
    if (this.canvas.style.display === "none") return;
    const tgt = e.target as HTMLElement | null;
    if (tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA" || tgt.isContentEditable)) {
      return;
    }
    const mod = e.metaKey || e.ctrlKey;
    let handled = false;
    if (mod && (e.key === "c" || e.key === "C")) handled = this.copySelection();
    else if (mod && (e.key === "v" || e.key === "V")) handled = this.pasteClipboard();
    else if (e.key === "Delete" || e.key === "Backspace") handled = this.deleteSelection();
    if (handled) e.preventDefault();
  }

  // --- Pointer interaction ---------------------------------------------

  private pointerDown(e: PointerEvent): void {
    if (e.button === 2) return; // delete is handled by contextmenu
    const t = this.transform();
    if (t.plotW <= 0 || t.plotH <= 0) return;
    const { x, y } = this.eventXY(e);
    const beat = xToBeat(t, x);
    const midiF = yToMidi(t, y);
    const midi = Math.round(midiF);

    const inRuler = y <= t.padTop;
    const hit = inRuler ? null : this.noteAt(beat, midiF);

    // Loop-region edge handles (ruler, when looping is on) beat scrubbing.
    if (inRuler && this.project.loopEnabled) {
      const side = this.loopEdgeAt(t, x);
      if (side) {
        this.drag = { kind: "loopEdge", side };
        this.canvas.setPointerCapture(e.pointerId);
        return;
      }
    }

    // Scrub the top ruler band (loop-edge handles already checked above).
    if (inRuler) {
      this.drag = { kind: "scrub" };
      this.canvas.setPointerCapture(e.pointerId);
      this.scrubTo(t, x);
      return;
    }

    // Shift is the selection modifier (so a plain drag still paints): Shift+click
    // a note toggles it in/out of the selection; Shift+drag empty space rubber-
    // band-selects (a zero-area box clears the selection).
    if (e.shiftKey) {
      if (hit) {
        if (this.selection.has(hit.note)) this.selection.delete(hit.note);
        else this.selection.add(hit.note);
      } else {
        this.drag = { kind: "marquee", x0: x, y0: y, x1: x, y1: y };
        this.canvas.setPointerCapture(e.pointerId);
        this.updateMarqueeSelection(t);
      }
      return;
    }

    // Grab the playhead line in empty space (a note under it still edits).
    const nearLine = Math.abs(x - beatToX(t, this.playhead)) <= SCRUB_PX * this.dpr;
    if (nearLine && !hit) {
      this.drag = { kind: "scrub" };
      this.canvas.setPointerCapture(e.pointerId);
      this.scrubTo(t, x);
      return;
    }

    // Grab a pitch-bend control point — takes priority over moving the note it
    // sits on, since the dot is the smaller, more specific target. If the point
    // is part of the selection, drag the whole selection (notes + points);
    // otherwise edit just this one.
    const dot = this.bendDotAt(t, x, y);
    if (dot) {
      const point = dot.note.bend?.[dot.index];
      this.drag =
        point && this.pointSelection.has(point)
          ? this.beginMove(beat, midiF)
          : { kind: "bendPoint", note: dot.note, clip: dot.clip, index: dot.index };
      this.canvas.setPointerCapture(e.pointerId);
      return;
    }

    if (beat < t.viewStart || beat > t.viewEnd || midi < t.viewLo || midi > t.viewHi) {
      return;
    }
    if (hit) {
      const absStart = hit.clip.start + hit.note.start;
      const leftX = beatToX(t, absStart);
      const rightX = beatToX(t, absStart + hit.note.duration);
      const wide = rightX - leftX > 2 * EDGE_PX * this.dpr;
      if (wide && x >= rightX - EDGE_PX * this.dpr) {
        this.drag = { kind: "resize", note: hit.note, clip: hit.clip };
      } else {
        // Grabbing an unselected note makes it the sole selection; then move the
        // whole selection together (notes + any selected points).
        if (!this.selection.has(hit.note)) {
          this.selection.clear();
          this.pointSelection.clear();
          this.selection.add(hit.note);
        }
        this.drag = this.beginMove(beat, midiF);
      }
    } else {
      // Paint a new note and immediately resize-drag it to set its length; the
      // new note becomes the selection.
      const clip = this.editClip();
      if (!clip) return;
      const note: Note = {
        start: Math.max(0, snap(beat - clip.start)),
        duration: SNAP,
        midi,
        velocity: DEFAULT_VELOCITY,
      };
      clip.notes.push(note);
      this.selection.clear();
      this.pointSelection.clear();
      this.selection.add(note);
      this.drag = { kind: "resize", note, clip };
    }
    this.canvas.setPointerCapture(e.pointerId);
  }

  private pointerMove(e: PointerEvent): void {
    const t = this.transform();
    if (t.plotW <= 0 || t.plotH <= 0) return;
    const { x, y } = this.eventXY(e);
    const beat = xToBeat(t, x);
    const midiF = yToMidi(t, y);

    if (this.drag?.kind === "loopEdge") {
      this.dragLoopEdge(t, x, this.drag.side);
      return;
    }
    if (this.drag?.kind === "scrub") {
      this.scrubTo(t, x);
      return;
    }
    if (this.drag?.kind === "bendPoint") {
      this.dragBendPoint(this.drag, beat, midiF, e.shiftKey, t);
      return;
    }
    if (this.drag?.kind === "marquee") {
      this.drag.x1 = x;
      this.drag.y1 = y;
      this.updateMarqueeSelection(t);
      return;
    }

    if (!this.drag) {
      // Scrub / loop-edge zones take cursor priority over note hover.
      const inRuler = y <= t.padTop;
      const nearLine = Math.abs(x - beatToX(t, this.playhead)) <= SCRUB_PX * this.dpr;
      // Hover feedback: highlight + cursor shape.
      const hit = inRuler ? null : this.noteAt(beat, midiF);
      this.hovered = hit?.note ?? null;
      // Shift = selection mode: hint marquee (empty) vs toggle (on a note).
      if (e.shiftKey && !inRuler) {
        this.canvas.style.cursor = hit ? "pointer" : "cell";
        return;
      }
      if (inRuler && this.project.loopEnabled && this.loopEdgeAt(t, x)) {
        this.canvas.style.cursor = "ew-resize";
        return;
      }
      if (inRuler || (nearLine && !hit)) {
        this.canvas.style.cursor = "col-resize";
        return;
      }
      // A bend control point takes cursor priority over the note under it.
      if (!inRuler && this.bendDotAt(t, x, y)) {
        this.canvas.style.cursor = "pointer";
        return;
      }
      if (hit) {
        const absStart = hit.clip.start + hit.note.start;
        const rightX = beatToX(t, absStart + hit.note.duration);
        const leftX = beatToX(t, absStart);
        const wide = rightX - leftX > 2 * EDGE_PX * this.dpr;
        this.canvas.style.cursor =
          wide && x >= rightX - EDGE_PX * this.dpr ? "ew-resize" : "move";
      } else {
        this.canvas.style.cursor = "crosshair";
      }
      return;
    }

    if (this.drag.kind === "move") {
      const d = this.drag;
      // Snap the (beat, pitch) delta — not each item — so relative spacing is
      // preserved, then clamp it so every moved thing stays in bounds (rigidly).
      let dBeat = snap(beat - d.grabBeat);
      let dMidi = Math.round(midiF - d.grabMidi);
      let dbLo = -Infinity;
      let dbHi = Infinity;
      let dmLo = -Infinity;
      let dmHi = Infinity;
      for (const it of d.notes) {
        dbLo = Math.max(dbLo, t.viewStart - it.startAbs);
        dbHi = Math.min(dbHi, t.viewEnd - (it.startAbs + it.note.duration));
        dmLo = Math.max(dmLo, t.viewLo - it.startMidi);
        dmHi = Math.min(dmHi, t.viewHi - it.startMidi);
      }
      for (const it of d.points) {
        // A point stays within its note's [~0, duration] and the pitch view.
        dbLo = Math.max(dbLo, 1e-3 - it.startT);
        dbHi = Math.min(dbHi, it.note.duration - it.startT);
        const absPitch = it.note.midi + it.startOffset;
        dmLo = Math.max(dmLo, t.viewLo - absPitch);
        dmHi = Math.min(dmHi, t.viewHi - absPitch);
      }
      dBeat = dbLo <= dbHi ? clamp(dBeat, dbLo, dbHi) : 0;
      dMidi = dmLo <= dmHi ? clamp(dMidi, dmLo, dmHi) : 0;
      for (const it of d.notes) {
        it.note.start = Math.max(0, it.startAbs + dBeat - it.clip.start);
        it.note.midi = it.startMidi + dMidi;
      }
      if (d.points.length) {
        for (const it of d.points) {
          it.point.t = it.startT + dBeat;
          it.point.offset = it.startOffset + dMidi;
        }
        // Points may have crossed unselected ones; re-sort each affected note.
        const touched = new Set(d.points.map((it) => it.note));
        for (const n of touched) n.bend?.sort((a, b) => a.t - b.t);
      }
    } else {
      const d = this.drag;
      const absStart = d.clip.start + d.note.start;
      const dur = snap(beat - absStart);
      d.note.duration = Math.max(SNAP, Math.min(dur, t.viewEnd - absStart));
      // Keep bend control points inside the (possibly shortened) note so the
      // ribbon's held tail and the dots stay within its time span.
      if (d.note.bend) for (const p of d.note.bend) p.t = Math.min(p.t, d.note.duration);
    }
  }

  private pointerUp(e: PointerEvent): void {
    if (!this.drag) return;
    // Only note edits re-serialize the schedule; scrub/loop-edge already acted live.
    const isEdit =
      this.drag.kind === "move" ||
      this.drag.kind === "resize" ||
      this.drag.kind === "bendPoint";
    this.drag = null;
    this.canvas.releasePointerCapture(e.pointerId);
    if (isEdit) this.commit();
  }

  private contextMenu(e: MouseEvent): void {
    e.preventDefault();
    const t = this.transform();
    const { x, y } = this.eventXY(e);

    // Right-click a bend control point deletes just that point (not the note).
    const dot = this.bendDotAt(t, x, y);
    if (dot) {
      dot.note.bend?.splice(dot.index, 1);
      this.commit();
      return;
    }

    const hit = this.noteAt(xToBeat(t, x), yToMidi(t, y));
    if (!hit) return;
    const idx = hit.clip.notes.indexOf(hit.note);
    if (idx >= 0) {
      hit.clip.notes.splice(idx, 1);
      this.selection.delete(hit.note);
      if (this.hovered === hit.note) this.hovered = null;
      this.commit();
    }
  }

  /** Double-click a note (selected synth track) to add a bend control point on
   *  its pitch curve at the clicked time — then drag it to bend. Placing the
   *  point *on* the current curve means adding it doesn't change the sound until
   *  it's dragged. */
  private doubleClick(e: MouseEvent): void {
    e.preventDefault();
    if (!this.isSynthSelected()) return;
    const t = this.transform();
    if (t.plotW <= 0 || t.plotH <= 0) return;
    const { x, y } = this.eventXY(e);
    const beat = xToBeat(t, x);
    const hit = this.noteAt(beat, yToMidi(t, y));
    if (!hit) return;
    const { note, clip } = hit;
    const absStart = clip.start + note.start;
    let tRel = Math.round((beat - absStart) / SNAP) * SNAP;
    tRel = clamp(tRel, 1e-3, note.duration); // strictly inside the note
    const offset = bendPitch(note, tRel) - note.midi; // land on the existing curve
    const bend = (note.bend ??= []);
    bend.push({ t: tRel, offset });
    bend.sort((a, b) => a.t - b.t);
    this.commit();
  }

  // --- Rendering -------------------------------------------------------

  /** Trace the ribbon following a note's pitch-bend curve into the current path
   *  (caller fills or strokes). Constant vertical thickness centred on the
   *  curve; the last breakpoint's pitch is held to the note's end. */
  private bendRibbonPath(t: Transform, n: Note, absStart: number): void {
    const ctx = this.ctx;
    const hh = t.rowH * 0.45;
    const pts = n.bend ?? [];
    // Curve vertices in (beat, midi): head, each breakpoint, then the held end.
    const verts: Array<[number, number]> = [[absStart, n.midi]];
    for (const p of pts) verts.push([absStart + p.t, n.midi + p.offset]);
    const endOffset = pts.length ? pts[pts.length - 1].offset : 0;
    verts.push([absStart + n.duration, n.midi + endOffset]);
    ctx.beginPath();
    for (let i = 0; i < verts.length; i++) {
      const vx = beatToX(t, verts[i][0]);
      const vy = midiToY(t, verts[i][1]) - hh;
      if (i === 0) ctx.moveTo(vx, vy);
      else ctx.lineTo(vx, vy);
    }
    for (let i = verts.length - 1; i >= 0; i--) {
      ctx.lineTo(beatToX(t, verts[i][0]), midiToY(t, verts[i][1]) + hh);
    }
    ctx.closePath();
  }

  private resize(): void {
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * this.dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * this.dpr);
  }

  private draw(): void {
    const { ctx } = this;
    const t = this.transform();
    ctx.clearRect(0, 0, t.W, t.H);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, t.W, t.H);
    if (t.plotW <= 0 || t.plotH <= 0) return;

    // Top ruler / scrub band.
    ctx.fillStyle = "rgba(204, 204, 204, 0.05)";
    ctx.fillRect(0, 0, t.W, t.padTop);

    // The white "active" outline marks a single note being resized/bent or
    // hovered; a move acts on the whole selection (shown via the cyan outline).
    const active =
      this.drag?.kind === "resize" || this.drag?.kind === "bendPoint"
        ? this.drag.note
        : this.hovered;

    // Key-structure row tints + used-pitch highlight, clipped to the plot so
    // they don't bleed into the ruler.
    const used = new Set<number>();
    for (const track of this.project.tracks) {
      for (const clip of track.clips) {
        for (const n of clip.notes) used.add(n.midi);
      }
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(t.padX, t.padTop, t.plotW, t.plotH);
    ctx.clip();
    for (let m = Math.floor(t.viewLo); m <= Math.ceil(t.viewHi); m++) {
      const top = midiToY(t, m + 0.5); // row m spans [m+0.5, m-0.5]
      const pc = ((m % 12) + 12) % 12;
      ctx.fillStyle = BLACK_KEYS.has(pc) ? ROW_BLACK : ROW_WHITE;
      ctx.fillRect(t.padX, top, t.plotW, t.rowH);
      if (used.has(m)) {
        ctx.fillStyle = ROW_USED;
        ctx.fillRect(t.padX, top, t.plotW, t.rowH);
      }
    }
    ctx.restore();

    // Row separators — faint per-semitone, stronger at octave boundaries (C).
    ctx.lineWidth = 1;
    for (let m = Math.ceil(t.viewLo); m <= Math.ceil(t.viewHi) + 1; m++) {
      const y = midiToY(t, m - 0.5); // boundary below row m
      ctx.strokeStyle = ((m % 12) + 12) % 12 === 0 ? GRID_STRONG : GRID;
      ctx.beginPath();
      ctx.moveTo(t.padX, y);
      ctx.lineTo(t.W - t.padX, y);
      ctx.stroke();
    }

    // Beat / bar gridlines — bar boundaries (every BEATS_PER_BAR) are drawn
    // thicker and brighter than the in-bar beats.
    for (let b = Math.ceil(t.viewStart); b <= t.viewEnd; b++) {
      const x = beatToX(t, b);
      const isBar = b % BEATS_PER_BAR === 0;
      ctx.strokeStyle = isBar ? BAR_LINE : BEAT_LINE;
      ctx.lineWidth = (isBar ? 2 : 1) * this.dpr;
      ctx.beginPath();
      ctx.moveTo(x, t.padTop);
      ctx.lineTo(x, t.H - t.padBottom);
      ctx.stroke();
    }
    ctx.lineWidth = 1;

    // Notes. The selected track is fully lit (sounding notes flash white,
    // hovered/dragging note outlined); other tracks render dimmed for context.
    // Clipped to the plot so panned/zoomed notes don't bleed into the ruler.
    ctx.save();
    ctx.beginPath();
    ctx.rect(t.padX, t.padTop, t.plotW, t.plotH);
    ctx.clip();
    this.project.tracks.forEach((track, ti) => {
      const selected = ti === this.selectedTrack;
      const labelColor = darken(track.color, 0.45);
      for (const clip of track.clips) {
        for (const n of clip.notes) {
          const absStart = clip.start + n.start;
          const x = beatToX(t, absStart);
          const x2 = beatToX(t, absStart + n.duration);
          const y = midiToY(t, n.midi) - t.rowH * 0.45;
          const w = Math.max(2 * this.dpr, x2 - x);
          const h = t.rowH * 0.9;
          const playing =
            selected && this.playhead >= absStart && this.playhead < absStart + n.duration;
          // A note with a pitch-bend envelope draws as a ribbon following the
          // curve; a flat note stays the classic block. Drums are never bent.
          const hasBend = !!n.bend && n.bend.length > 0 && track.instrument.kind !== "drums";
          ctx.globalAlpha = selected ? 0.35 + 0.65 * n.velocity : 0.16;
          ctx.fillStyle = playing ? "#ffffff" : track.color;
          if (hasBend) {
            this.bendRibbonPath(t, n, absStart);
            ctx.fill();
          } else {
            ctx.fillRect(x, y, w, h);
          }
          ctx.globalAlpha = 1;
          if (selected && n === active) {
            ctx.strokeStyle = OUTLINE;
            ctx.lineWidth = 1.5 * this.dpr;
            if (hasBend) {
              this.bendRibbonPath(t, n, absStart);
              ctx.stroke();
            } else {
              ctx.strokeRect(x, y, w, h);
            }
          }
          // Multi-selection outline (cyan) — drag/copy/delete act on these.
          if (selected && this.selection.has(n)) {
            ctx.strokeStyle = SELECT;
            ctx.lineWidth = 2 * this.dpr;
            if (hasBend) {
              this.bendRibbonPath(t, n, absStart);
              ctx.stroke();
            } else {
              ctx.strokeRect(x, y, w, h);
            }
          }
          // Note name at the head (base pitch) — only on the editable track, and
          // only when there's room (avoids unreadable clutter on tiny notes).
          if (selected && h >= 9 * this.dpr) {
            const fontPx = clamp(h * 0.6, 8 * this.dpr, 13 * this.dpr);
            ctx.font = `${fontPx}px ui-sans-serif, system-ui`;
            const label = midiToName(n.midi);
            const padPx = 3 * this.dpr;
            if (w >= ctx.measureText(label).width + 2 * padPx) {
              ctx.fillStyle = labelColor;
              ctx.textBaseline = "middle";
              ctx.fillText(label, x + padPx, y + h / 2);
            }
          }
          // Bend control points (selected track) — the draggable handles;
          // point-selected ones are filled cyan and a touch larger.
          if (selected && hasBend) {
            ctx.lineWidth = 1 * this.dpr;
            for (const p of n.bend!) {
              const sel = this.pointSelection.has(p);
              const r = (sel ? BEND_DOT_R + 1 : BEND_DOT_R) * this.dpr;
              ctx.fillStyle = sel ? SELECT : "#ffffff";
              ctx.strokeStyle = sel ? "#ffffff" : "rgba(0, 0, 0, 0.6)";
              ctx.beginPath();
              ctx.arc(beatToX(t, absStart + p.t), midiToY(t, n.midi + p.offset), r, 0, Math.PI * 2);
              ctx.fill();
              ctx.stroke();
            }
          }
        }
      }
    });
    ctx.restore();

    // Loop region (gold) — faint full-height fill, a brighter band in the
    // ruler, and draggable edge lines.
    if (this.project.loopEnabled) {
      const rsX = beatToX(t, this.project.loopRegionStart);
      const reX = beatToX(t, this.project.loopRegionEnd);
      ctx.fillStyle = "rgba(255, 204, 85, 0.06)";
      ctx.fillRect(rsX, t.padTop, reX - rsX, t.H - t.padBottom - t.padTop);
      ctx.fillStyle = "rgba(255, 204, 85, 0.28)";
      ctx.fillRect(rsX, 0, reX - rsX, t.padTop);
      ctx.strokeStyle = GOLD;
      ctx.lineWidth = 2 * this.dpr;
      for (const ex of [rsX, reX]) {
        ctx.beginPath();
        ctx.moveTo(ex, 0);
        ctx.lineTo(ex, t.H - t.padBottom);
        ctx.stroke();
      }
    }

    // Playhead — full height through the ruler, with a grab handle on top.
    const px = beatToX(t, this.playhead);
    ctx.strokeStyle = PLAYHEAD;
    ctx.lineWidth = 2 * this.dpr;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, t.H - t.padBottom);
    ctx.stroke();
    const hw = 5 * this.dpr;
    ctx.fillStyle = PLAYHEAD;
    ctx.beginPath();
    ctx.moveTo(px - hw, 0);
    ctx.lineTo(px + hw, 0);
    ctx.lineTo(px, t.padTop * 0.7);
    ctx.closePath();
    ctx.fill();

    // Rubber-band marquee (Shift+drag).
    if (this.drag?.kind === "marquee") {
      const d = this.drag;
      const rx = Math.min(d.x0, d.x1);
      const ry = Math.min(d.y0, d.y1);
      const rw = Math.abs(d.x1 - d.x0);
      const rh = Math.abs(d.y1 - d.y0);
      ctx.fillStyle = "rgba(127, 209, 255, 0.12)";
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = SELECT;
      ctx.lineWidth = 1 * this.dpr;
      ctx.setLineDash([4 * this.dpr, 3 * this.dpr]);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);
    }
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener("pointerdown", this.onDown);
    this.canvas.removeEventListener("pointermove", this.onMove);
    this.canvas.removeEventListener("pointerup", this.onUp);
    this.canvas.removeEventListener("contextmenu", this.onContext);
    this.canvas.removeEventListener("dblclick", this.onDblClick);
    this.canvas.removeEventListener("wheel", this.onWheel);
    window.removeEventListener("keydown", this.onKeyDown);
    this.canvas.remove();
  }
}
