//! ReDrum-style step sequencer — the percussion editor, a sibling to the
//! Canvas2D `PianoRoll`. Self-contained DOM (no UI framework), so synth mode
//! stays decoupled from the analysis App / WebGPU scene.
//!
//! It edits the selected *drum* track's clip, but a drum pattern is just notes:
//! each lane is a fixed MIDI note (see `drumkit.ts`), and a lit step is a short
//! note at that step's beat with a velocity set by the H/M/S level. So the same
//! `flattenToSchedule` → transport → Rust `DrumKit` path the piano-roll uses
//! carries drum patterns unchanged; this view only authors the notes.
//!
//! The grid spans the loop window `[loopStart, loopEnd]` as `DRUM_STEPS` steps,
//! so the default 4-beat loop yields the classic 16-step / 1-bar pattern.

import {
  DRUM_LANES,
  DRUM_STEPS,
  LEVEL_VELOCITY,
  beatToStep,
  stepDuration,
  velocityToLevel,
  type Level,
} from "./drumkit";
import type { Clip, Note, Project } from "./model";

export interface DrumMachineCallbacks {
  /** Steps toggled — re-serialize the schedule. */
  onNotesChange?: (project: Project) => void;
  /** Audition a lane (clicking its label) — trigger that drum once, live. */
  onAudition?: (midi: number) => void;
}

const LEVELS: Level[] = ["soft", "med", "hard"];
const LEVEL_LABEL: Record<Level, string> = { soft: "S", med: "M", hard: "H" };
const LEVEL_ALPHA: Record<Level, number> = { soft: 0.34, med: 0.66, hard: 1.0 };
const BEATS_PER_BAR = 4;

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16) || 0,
    parseInt(h.slice(2, 4), 16) || 0,
    parseInt(h.slice(4, 6), 16) || 0,
  ];
}

let stylesInjected = false;
function injectStyles(): void {
  if (stylesInjected) return;
  stylesInjected = true;
  const css = `
.dm-root { position:fixed; left:0; z-index:5;
  top:var(--track-bar-h, 0px);
  width:100vw; height:calc(100dvh - var(--track-bar-h, 0px) - var(--synth-panel-h, 0px));
  background:rgba(12,12,14,0.92); color:#cccccc;
  font:11px/1 ui-sans-serif,system-ui; pointer-events:auto; user-select:none;
  display:flex; flex-direction:column; box-sizing:border-box; }
.dm-toolbar { display:flex; gap:16px; align-items:center; padding:7px 12px;
  border-bottom:1px solid rgba(204,204,204,0.12); flex:0 0 auto; }
.dm-group { display:flex; align-items:center; gap:4px; }
.dm-label { opacity:0.55; letter-spacing:0.1em; margin-right:2px; }
.dm-btn { min-width:18px; text-align:center; padding:3px 8px; border-radius:3px;
  cursor:pointer; background:rgba(204,204,204,0.1); border:1px solid transparent; }
.dm-btn:hover { background:rgba(204,204,204,0.18); }
.dm-btn-active { background:rgba(127,209,255,0.20); border-color:#7fd1ff; color:#ffffff; }
.dm-grid { flex:1 1 auto; display:flex; flex-direction:column; gap:3px;
  padding:8px 12px 12px; box-sizing:border-box; overflow:hidden; }
.dm-header { display:flex; align-items:center; gap:6px; flex:0 0 16px; }
.dm-row { display:flex; align-items:stretch; gap:6px; flex:1 1 0; min-height:18px; }
.dm-name { flex:0 0 92px; display:flex; align-items:center; padding:0 8px;
  border-radius:3px; background:rgba(204,204,204,0.06); cursor:pointer;
  white-space:nowrap; overflow:hidden; }
.dm-name:hover { background:rgba(204,204,204,0.14); }
.dm-steps { flex:1 1 auto; display:grid; grid-template-columns:repeat(${DRUM_STEPS}, 1fr);
  gap:3px; }
.dm-numbers { flex:1 1 auto; display:grid; grid-template-columns:repeat(${DRUM_STEPS}, 1fr);
  gap:3px; }
.dm-num { text-align:center; opacity:0.4; font-variant-numeric:tabular-nums; }
.dm-spacer { flex:0 0 92px; }
.dm-cell { border-radius:3px; background:rgba(204,204,204,0.07); cursor:pointer;
  touch-action:none; transition:background 0.04s; }
.dm-cell:hover { background:rgba(204,204,204,0.16); }
/* Quarter-note group boundaries: brighten every 4th step (the downbeats). */
.dm-cell.dm-beat, .dm-num.dm-beat { background:rgba(204,204,204,0.11); }
.dm-num.dm-beat { background:none; opacity:0.75; }
/* Current step under the playhead. */
.dm-cell.dm-playcol { box-shadow:inset 0 0 0 2px rgba(255,255,255,0.65); }`;
  const el = document.createElement("style");
  el.textContent = css;
  document.head.appendChild(el);
}

export class DrumMachine {
  private root: HTMLDivElement;
  private project!: Project;
  private callbacks: DrumMachineCallbacks;
  private selectedTrack = 0;
  private paintLevel: Level = "hard";

  // Toolbar elements.
  private levelButtons: HTMLDivElement[] = [];

  // cells[lane][step] — the grid buttons, kept for fast visual updates.
  private cells: HTMLDivElement[][] = [];
  private playCol = -1;

  // Drag-paint state: on pointerdown we decide paint-vs-erase, then apply the
  // same action to any cell the pointer enters until release.
  private dragging: "paint" | "erase" | null = null;
  private onWindowUp = () => {
    this.dragging = null;
  };

  constructor(project: Project, callbacks: DrumMachineCallbacks = {}) {
    injectStyles();
    this.callbacks = callbacks;

    this.root = document.createElement("div");
    this.root.className = "dm-root gui-el";
    this.root.style.display = "none"; // shown only when a drum track is selected
    document.body.appendChild(this.root);

    this.buildToolbar();
    this.buildGrid();
    window.addEventListener("pointerup", this.onWindowUp);

    this.setProject(project);
  }

  // --- Public API (mirrors PianoRoll where it overlaps) ----------------

  setProject(project: Project): void {
    this.project = project;
    this.refreshCells();
  }

  setSelectedTrack(index: number): void {
    this.selectedTrack = index;
    this.refreshCells();
  }

  setPlayhead(beat: number): void {
    const { start, stepBeats } = this.bounds();
    const col =
      stepBeats > 0 ? ((Math.floor((beat - start) / stepBeats) % DRUM_STEPS) + DRUM_STEPS) % DRUM_STEPS : -1;
    if (col === this.playCol) return;
    this.highlightColumn(this.playCol, false);
    this.highlightColumn(col, true);
    this.playCol = col;
  }

  setVisible(visible: boolean): void {
    this.root.style.display = visible ? "flex" : "none";
  }

  dispose(): void {
    window.removeEventListener("pointerup", this.onWindowUp);
    this.root.remove();
  }

  // --- Toolbar ---------------------------------------------------------

  private buildToolbar(): void {
    const bar = document.createElement("div");
    bar.className = "dm-toolbar";

    const mkButton = (label: string, onClick: () => void): HTMLDivElement => {
      const b = document.createElement("div");
      b.className = "dm-btn";
      b.textContent = label;
      b.addEventListener("click", onClick);
      return b;
    };
    const mkGroup = (label: string, ...children: HTMLElement[]): HTMLDivElement => {
      const g = document.createElement("div");
      g.className = "dm-group";
      if (label) {
        const l = document.createElement("span");
        l.className = "dm-label";
        l.textContent = label;
        g.appendChild(l);
      }
      g.append(...children);
      return g;
    };

    // Transport / tempo / loop live in the global transport bar
    // (SequencerLayout); this toolbar carries only drum-specific controls.
    // H/M/S level selector — the velocity painted into newly toggled steps.
    this.levelButtons = LEVELS.map((lvl) => {
      const b = mkButton(LEVEL_LABEL[lvl], () => this.setPaintLevel(lvl));
      b.classList.toggle("dm-btn-active", lvl === this.paintLevel);
      return b;
    });
    const level = mkGroup("VEL", ...this.levelButtons);

    const clear = mkGroup("", mkButton("clear", () => this.clearPattern()));

    bar.append(level, clear);
    this.root.appendChild(bar);
  }

  private setPaintLevel(level: Level): void {
    this.paintLevel = level;
    this.levelButtons.forEach((b, i) => b.classList.toggle("dm-btn-active", LEVELS[i] === level));
  }

  // --- Grid ------------------------------------------------------------

  private buildGrid(): void {
    const grid = document.createElement("div");
    grid.className = "dm-grid";

    // Header: a spacer over the name column, then step numbers (1-based).
    const header = document.createElement("div");
    header.className = "dm-header";
    const spacer = document.createElement("div");
    spacer.className = "dm-spacer";
    const numbers = document.createElement("div");
    numbers.className = "dm-numbers";
    for (let s = 0; s < DRUM_STEPS; s++) {
      const n = document.createElement("div");
      n.className = "dm-num";
      if (s % BEATS_PER_BAR === 0) n.classList.add("dm-beat");
      n.textContent = `${s + 1}`;
      numbers.appendChild(n);
    }
    header.append(spacer, numbers);
    grid.appendChild(header);

    // One row per lane: a clickable name (auditions the drum) + step cells.
    this.cells = DRUM_LANES.map((lane, li) => {
      const row = document.createElement("div");
      row.className = "dm-row";
      const name = document.createElement("div");
      name.className = "dm-name";
      name.textContent = lane.name;
      name.addEventListener("click", () => this.callbacks.onAudition?.(lane.midi));
      const steps = document.createElement("div");
      steps.className = "dm-steps";

      const rowCells: HTMLDivElement[] = [];
      for (let s = 0; s < DRUM_STEPS; s++) {
        const cell = document.createElement("div");
        cell.className = "dm-cell";
        if (s % BEATS_PER_BAR === 0) cell.classList.add("dm-beat");
        cell.addEventListener("pointerdown", (e) => this.cellDown(e, li, s));
        cell.addEventListener("pointerenter", () => this.cellEnter(li, s));
        cell.addEventListener("contextmenu", (e) => {
          e.preventDefault();
          this.setStep(li, s, null);
        });
        steps.appendChild(cell);
        rowCells.push(cell);
      }
      row.append(name, steps);
      grid.appendChild(row);
      return rowCells;
    });

    this.root.appendChild(grid);
  }

  /** Pointer-down on a cell: decide paint-vs-erase from its current state, then
   *  apply. Subsequent cells entered while held repeat the same action. */
  private cellDown(e: PointerEvent, lane: number, step: number): void {
    if (e.button === 2) return; // right-click handled by contextmenu (erase)
    const existing = this.levelAt(lane, step);
    // Toggle semantics: a step already lit at the current paint level erases;
    // anything else paints (or re-levels) at the current level.
    this.dragging = existing === this.paintLevel ? "erase" : "paint";
    this.applyDrag(lane, step);
  }

  private cellEnter(lane: number, step: number): void {
    if (this.dragging) this.applyDrag(lane, step);
  }

  private applyDrag(lane: number, step: number): void {
    this.setStep(lane, step, this.dragging === "erase" ? null : this.paintLevel);
  }

  /** Set a lane/step to a level (or null to clear) by mutating the clip's notes,
   *  then update the cell visual and report the change. */
  private setStep(lane: number, step: number, level: Level | null): void {
    const clip = this.editClip();
    if (!clip) return;
    const laneMidi = DRUM_LANES[lane].midi;
    const { start, stepBeats } = this.bounds();
    const noteStart = step * stepBeats; // clip-relative (drum clips start at 0)

    const idx = clip.notes.findIndex(
      (n) => Math.round(n.midi) === laneMidi && beatToStep(clip.start + n.start, start, stepBeats) === step,
    );

    if (level === null) {
      if (idx >= 0) clip.notes.splice(idx, 1);
    } else if (idx >= 0) {
      clip.notes[idx].velocity = LEVEL_VELOCITY[level]; // re-level in place
    } else {
      clip.notes.push({
        start: noteStart,
        duration: stepBeats * 0.5, // drums are one-shots; duration is cosmetic
        midi: laneMidi,
        velocity: LEVEL_VELOCITY[level],
      });
    }
    this.paintCell(lane, step, level);
    this.callbacks.onNotesChange?.(this.project);
  }

  private clearPattern(): void {
    const clip = this.currentClip();
    if (!clip || clip.notes.length === 0) return;
    clip.notes = [];
    this.refreshCells();
    this.callbacks.onNotesChange?.(this.project);
  }

  // --- Cell <-> note mapping -------------------------------------------

  /** Pattern window: drum steps span the loop `[loopStart, loopEnd]`. */
  private bounds(): { start: number; stepBeats: number } {
    const p = this.project;
    return { start: p.loopStart, stepBeats: stepDuration(p.loopStart, p.loopEnd) };
  }

  private drumTrack() {
    const t = this.project.tracks[this.selectedTrack];
    return t && t.instrument.kind === "drums" ? t : null;
  }

  /** The selected drum track's clip, if any (read-only callers). */
  private currentClip(): Clip | null {
    return this.drumTrack()?.clips[0] ?? null;
  }

  /** Like `currentClip` but creates the clip if the track has none. */
  private editClip(): Clip | null {
    const track = this.drumTrack();
    if (!track) return null;
    if (track.clips.length === 0) {
      track.clips.push({ id: `${track.id}-clip`, start: 0, length: this.project.loopEnd, notes: [] });
    }
    return track.clips[0];
  }

  /** The level currently stored at a lane/step, or null if empty. */
  private levelAt(lane: number, step: number): Level | null {
    const clip = this.currentClip();
    if (!clip) return null;
    const laneMidi = DRUM_LANES[lane].midi;
    const { start, stepBeats } = this.bounds();
    const note = clip.notes.find(
      (n: Note) => Math.round(n.midi) === laneMidi && beatToStep(clip.start + n.start, start, stepBeats) === step,
    );
    return note ? velocityToLevel(note.velocity) : null;
  }

  // --- Visuals ---------------------------------------------------------

  private trackRgb(): [number, number, number] {
    const t = this.project.tracks[this.selectedTrack];
    return hexToRgb(t?.color ?? "#7fd1ff");
  }

  /** Recompute every cell from the current clip (cheap: 160 cells). */
  private refreshCells(): void {
    for (let l = 0; l < this.cells.length; l++) {
      for (let s = 0; s < DRUM_STEPS; s++) this.paintCell(l, s, null);
    }
    const clip = this.currentClip();
    if (!clip) return;
    const { start, stepBeats } = this.bounds();
    for (const n of clip.notes) {
      const lane = DRUM_LANES.findIndex((d) => d.midi === Math.round(n.midi));
      if (lane < 0) continue;
      const step = beatToStep(clip.start + n.start, start, stepBeats);
      if (step === null) continue;
      this.paintCell(lane, step, velocityToLevel(n.velocity));
    }
  }

  /** Set one cell's fill to reflect its level (null = empty). */
  private paintCell(lane: number, step: number, level: Level | null): void {
    const cell = this.cells[lane]?.[step];
    if (!cell) return;
    if (level) {
      const [r, g, b] = this.trackRgb();
      cell.style.background = `rgba(${r}, ${g}, ${b}, ${LEVEL_ALPHA[level]})`;
    } else {
      cell.style.background = "";
    }
  }

  private highlightColumn(col: number, on: boolean): void {
    if (col < 0) return;
    for (const row of this.cells) row[col]?.classList.toggle("dm-playcol", on);
  }
}
