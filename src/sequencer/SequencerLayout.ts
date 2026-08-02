//! The synth-mode DAW shell — a resizable vertical 3-pane layout plus the global
//! transport bar. Self-contained DOM (no UI framework), consistent with the rest
//! of `src/sequencer/`.
//!
//! Layout (top→bottom): transport bar (fixed height `Ht`) · arrangement pane
//! (height `A`, resizable) · splitter · **editor band** (the gap) · splitter ·
//! params pane (height `P`, resizable).
//!
//! Rather than re-parent the editors, this owns the existing CSS-var contract:
//! it is the **sole publisher** of `--track-bar-h = Ht+A+S` and
//! `--synth-panel-h = P+S`, which are exactly the editor band's top/bottom insets.
//! `PianoRoll`/`DrumMachine` are `position:fixed` and already fill
//! `[--track-bar-h, 100vh − --synth-panel-h]` via those vars, so a splitter drag
//! just re-publishes the vars and they reflow — **no editor edits required**
//! (PianoRoll is owned by another agent). The vars are set from explicit drag /
//! window-resize math only, never from a ResizeObserver on our own panes, so
//! there's no resize-feedback loop.
//!
//! Pane heights persist under their own localStorage key (separate from the
//! project so the project schema/version logic is untouched).

import type { Project } from "./model";
import { PITCH_CLASS_NAMES } from "./tuning";

export interface TransportCallbacks {
  onPlayPause(): void;
  onStop(): void;
  onTempo(bpm: number): void;
  /** Loop length in beats (the 4/8/16/32 buttons — also the view window). */
  onLoopLength(beats: number): void;
  onLoopToggle(): void;
  onAddTrack(kind: "synth" | "drums"): void;
  /** Toggle just intonation on/off. */
  onToggleTuning(): void;
  /** Set the just-intonation root (tonic) pitch class, 0..11. */
  onSetRoot(pc: number): void;
}

const UI_KEY = "autocorrelation.ui";
const HT = 34; // transport bar height (px) — fixed via CSS, mirrored here
const S = 6; // splitter thickness (px)
const EDITOR_MIN = 120; // editor band never collapses below this
const A_MIN = 60; // arrangement pane minimum
const P_MIN = 0; // params pane can fully collapse (drum tracks have no params)
const LOOP_OPTIONS = [4, 8, 16, 32];

interface UiPrefs {
  arrangementH: number;
  paramsH: number;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function loadPrefs(): Partial<UiPrefs> {
  try {
    const raw = localStorage.getItem(UI_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw) as Partial<UiPrefs>;
    return {
      arrangementH: typeof p.arrangementH === "number" ? p.arrangementH : undefined,
      paramsH: typeof p.paramsH === "number" ? p.paramsH : undefined,
    };
  } catch {
    return {}; // absent / disabled / corrupt — fall back to defaults
  }
}

let stylesInjected = false;
function injectStyles(): void {
  if (stylesInjected) return;
  stylesInjected = true;
  const css = `
.sl-transport { position:fixed; top:0; left:0; right:0; height:${HT}px; z-index:7;
  box-sizing:border-box; display:flex; gap:16px; align-items:center; padding:0 12px;
  background:rgba(12,12,14,0.95); border-bottom:1px solid rgba(204,204,204,0.15);
  color:#cccccc; font:11px/1 ui-sans-serif,system-ui; pointer-events:auto; user-select:none; }
.sl-pane { position:fixed; left:0; right:0; z-index:7; box-sizing:border-box;
  background:rgba(12,12,14,0.92); overflow:hidden; }
.sl-arrangement { border-bottom:1px solid rgba(204,204,204,0.12); }
.sl-params { border-top:1px solid rgba(204,204,204,0.15); }
.sl-split { position:fixed; left:0; right:0; height:${S}px; z-index:8; cursor:row-resize;
  background:rgba(204,204,204,0.06); touch-action:none; }
.sl-split:hover, .sl-split.sl-dragging { background:rgba(127,209,255,0.35); }
.sl-placeholder { position:fixed; left:0; right:0; z-index:4; display:flex;
  align-items:center; justify-content:center; color:rgba(204,204,204,0.4);
  font:12px/1 ui-sans-serif,system-ui; pointer-events:none; user-select:none; }
.sl-params-ph { position:absolute; inset:0; z-index:1; display:flex; align-items:center;
  justify-content:center; color:rgba(204,204,204,0.4); font:12px/1 ui-sans-serif,system-ui;
  pointer-events:none; }
.sl-group { display:flex; align-items:center; gap:4px; }
.sl-label { opacity:0.55; letter-spacing:0.1em; margin-right:2px; }
.sl-spacer { flex:1 1 auto; }
.sl-btn { min-width:18px; text-align:center; padding:4px 8px; border-radius:3px; cursor:pointer;
  background:rgba(204,204,204,0.1); border:1px solid transparent; }
.sl-btn:hover { background:rgba(204,204,204,0.18); }
.sl-btn-active { background:rgba(127,209,255,0.20); border-color:#7fd1ff; color:#ffffff; }
.sl-readout { min-width:30px; text-align:center; font-variant-numeric:tabular-nums; opacity:0.9; }
.sl-bpm { cursor:ns-resize; min-width:30px; touch-action:none; font-variant-numeric:tabular-nums; }
.sl-loop-on { background:rgba(255,204,85,0.22); border-color:#ffcc55; color:#ffd87a; }
.sl-add { padding:4px 9px; border-radius:4px; cursor:pointer; opacity:0.75;
  background:rgba(204,204,204,0.06); border:1px dashed rgba(204,204,204,0.3); white-space:nowrap; }
.sl-add:hover { opacity:1; background:rgba(204,204,204,0.14); }`;
  const el = document.createElement("style");
  el.textContent = css;
  document.head.appendChild(el);
}

export class SequencerLayout {
  readonly arrangementContainer: HTMLDivElement;
  readonly paramsContainer: HTMLDivElement;

  private cb: TransportCallbacks;
  private transport: HTMLDivElement;
  private split1: HTMLDivElement; // arrangement | editor
  private split2: HTMLDivElement; // editor | params
  private editorPlaceholder: HTMLDivElement;
  private paramsPlaceholder: HTMLDivElement;

  // Transport readouts.
  private playButton!: HTMLDivElement;
  private posReadout!: HTMLSpanElement;
  private bpmReadout!: HTMLDivElement;
  private loopButtons: HTMLDivElement[] = [];
  private loopToggleButton!: HTMLDivElement;
  private jiButton!: HTMLDivElement;
  private rootButton!: HTMLDivElement;
  private tuningRoot = 0;
  private project: Project | null = null;

  // Pane heights (px).
  private arrangementH: number;
  private paramsH: number;

  private onResize = () => this.reclampAndPublish();

  constructor(cb: TransportCallbacks) {
    injectStyles();
    this.cb = cb;

    const prefs = loadPrefs();
    const avail = window.innerHeight - HT - 2 * S;
    this.arrangementH = prefs.arrangementH ?? Math.round(0.3 * avail);
    this.paramsH = prefs.paramsH ?? 110;

    this.transport = this.buildTransport();

    this.arrangementContainer = document.createElement("div");
    this.arrangementContainer.className = "sl-pane sl-arrangement gui-el";
    this.paramsContainer = document.createElement("div");
    this.paramsContainer.className = "sl-pane sl-params gui-el";

    this.split1 = this.buildSplitter("split1");
    this.split2 = this.buildSplitter("split2");

    this.editorPlaceholder = document.createElement("div");
    this.editorPlaceholder.className = "sl-placeholder gui-el";
    this.editorPlaceholder.textContent = "Select a clip in the arrangement to edit";
    this.editorPlaceholder.style.display = "none";

    this.paramsPlaceholder = document.createElement("div");
    this.paramsPlaceholder.className = "sl-params-ph";
    this.paramsPlaceholder.textContent = "No parameters for this track";
    this.paramsPlaceholder.style.display = "none";
    this.paramsContainer.appendChild(this.paramsPlaceholder);

    document.body.append(
      this.transport,
      this.arrangementContainer,
      this.paramsContainer,
      this.split1,
      this.split2,
      this.editorPlaceholder,
    );

    window.addEventListener("resize", this.onResize);
    this.reclampAndPublish();
  }

  // --- Public API ------------------------------------------------------

  setProject(p: Project): void {
    this.project = p;
    this.bpmReadout.textContent = `${Math.round(p.bpm)}`;
    this.updateLoopButtons();
    this.loopToggleButton.classList.toggle("sl-loop-on", p.loopEnabled);
    const just = p.tuning?.mode === "just";
    this.tuningRoot = p.tuning?.root ?? 0;
    this.jiButton.classList.toggle("sl-btn-active", just);
    this.rootButton.textContent = PITCH_CLASS_NAMES[this.tuningRoot];
    this.rootButton.style.opacity = just ? "1" : "0.4"; // root only matters when on
  }

  setPlaying(playing: boolean): void {
    this.playButton.textContent = playing ? "⏸" : "▶";
  }

  setPlayhead(beat: number): void {
    const b = Math.max(0, beat);
    this.posReadout.textContent = `${Math.floor(b / 4) + 1}.${Math.floor(b % 4) + 1}`;
  }

  /** Show the "select a clip" overlay across the editor band. */
  setEditorPlaceholder(show: boolean): void {
    this.editorPlaceholder.style.display = show ? "flex" : "none";
  }

  /** Show a placeholder in the params pane (drum track / no selection). */
  setParamsPlaceholder(show: boolean, text = "No parameters for this track"): void {
    this.paramsPlaceholder.textContent = text;
    this.paramsPlaceholder.style.display = show ? "flex" : "none";
  }

  dispose(): void {
    window.removeEventListener("resize", this.onResize);
    document.documentElement.style.removeProperty("--track-bar-h");
    document.documentElement.style.removeProperty("--synth-panel-h");
    for (const el of [
      this.transport,
      this.arrangementContainer,
      this.paramsContainer,
      this.split1,
      this.split2,
      this.editorPlaceholder,
    ]) {
      el.remove();
    }
  }

  // --- Transport bar ---------------------------------------------------

  private buildTransport(): HTMLDivElement {
    const bar = document.createElement("div");
    bar.className = "sl-transport gui-el";

    const mkButton = (label: string, onClick: () => void): HTMLDivElement => {
      const b = document.createElement("div");
      b.className = "sl-btn";
      b.textContent = label;
      b.addEventListener("click", onClick);
      return b;
    };
    const mkGroup = (label: string, ...kids: HTMLElement[]): HTMLDivElement => {
      const g = document.createElement("div");
      g.className = "sl-group";
      if (label) {
        const l = document.createElement("span");
        l.className = "sl-label";
        l.textContent = label;
        g.appendChild(l);
      }
      g.append(...kids);
      return g;
    };

    this.playButton = mkButton("▶", () => this.cb.onPlayPause());
    this.posReadout = document.createElement("span");
    this.posReadout.className = "sl-readout";
    this.posReadout.textContent = "1.1";
    this.loopToggleButton = mkButton("⟳", () => this.cb.onLoopToggle());
    const transport = mkGroup(
      "",
      this.playButton,
      mkButton("■", () => this.cb.onStop()),
      this.loopToggleButton,
      this.posReadout,
    );

    // BPM: drag vertically (Shift = fine) or scroll. Same gesture as the editors.
    this.bpmReadout = document.createElement("div");
    this.bpmReadout.className = "sl-btn sl-bpm";
    this.bpmReadout.textContent = "120";
    let dragging = false;
    let startY = 0;
    let startBpm = 120;
    this.bpmReadout.addEventListener("pointerdown", (e) => {
      dragging = true;
      startY = e.clientY;
      startBpm = this.project?.bpm ?? 120;
      this.bpmReadout.setPointerCapture(e.pointerId);
    });
    const applyBpm = (v: number) => {
      const bpm = clamp(Math.round(v), 20, 300);
      this.bpmReadout.textContent = `${bpm}`;
      this.cb.onTempo(bpm);
    };
    this.bpmReadout.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      applyBpm(startBpm + (startY - e.clientY) * (e.shiftKey ? 0.1 : 0.5));
    });
    this.bpmReadout.addEventListener("pointerup", (e) => {
      dragging = false;
      this.bpmReadout.releasePointerCapture(e.pointerId);
    });
    this.bpmReadout.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        applyBpm((this.project?.bpm ?? 120) - Math.sign(e.deltaY));
      },
      { passive: false },
    );
    const bpm = mkGroup("BPM", this.bpmReadout);

    this.loopButtons = LOOP_OPTIONS.map((beats) =>
      mkButton(String(beats), () => this.cb.onLoopLength(beats)),
    );
    const loop = mkGroup("LOOP", ...this.loopButtons);

    // Tuning: JI on/off + the just-intonation root (click cycles, wheel ±).
    this.jiButton = mkButton("JI", () => this.cb.onToggleTuning());
    this.rootButton = mkButton("C", () => this.cb.onSetRoot((this.tuningRoot + 1) % 12));
    this.rootButton.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.cb.onSetRoot(((this.tuningRoot - Math.sign(e.deltaY)) % 12 + 12) % 12);
      },
      { passive: false },
    );
    const tune = mkGroup("TUNE", this.jiButton, this.rootButton);

    const spacer = document.createElement("div");
    spacer.className = "sl-spacer";

    const mkAdd = (label: string, kind: "synth" | "drums"): HTMLDivElement => {
      const b = document.createElement("div");
      b.className = "sl-add";
      b.textContent = label;
      b.addEventListener("click", () => this.cb.onAddTrack(kind));
      return b;
    };
    const add = mkGroup("", mkAdd("+ Synth", "synth"), mkAdd("+ Drums", "drums"));

    bar.append(transport, bpm, loop, tune, spacer, add);
    return bar;
  }

  private updateLoopButtons(): void {
    if (!this.project) return;
    const len = this.project.loopEnd - this.project.loopStart;
    this.loopButtons.forEach((b, i) => b.classList.toggle("sl-btn-active", LOOP_OPTIONS[i] === len));
  }

  // --- Splitters / geometry --------------------------------------------

  private buildSplitter(which: "split1" | "split2"): HTMLDivElement {
    const el = document.createElement("div");
    el.className = "sl-split gui-el";
    let startY = 0;
    let startA = 0;
    let startP = 0;
    let active = false;
    el.addEventListener("pointerdown", (e) => {
      active = true;
      startY = e.clientY;
      startA = this.arrangementH;
      startP = this.paramsH;
      el.classList.add("sl-dragging");
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", (e) => {
      if (!active) return;
      const dy = e.clientY - startY;
      if (which === "split1") {
        // Drag down grows the arrangement; keep the editor ≥ EDITOR_MIN.
        const aMax = window.innerHeight - HT - 2 * S - this.paramsH - EDITOR_MIN;
        this.arrangementH = clamp(startA + dy, A_MIN, Math.max(A_MIN, aMax));
      } else {
        // Drag down shrinks the params pane.
        const pMax = window.innerHeight - HT - 2 * S - this.arrangementH - EDITOR_MIN;
        this.paramsH = clamp(startP - dy, P_MIN, Math.max(P_MIN, pMax));
      }
      this.publish();
    });
    const end = (e: PointerEvent) => {
      if (!active) return;
      active = false;
      el.classList.remove("sl-dragging");
      el.releasePointerCapture(e.pointerId);
      this.savePrefs();
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    return el;
  }

  /** Re-clamp the panes to the current viewport (e.g. after a window resize) so
   *  the editor band keeps its minimum, then publish. */
  private reclampAndPublish(): void {
    const avail = window.innerHeight - HT - 2 * S;
    this.arrangementH = clamp(this.arrangementH, A_MIN, Math.max(A_MIN, avail - P_MIN - EDITOR_MIN));
    this.paramsH = clamp(this.paramsH, P_MIN, Math.max(P_MIN, avail - this.arrangementH - EDITOR_MIN));
    this.publish();
  }

  /** Position the panes/splitters and publish the editor-band CSS vars. */
  private publish(): void {
    const topVar = HT + this.arrangementH + S; // --track-bar-h
    const botVar = this.paramsH + S; // --synth-panel-h

    this.arrangementContainer.style.top = `${HT}px`;
    this.arrangementContainer.style.height = `${this.arrangementH}px`;
    this.split1.style.top = `${HT + this.arrangementH}px`;
    this.split2.style.bottom = `${this.paramsH}px`;
    this.paramsContainer.style.bottom = "0px";
    this.paramsContainer.style.height = `${this.paramsH}px`;

    // The editor placeholder fills the band between the two vars.
    this.editorPlaceholder.style.top = `${topVar}px`;
    this.editorPlaceholder.style.bottom = `${botVar}px`;

    document.documentElement.style.setProperty("--track-bar-h", `${topVar}px`);
    document.documentElement.style.setProperty("--synth-panel-h", `${botVar}px`);
  }

  private savePrefs(): void {
    try {
      const data: UiPrefs = { arrangementH: this.arrangementH, paramsH: this.paramsH };
      localStorage.setItem(UI_KEY, JSON.stringify(data));
    } catch {
      /* quota / disabled — non-fatal */
    }
  }
}
