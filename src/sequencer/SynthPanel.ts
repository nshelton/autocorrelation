//! Custom synth control panel — self-contained DOM, no UI framework. A view of
//! the *selected track's* instrument: `setInstrument` loads a track's engine +
//! params into the controls; edits mutate that instrument in place (the project
//! model is the source of truth) and report via `onParam` / `onEngine` so the
//! caller can forward them to the right track in the worklet.
//!
//! Controls are organized into labeled groups (OSC, FILTER, F.ENV, AMP, LFO,
//! DRIVE, OUT) that wrap to fit the params pane; discrete params (LFO target /
//! shape, drive mode) render as segmented selectors instead of sliders. Every
//! control reports a plain number through `onParam` — selector indices included
//! — so the worklet/Rust `set_param` path needs no special-casing.
//!
//! Built bespoke rather than with tweakpane so synth mode stays dependency-free
//! and visually consistent with the Canvas2D piano-roll.

import { INSTRUMENT_DEFAULTS, type Instrument } from "./model";

const ACCENT = "#7fd1ff"; // matches the piano-roll lead color

// Selectable oscillator engines. Order = the index sent to Rust set_engine.
const ENGINES = ["Subtractive", "Simplex"];

interface ParamSpec {
  key: string;
  label: string;
  /** "slider" (default) or "select" (a segmented index picker). */
  type?: "slider" | "select";
  // --- slider ---
  min?: number;
  max?: number;
  /** Log feels right for filter cutoff / LFO rate; everything else is linear. */
  scale?: "linear" | "log";
  /** Snap to this increment (integers for octave/semi/detune). Omitted = smooth. */
  step?: number;
  format?: (v: number) => string;
  // --- select ---
  /** Option labels; the stored value is the option index. */
  options?: string[];
}

interface Group {
  title: string;
  specs: ParamSpec[];
}

const secs = (v: number) => (v < 1 ? `${(v * 1000).toFixed(0)} ms` : `${v.toFixed(2)} s`);
const unit = (v: number) => v.toFixed(2);
const signed = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
const intc = (v: number) => `${v > 0 ? "+" : ""}${Math.round(v)}`; // signed integer

const GROUPS: Group[] = [
  {
    title: "OSC",
    specs: [
      { key: "octave", label: "oct", min: -3, max: 3, step: 1, format: intc },
      { key: "semi", label: "semi", min: -12, max: 12, step: 1, format: intc },
      { key: "fine", label: "fine", min: -100, max: 100, step: 1, format: (v) => `${intc(v)}¢` },
      { key: "detune", label: "detune", min: 0, max: 50, step: 1, format: (v) => `${Math.round(v)}¢` },
    ],
  },
  {
    title: "FILTER",
    specs: [
      { key: "cutoff", label: "cutoff", min: 20, max: 16000, scale: "log",
        format: (v) => (v >= 1000 ? `${(v / 1000).toFixed(2)} kHz` : `${v.toFixed(0)} Hz`) },
      { key: "resonance", label: "reso", min: 0, max: 1, format: unit },
    ],
  },
  {
    title: "F.ENV",
    specs: [
      { key: "filterEnvAmount", label: "amount", min: -1, max: 1, format: signed },
      { key: "fAttack", label: "atk", min: 0, max: 2, format: secs },
      { key: "fDecay", label: "dec", min: 0, max: 2, format: secs },
      { key: "fSustain", label: "sus", min: 0, max: 1, format: unit },
      { key: "fRelease", label: "rel", min: 0, max: 4, format: secs },
    ],
  },
  {
    title: "AMP",
    specs: [
      { key: "attack", label: "atk", min: 0, max: 2, format: secs },
      { key: "decay", label: "dec", min: 0, max: 2, format: secs },
      { key: "sustain", label: "sus", min: 0, max: 1, format: unit },
      { key: "release", label: "rel", min: 0, max: 4, format: secs },
    ],
  },
  {
    title: "LFO",
    specs: [
      { key: "lfoTarget", label: "target", type: "select", options: ["pitch", "cutoff", "amp"] },
      { key: "lfoShape", label: "shape", type: "select", options: ["sin", "tri", "sqr", "saw"] },
      { key: "lfoRate", label: "rate", min: 0.05, max: 30, scale: "log",
        format: (v) => `${v.toFixed(2)} Hz` },
      { key: "lfoDepth", label: "depth", min: 0, max: 1, format: unit },
    ],
  },
  {
    title: "DRIVE",
    specs: [
      { key: "driveMode", label: "mode", type: "select", options: ["pre", "post"] },
      { key: "drive", label: "amount", min: 0, max: 1, format: unit },
    ],
  },
  {
    title: "OUT",
    specs: [{ key: "gain", label: "gain", min: 0, max: 1, format: unit }],
  },
];

const clamp01 = (t: number) => Math.max(0, Math.min(1, t));

function normToValue(spec: ParamSpec, t: number): number {
  const min = spec.min ?? 0;
  const max = spec.max ?? 1;
  t = clamp01(t);
  return spec.scale === "log" ? min * Math.pow(max / min, t) : min + t * (max - min);
}

function valueToNorm(spec: ParamSpec, v: number): number {
  const min = spec.min ?? 0;
  const max = spec.max ?? 1;
  return spec.scale === "log"
    ? Math.log(v / min) / Math.log(max / min)
    : (v - min) / (max - min);
}

/** Shared shape for both control kinds so the panel can treat them uniformly. */
interface PanelControl {
  readonly row: HTMLDivElement;
  /** Programmatic set (loading a track) — updates DOM only, no callback. */
  display(value: number): void;
  /** Reset to the param default, reporting the change (used by "reset"). */
  reset(): void;
}

let stylesInjected = false;
function injectStyles(): void {
  if (stylesInjected) return;
  stylesInjected = true;
  const css = `
.sp-root { position:absolute; inset:0;
  background:rgba(12,12,14,0.92);
  color:#cccccc; font:11px/1.4 ui-sans-serif,system-ui;
  pointer-events:auto; user-select:none; display:flex; align-items:stretch; overflow:auto; }
.sp-title { display:flex; flex-direction:column; justify-content:center; gap:7px;
  padding:0 14px; border-right:1px solid rgba(204,204,204,0.12); white-space:nowrap;
  position:sticky; left:0; background:rgba(12,12,14,0.96); z-index:1; }
.sp-titletop { display:flex; justify-content:space-between; align-items:center;
  gap:14px; letter-spacing:0.12em; font-weight:600; }
.sp-titletop > span:first-child { cursor:pointer; }
.sp-reset { font-weight:400; letter-spacing:0; opacity:0.55; cursor:pointer; }
.sp-reset:hover { opacity:1; }
.sp-engines { display:flex; gap:4px; }
.sp-engine { padding:3px 9px; border-radius:3px; cursor:pointer;
  background:rgba(204,204,204,0.1); border:1px solid transparent; }
.sp-engine:hover { background:rgba(204,204,204,0.18); }
.sp-engine-active { background:rgba(127,209,255,0.20); border-color:${ACCENT}; color:#ffffff; }
.sp-groups { display:flex; flex:1; flex-wrap:wrap; align-content:flex-start;
  gap:10px 18px; padding:9px 16px; }
.sp-collapsed .sp-groups { display:none; }
.sp-grp { display:flex; flex-direction:column; gap:5px; }
.sp-grp-h { font-size:9px; letter-spacing:0.16em; opacity:0.45;
  padding-bottom:2px; border-bottom:1px solid rgba(204,204,204,0.1); }
.sp-grp-body { display:flex; flex-direction:column; gap:6px; }
.sp-row { display:flex; flex-direction:column; gap:3px; width:104px; }
.sp-topline { display:flex; justify-content:space-between; gap:8px; }
.sp-label { opacity:0.8; }
.sp-value { font-variant-numeric:tabular-nums; opacity:0.95; }
.sp-track { position:relative; height:6px; border-radius:3px;
  background:rgba(204,204,204,0.12); cursor:ew-resize; touch-action:none; }
.sp-fill { position:absolute; left:0; top:0; bottom:0; border-radius:3px;
  background:${ACCENT}; width:0; }
.sp-seg { display:flex; gap:3px; }
.sp-seg-btn { flex:1 1 0; text-align:center; padding:2px 0; border-radius:3px;
  cursor:pointer; background:rgba(204,204,204,0.1); border:1px solid transparent;
  font-size:10px; }
.sp-seg-btn:hover { background:rgba(204,204,204,0.18); }
.sp-seg-active { background:rgba(127,209,255,0.20); border-color:${ACCENT}; color:#ffffff; }`;
  const el = document.createElement("style");
  el.textContent = css;
  document.head.appendChild(el);
}

/** One slider row: owns its DOM and translates pointer/wheel into values. */
class SliderControl implements PanelControl {
  readonly row: HTMLDivElement;
  private fill: HTMLDivElement;
  private valueEl: HTMLSpanElement;
  private value: number;

  constructor(
    private spec: ParamSpec,
    initial: number,
    private onChange: (value: number) => void,
  ) {
    this.value = initial;

    this.row = document.createElement("div");
    this.row.className = "sp-row";
    const top = document.createElement("div");
    top.className = "sp-topline";
    const label = document.createElement("span");
    label.className = "sp-label";
    label.textContent = spec.label;
    this.valueEl = document.createElement("span");
    this.valueEl.className = "sp-value";
    top.append(label, this.valueEl);

    const track = document.createElement("div");
    track.className = "sp-track";
    this.fill = document.createElement("div");
    this.fill.className = "sp-fill";
    track.appendChild(this.fill);

    this.row.append(top, track);
    this.render();

    let dragging = false;
    const setFromX = (clientX: number) => {
      const rect = track.getBoundingClientRect();
      this.set(normToValue(spec, (clientX - rect.left) / rect.width));
    };
    track.addEventListener("pointerdown", (e) => {
      dragging = true;
      track.setPointerCapture(e.pointerId);
      setFromX(e.clientX);
    });
    track.addEventListener("pointermove", (e) => {
      if (dragging) setFromX(e.clientX);
    });
    track.addEventListener("pointerup", (e) => {
      dragging = false;
      track.releasePointerCapture(e.pointerId);
    });
    track.addEventListener("dblclick", () => this.reset());
    track.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        // Stepped params nudge by one step; smooth params by 2% of the range.
        if (spec.step) {
          this.set(this.value - Math.sign(e.deltaY) * spec.step);
        } else {
          const t = clamp01(valueToNorm(spec, this.value) - Math.sign(e.deltaY) * 0.02);
          this.set(normToValue(spec, t));
        }
      },
      { passive: false },
    );
  }

  /** Clamp + snap a raw value to the spec's range/step. */
  private quantize(value: number): number {
    if (this.spec.step) value = Math.round(value / this.spec.step) * this.spec.step;
    return Math.max(this.spec.min ?? 0, Math.min(this.spec.max ?? 1, value));
  }

  /** User-driven set: updates DOM and reports the change. */
  set(value: number): void {
    this.value = this.quantize(value);
    this.render();
    this.onChange(this.value);
  }

  display(value: number): void {
    this.value = this.quantize(value);
    this.render();
  }

  reset(): void {
    this.set(INSTRUMENT_DEFAULTS[this.spec.key]);
  }

  private render(): void {
    this.fill.style.width = `${clamp01(valueToNorm(this.spec, this.value)) * 100}%`;
    this.valueEl.textContent = (this.spec.format ?? unit)(this.value);
  }
}

/** A segmented index picker for a discrete param (LFO target/shape, drive mode).
 *  The selected option index IS the stored value, mirroring how `engine` works. */
class SelectControl implements PanelControl {
  readonly row: HTMLDivElement;
  private buttons: HTMLDivElement[] = [];
  private index: number;

  constructor(
    private spec: ParamSpec,
    initial: number,
    private onChange: (value: number) => void,
  ) {
    this.index = initial;
    const options = spec.options ?? [];

    this.row = document.createElement("div");
    this.row.className = "sp-row";
    const top = document.createElement("div");
    top.className = "sp-topline";
    const label = document.createElement("span");
    label.className = "sp-label";
    label.textContent = spec.label;
    top.appendChild(label);

    const seg = document.createElement("div");
    seg.className = "sp-seg";
    this.buttons = options.map((name, i) => {
      const b = document.createElement("div");
      b.className = "sp-seg-btn";
      b.textContent = name;
      b.addEventListener("click", () => this.set(i));
      seg.appendChild(b);
      return b;
    });

    this.row.append(top, seg);
    this.render();
  }

  set(index: number): void {
    this.index = index;
    this.render();
    this.onChange(index);
  }

  display(value: number): void {
    this.index = Math.round(value);
    this.render();
  }

  reset(): void {
    this.set(INSTRUMENT_DEFAULTS[this.spec.key] ?? 0);
  }

  private render(): void {
    this.buttons.forEach((b, i) => b.classList.toggle("sp-seg-active", i === this.index));
  }
}

export class SynthPanel {
  private root: HTMLDivElement;
  private controls = new Map<string, PanelControl>();
  private engineButtons: HTMLDivElement[] = [];
  private titleEl: HTMLSpanElement;
  private instrument: Instrument | null = null;

  constructor(
    mount: HTMLElement,
    private onParam: (key: string, value: number) => void,
    private onEngine: (index: number) => void,
  ) {
    injectStyles();

    this.root = document.createElement("div");
    this.root.className = "sp-root gui-el";

    const titleCell = document.createElement("div");
    titleCell.className = "sp-title";

    const top = document.createElement("div");
    top.className = "sp-titletop";
    this.titleEl = document.createElement("span");
    this.titleEl.textContent = "SYNTH";
    this.titleEl.addEventListener("click", () => this.root.classList.toggle("sp-collapsed"));
    const reset = document.createElement("span");
    reset.className = "sp-reset";
    reset.textContent = "reset";
    reset.addEventListener("click", () => this.resetAll());
    top.append(this.titleEl, reset);

    const engines = document.createElement("div");
    engines.className = "sp-engines";
    this.engineButtons = ENGINES.map((name, i) => {
      const b = document.createElement("div");
      b.className = "sp-engine";
      b.textContent = name;
      b.addEventListener("click", () => this.setEngine(i));
      engines.appendChild(b);
      return b;
    });

    titleCell.append(top, engines);

    const groups = document.createElement("div");
    groups.className = "sp-groups";
    for (const group of GROUPS) {
      const grp = document.createElement("div");
      grp.className = "sp-grp";
      const h = document.createElement("div");
      h.className = "sp-grp-h";
      h.textContent = group.title;
      const body = document.createElement("div");
      body.className = "sp-grp-body";
      for (const spec of group.specs) {
        const onChange = (value: number) => {
          if (this.instrument) this.instrument.params[spec.key] = value;
          this.onParam(spec.key, value);
        };
        const initial = INSTRUMENT_DEFAULTS[spec.key];
        const control: PanelControl =
          spec.type === "select"
            ? new SelectControl(spec, initial, onChange)
            : new SliderControl(spec, initial, onChange);
        this.controls.set(spec.key, control);
        body.appendChild(control.row);
      }
      grp.append(h, body);
      groups.appendChild(grp);
    }

    this.root.append(titleCell, groups);
    // Mounts into the layout's params pane (which owns `--synth-panel-h` now).
    mount.appendChild(this.root);
  }

  /** Show a track's instrument: load its values into the controls + engine. */
  setInstrument(instrument: Instrument, label = "SYNTH"): void {
    this.instrument = instrument;
    this.titleEl.textContent = label;
    for (const [key, control] of this.controls) {
      control.display(instrument.params[key] ?? INSTRUMENT_DEFAULTS[key]);
    }
    this.engineButtons.forEach((b, i) =>
      b.classList.toggle("sp-engine-active", i === instrument.engine),
    );
  }

  /** Show/hide the panel within the params pane. Drums have no synth params, so
   *  the host hides it and shows a placeholder; the pane height is owned by the
   *  layout, not this panel. */
  setVisible(visible: boolean): void {
    this.root.style.display = visible ? "" : "none";
  }

  private setEngine(index: number): void {
    if (this.instrument) this.instrument.engine = index;
    this.engineButtons.forEach((b, i) =>
      b.classList.toggle("sp-engine-active", i === index),
    );
    this.onEngine(index);
  }

  private resetAll(): void {
    for (const control of this.controls.values()) control.reset();
  }

  dispose(): void {
    this.root.remove();
  }
}
