import { Group, Object3D } from "three";
import {
  TextLabel,
  type TextLabelAnchorX,
  TextLabelAnchorY,
} from "./TextLabel";
import type { FeatureStore } from "../../store/FeatureStore";
import type { ParamStore } from "../../params/ParamStore";

export interface DebugLabelsOptions {
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
}

// Mirror of crates/dsp/src/perf.rs PERF_METRIC_NAMES indexes.
const PERF_TOTAL_MS = 0;
const PERF_FREQ_HZ = 1;

export class DebugLabels {
  readonly object3d: Object3D = new Group();

  private labels: TextLabel[] = [];
  private beatSummary: TextLabel;
  private configSummary: TextLabel;
  private nextDynamicUpdateMs = 0;

  constructor(private opts: DebugLabelsOptions) {
    this.addStaticLabel("waveform", 0x66ffcc, -1, 2);
    this.addStaticLabel("bufferAcf", 0xcc99ff, 0, 2);
    this.addStaticLabel("spectrum", 0xffaa66, 2, 2);
    this.addStaticLabel("rms", 0x888888, -2, 1, "left");
    this.addStaticLabel("onset", 0x888888, -2, 0.5, "left");
    this.addStaticLabel("onsetAcf", 0x00ffff, -2, -0, "left");

    this.beatSummary = this.createLabel("beat: --", 0x66ccff, 2, 0);
    this.configSummary = this.createLabel("cfg: --", 0x888888, -0, 2.1);
  }

  update(): void {
    const now = performance.now();
    if (now < this.nextDynamicUpdateMs) return;
    this.nextDynamicUpdateMs = now + 250;

    const beatState = this.opts.store.get("beatState");
    const beatGrid = this.opts.store.get("beatGrid");
    const bpm = beatState[0];
    const beatScore = beatState[1];
    const period = beatGrid[0];
    const phase = beatGrid[1];
    const gridScore = beatGrid[2];

    this.beatSummary.setText(
      `beat: bpm=${this.formatNumber(bpm, 1, 5)} period=${this.formatNumber(
        period,
        1,
        4,
      )} phase=${this.formatNumber(phase, 1, 4)} score=${this.formatNumber(
        Number.isNaN(beatScore) ? gridScore : beatScore,
        2,
        4,
      )}`,
    );

    this.configSummary.setText(
      `rms=${this.opts.store.get("rms").length} hop=${this.opts.paramStore.get(
        "dsp.hopSize",
      )} sr=${Math.round(this.opts.audioContext.sampleRate)}` +
        this.formatPerfLabel(),
    );
  }

  private formatPerfLabel(): string {
    const perf = this.opts.store.get("dspPerf");
    if (perf.length < 2) return "dsp: --";
    const totalMs = perf[PERF_TOTAL_MS];
    const freqHz = perf[PERF_FREQ_HZ];
    const totalStr = Number.isFinite(totalMs)
      ? `${totalMs.toFixed(2).padStart(5)}ms`
      : "   --ms";
    const freqStr = Number.isFinite(freqHz)
      ? `${freqHz.toFixed(1).padStart(5)}Hz`
      : "   --Hz";
    return ` ${totalStr} ${freqStr}`;
  }

  dispose(): void {
    for (const label of this.labels) label.dispose();
    this.labels = [];
    this.object3d.parent?.remove(this.object3d);
  }

  private addStaticLabel(
    text: string,
    color: number,
    x: number,
    y: number,
    anchorX: TextLabelAnchorX = "right",
    anchorY: TextLabelAnchorY = "top",
  ): TextLabel {
    return this.createLabel(text, color, x, y, anchorX, anchorY);
  }

  private createLabel(
    text: string,
    color: number,
    x: number,
    y: number,
    anchorX: TextLabelAnchorX = "right",
    anchorY: TextLabelAnchorY = "top",
  ): TextLabel {
    const label = new TextLabel({
      text,
      color,
      background: "rgba(0,0,0,0)",
      height: 0.15,
      textureHeight: 128,
      anchorX,
      anchorY,
    });
    label.object3d.position.set(x, y, 0);
    this.object3d.add(label.object3d);
    this.labels.push(label);
    return label;
  }

  private formatNumber(value: number, digits: number, width: number): string {
    const s =
      value === undefined || Number.isNaN(value) ? "--" : value.toFixed(digits);
    return s.padStart(width);
  }
}
