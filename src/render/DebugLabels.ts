import { Group, Object3D } from "three";
import { TextLabel } from "./TextLabel";
import type { FeatureStore } from "../store/FeatureStore";
import type { ParamStore } from "../params/ParamStore";

export interface DebugLabelsOptions {
  store: FeatureStore;
  paramStore: ParamStore;
  audioContext: AudioContext;
}

export class DebugLabels {
  readonly object3d: Object3D = new Group();

  private labels: TextLabel[] = [];
  private beatSummary: TextLabel;
  private configSummary: TextLabel;
  private nextDynamicUpdateMs = 0;

  constructor(private opts: DebugLabelsOptions) {
    this.addStaticLabel("waveform", 0x66ffcc, -0.5, 1);
    this.addStaticLabel("bufferAcf", 0xcc99ff, -0, 1);
    this.addStaticLabel("spectrum", 0xffaa66, 1.5, 1);
    this.addStaticLabel("rms + beatGrid", 0xffffff, -1.74, 0.58);
    this.addStaticLabel("onset", 0xff9966, -1, 0);
    this.addStaticLabel("onsetAcf", 0x6666bb, -1, -1.8);
    this.addStaticLabel("onsetAcfEnhanced", 0xff99cc, -1, -1.1);
    this.addStaticLabel("tea", 0xffff66, -1, -1);

    this.beatSummary = this.createLabel("beat: --", 0x66ccff, 0, -0.7);
    this.configSummary = this.createLabel("cfg: --", 0xffffff, 0, 2);
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
      `beat: bpm=${this.formatNumber(bpm, 1)} period=${this.formatNumber(
        period,
        1,
      )} phase=${this.formatNumber(phase, 1)} score=${this.formatNumber(
        Number.isNaN(beatScore) ? gridScore : beatScore,
        2,
      )}`,
    );

    this.configSummary.setText(
      `cfg: rms=${this.opts.store.get("rms").length} hop=${this.opts.paramStore.get(
        "dsp.hopSize",
      )} sr=${Math.round(this.opts.audioContext.sampleRate)}`,
    );
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
  ): TextLabel {
    return this.createLabel(text, color, x, y);
  }

  private createLabel(
    text: string,
    color: number,
    x: number,
    y: number,
  ): TextLabel {
    const label = new TextLabel({
      text,
      color,
      background: "rgba(0, 0, 0, 0)",
      height: 0.2,
      textureWidth: 1536,
      textureHeight: 128,
    });
    label.object3d.position.set(x, y, 0.1);
    this.object3d.add(label.object3d);
    this.labels.push(label);
    return label;
  }

  private formatNumber(value: number, digits: number): string {
    if (value === undefined || Number.isNaN(value)) return "--";
    return value.toFixed(digits);
  }
}
