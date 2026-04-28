/// <reference types="@types/audioworklet" />
import "./worklet-polyfills";
import init, { Dsp } from "../wasm-pkg/dsp";

type WorkletInbound =
  | { type: "configure"; windowSize: number; rmsHistoryLen: number }
  | { type: "param"; key: "hopSize" | "smoothingTauSecs" | "dbFloor"; value: number };

class DSPProcessor extends AudioWorkletProcessor {
  private window: Float32Array = new Float32Array(2048);
  private hopCounter = 0;
  private dsp: Dsp | null = null;
  private ready = false;
  private windowSize = 2048;
  private hopSize = 1024;
  private rmsHistoryLen = 512;
  private smoothingTauSecs = 0.0956;
  private dbFloor = -100;
  private pendingConfigure: { windowSize: number; rmsHistoryLen: number } | null = null;

  constructor(options?: AudioWorkletNodeOptions) {
    super();
    const wasmModule = (options?.processorOptions as { wasmModule?: WebAssembly.Module } | undefined)?.wasmModule;
    if (!wasmModule) {
      throw new Error("[dsp-worklet] missing wasmModule in processorOptions");
    }
    this.port.onmessage = (e) => this.onMessage(e.data as WorkletInbound);
    this.boot(wasmModule);
  }

  private async boot(wasmModule: WebAssembly.Module) {
    await init({ module_or_path: wasmModule });
    const cfg = this.pendingConfigure ?? { windowSize: this.windowSize, rmsHistoryLen: this.rmsHistoryLen };
    this.applyConfigure(cfg);
    this.ready = true;
  }

  private onMessage(msg: WorkletInbound) {
    if (msg.type === "configure") {
      if (!this.ready) {
        this.pendingConfigure = { windowSize: msg.windowSize, rmsHistoryLen: msg.rmsHistoryLen };
        return;
      }
      this.applyConfigure(msg);
      return;
    }
    if (msg.type === "param") {
      if (msg.key === "hopSize") {
        this.hopSize = Math.min(msg.value, this.windowSize);
      } else if (msg.key === "smoothingTauSecs") {
        this.smoothingTauSecs = msg.value;
        if (this.ready && this.dsp) this.dsp.set_smoothing_tau(msg.value);
      } else if (msg.key === "dbFloor") {
        this.dbFloor = msg.value;
        if (this.ready && this.dsp) this.dsp.set_db_floor(msg.value);
      }
      return;
    }
    const _exhaustive: never = msg;
    void _exhaustive;
  }

  private applyConfigure(cfg: { windowSize: number; rmsHistoryLen: number }) {
    if (this.dsp) {
      this.dsp.free();
      this.dsp = null;
    }
    this.windowSize = cfg.windowSize;
    this.rmsHistoryLen = cfg.rmsHistoryLen;
    this.window = new Float32Array(this.windowSize);
    this.hopCounter = 0;
    this.hopSize = Math.min(this.hopSize, this.windowSize);
    this.dsp = new Dsp(this.windowSize, sampleRate, this.hopSize, this.rmsHistoryLen);
    this.dsp.set_smoothing_tau(this.smoothingTauSecs);
    this.dsp.set_db_floor(this.dbFloor);
    this.port.postMessage({
      type: "configured",
      waveformLen: this.windowSize,
      spectrumLen: this.windowSize / 2,
      bufferAcfLen: this.windowSize / 2,
      rmsLen: this.rmsHistoryLen,
      rmsAcfLen: this.rmsHistoryLen / 2,
    });
  }

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;
    if (!this.ready || !this.dsp) return true;

    const len = channel.length;

    if (len >= this.windowSize) {
      this.window.set(channel.subarray(len - this.windowSize));
    } else {
      this.window.copyWithin(0, len);
      this.window.set(channel, this.windowSize - len);
    }

    this.hopCounter += len;
    while (this.hopCounter >= this.hopSize) {
      this.dsp.process(this.window);
      const wf = new Float32Array(this.dsp.waveform());
      const sp = new Float32Array(this.dsp.spectrum());
      const rms = new Float32Array(this.dsp.rms_history());
      const ba = new Float32Array(this.dsp.buffer_acf());
      const ra = new Float32Array(this.dsp.rms_acf());
      this.port.postMessage(
        { type: "features", waveform: wf, spectrum: sp, rms, bufferAcf: ba, rmsAcf: ra },
        [wf.buffer, sp.buffer, rms.buffer, ba.buffer, ra.buffer],
      );
      this.hopCounter -= this.hopSize;
    }

    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
