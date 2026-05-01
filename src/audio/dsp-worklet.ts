/// <reference types="@types/audioworklet" />
import "./worklet-polyfills";
import init, { Dsp } from "../wasm-pkg/dsp";

type WorkletInbound =
  | { type: "configure"; windowSize: number; rmsHistoryLen: number }
  | { type: "param"; key: string; value: number };

class DSPProcessor extends AudioWorkletProcessor {
  private window: Float32Array = new Float32Array(2048);
  private hopCounter = 0;
  private dsp: Dsp | null = null;
  private ready = false;
  private windowSize = 2048;
  private hopSize = 1024;
  private rmsHistoryLen = 512;
  private smoothingTauSecs = 0.0956;
  private onsetSmoothingTauSecs = 0.05;
  private teaTauSecs = 4.0;
  private teaSigma = 5.0;
  private acfSmoothingSigma = 2.0;
  private dbFloor = -100;
  private phaseLock = 1.0;
  private bufferNames: string[] = [];
  private pendingConfigure: {
    windowSize: number;
    rmsHistoryLen: number;
  } | null = null;

  constructor(options?: AudioWorkletNodeOptions) {
    super();
    const wasmModule = (
      options?.processorOptions as
        | { wasmModule?: WebAssembly.Module }
        | undefined
    )?.wasmModule;
    if (!wasmModule) {
      throw new Error("[dsp-worklet] missing wasmModule in processorOptions");
    }
    this.port.onmessage = (e) => this.onMessage(e.data as WorkletInbound);
    this.boot(wasmModule);
  }

  private async boot(wasmModule: WebAssembly.Module) {
    await init({ module_or_path: wasmModule });
    const cfg = this.pendingConfigure ?? {
      windowSize: this.windowSize,
      rmsHistoryLen: this.rmsHistoryLen,
    };
    this.applyConfigure(cfg);
    this.ready = true;
  }

  private onMessage(msg: WorkletInbound) {
    if (msg.type === "configure") {
      if (!this.ready) {
        this.pendingConfigure = {
          windowSize: msg.windowSize,
          rmsHistoryLen: msg.rmsHistoryLen,
        };
        return;
      }
      this.applyConfigure(msg);
      return;
    }
    if (msg.type === "param") {
      // hopSize is the worklet's own concern (controls when we fire the FFT);
      // not a Dsp param. Cache so applyConfigure preserves it across rebuilds.
      if (msg.key === "hopSize") {
        this.hopSize = Math.min(msg.value, this.windowSize);
        return;
      }
      // Cache so applyConfigure can re-apply across rebuilds.
      if (msg.key === "smoothingTauSecs") this.smoothingTauSecs = msg.value;
      else if (msg.key === "onsetSmoothingTauSecs")
        this.onsetSmoothingTauSecs = msg.value;
      else if (msg.key === "teaTauSecs") this.teaTauSecs = msg.value;
      else if (msg.key === "teaSigma") this.teaSigma = msg.value;
      else if (msg.key === "acfSmoothingSigma")
        this.acfSmoothingSigma = msg.value;
      else if (msg.key === "dbFloor") this.dbFloor = msg.value;
      else if (msg.key === "phaseLock") this.phaseLock = msg.value;
      // Forward any other Dsp-recognized key directly. Unknown keys are
      // silently dropped on the Rust side.
      if (this.ready && this.dsp) this.dsp.set_param(msg.key, msg.value);
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
    this.dsp = new Dsp(
      this.windowSize,
      sampleRate,
      this.hopSize,
      this.rmsHistoryLen,
    );
    this.dsp.set_param("smoothingTauSecs", this.smoothingTauSecs);
    this.dsp.set_param("onsetSmoothingTauSecs", this.onsetSmoothingTauSecs);
    this.dsp.set_param("teaTauSecs", this.teaTauSecs);
    this.dsp.set_param("teaSigma", this.teaSigma);
    this.dsp.set_param("acfSmoothingSigma", this.acfSmoothingSigma);
    this.dsp.set_param("dbFloor", this.dbFloor);
    this.dsp.set_param("phaseLock", this.phaseLock);
    // Cache the buffer name list. Names are static across reconfigurations,
    // so this could fire only at boot, but rebuilding it on each configure
    // is cheap and removes the "did names change?" question.
    this.bufferNames = this.dsp.buffer_names();
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
      const buffers: Record<string, Float32Array> = {};
      const transferList: ArrayBuffer[] = [];
      for (const name of this.bufferNames) {
        const data = this.dsp.get_buffer(name);
        buffers[name] = data;
        transferList.push(data.buffer as ArrayBuffer);
      }
      this.port.postMessage({ type: "features", buffers }, transferList);
      this.hopCounter -= this.hopSize;
    }

    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
