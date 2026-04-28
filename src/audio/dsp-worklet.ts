/// <reference types="@types/audioworklet" />
import "./worklet-polyfills";
import init, { Dsp } from "../wasm-pkg/dsp";

const WINDOW_SIZE = 2048;
const HOP_SIZE = 1024;

class DSPProcessor extends AudioWorkletProcessor {
  private window = new Float32Array(WINDOW_SIZE);
  private hopCounter = 0;
  private dsp: Dsp | null = null;
  private ready = false;

  constructor(options?: AudioWorkletNodeOptions) {
    super();
    const wasmModule = (options?.processorOptions as { wasmModule?: WebAssembly.Module } | undefined)?.wasmModule;
    if (!wasmModule) {
      throw new Error("[dsp-worklet] missing wasmModule in processorOptions");
    }
    this.boot(wasmModule);
  }

  private async boot(wasmModule: WebAssembly.Module) {
    await init(wasmModule);
    this.dsp = new Dsp(WINDOW_SIZE);
    this.ready = true;
  }

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;
    if (!this.ready || !this.dsp) return true;

    const len = channel.length;

    // Slide the window: shift left by `len`, append new samples at the end.
    // Buffer always holds the most recent WINDOW_SIZE samples; zero-padded at boot.
    if (len >= WINDOW_SIZE) {
      // Edge case: chunk is at least as large as the window — just take the tail.
      this.window.set(channel.subarray(len - WINDOW_SIZE));
    } else {
      this.window.copyWithin(0, len);
      this.window.set(channel, WINDOW_SIZE - len);
    }

    // Fire FFT every HOP_SIZE new samples for ~47 Hz update rate (50% overlap).
    this.hopCounter += len;
    while (this.hopCounter >= HOP_SIZE) {
      this.dsp.process(this.window);
      const wf = new Float32Array(this.dsp.waveform());
      const sp = new Float32Array(this.dsp.spectrum());
      const rms = new Float32Array(this.dsp.rms_history());
      this.port.postMessage(
        { type: "features", waveform: wf, spectrum: sp, rms },
        [wf.buffer, sp.buffer, rms.buffer],
      );
      this.hopCounter -= HOP_SIZE;
    }

    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
