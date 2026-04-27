/// <reference types="@types/audioworklet" />
import "./worklet-polyfills";
import init, { Dsp } from "../wasm-pkg/dsp";

const WINDOW_SIZE = 2048;

class DSPProcessor extends AudioWorkletProcessor {
  private window = new Float32Array(WINDOW_SIZE);
  private filled = 0;
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

    let i = 0;
    while (i < channel.length) {
      const space = WINDOW_SIZE - this.filled;
      const take = Math.min(space, channel.length - i);
      this.window.set(channel.subarray(i, i + take), this.filled);
      this.filled += take;
      i += take;

      if (this.filled === WINDOW_SIZE) {
        if (this.ready && this.dsp) {
          const processed = this.dsp.process(this.window);
          const out = new Float32Array(processed);
          this.port.postMessage({ type: "waveform", buffer: out }, [out.buffer]);
        }
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
