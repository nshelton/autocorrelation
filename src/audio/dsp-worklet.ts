/// <reference types="@types/audioworklet" />

const WINDOW_SIZE = 2048;

class DSPProcessor extends AudioWorkletProcessor {
  private window = new Float32Array(WINDOW_SIZE);
  private filled = 0;

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
        const out = new Float32Array(this.window); // copy
        this.port.postMessage({ type: "waveform", buffer: out }, [out.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
