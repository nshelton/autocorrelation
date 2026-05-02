const PROCESSOR_NAME = "dsp-processor";

export type DspWorkletOptions = {
  context: AudioContext;
  source: AudioNode;
  workletUrl: string;
  wasmUrl: string;
};

/**
 * One-shot lifecycle: registers the processor (`addModule`), compiles the
 * wasm bytes, builds the `AudioWorkletNode`, and connects it to the source.
 * Single-use per AudioContext — to load fresh wasm, close the AudioContext
 * and construct a new `DspWorklet` against a fresh one. The wasm-pack glue
 * inside the worker caches its compiled `wasm` instance globally per scope,
 * so in-place swaps don't pick up new bytes.
 */
export class DspWorklet {
  readonly context: AudioContext;
  private readonly source: AudioNode;
  private readonly workletUrl: string;
  private readonly wasmUrl: string;
  private node: AudioWorkletNode | null = null;

  constructor(opts: DspWorkletOptions) {
    this.context = opts.context;
    this.source = opts.source;
    this.workletUrl = opts.workletUrl;
    this.wasmUrl = opts.wasmUrl;
  }

  async start(): Promise<AudioWorkletNode> {
    // Cache-bust so a fresh AudioContext built after `npm run wasm` picks up
    // the new bytes even when the URL string is unchanged. Vite serves the
    // file from disk on each request.
    const sep = this.wasmUrl.includes("?") ? "&" : "?";
    const fetchUrl = `${this.wasmUrl}${sep}t=${Date.now()}`;
    const wasmModule = await WebAssembly.compileStreaming(fetch(fetchUrl));

    await this.context.audioWorklet.addModule(this.workletUrl);
    const node = new AudioWorkletNode(this.context, PROCESSOR_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { wasmModule },
    });
    this.source.connect(node);
    this.node = node;
    return node;
  }

  current(): AudioWorkletNode {
    if (!this.node) throw new Error("[DspWorklet] not started");
    return this.node;
  }
}
