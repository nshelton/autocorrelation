const PROCESSOR_NAME = "synth-processor";

export type SynthWorkletOptions = {
  context: AudioContext;
  workletUrl: string;
  wasmUrl: string;
};

/**
 * Main-thread lifecycle for the synth worklet — the mirror of `DspWorklet`,
 * but a *generator*: the node has no input and connects straight to
 * `context.destination`. Single-use per AudioContext (same wasm-glue caching
 * constraint as DspWorklet — to load fresh wasm, rebuild against a new context).
 */
export class SynthWorklet {
  readonly context: AudioContext;
  private readonly workletUrl: string;
  private readonly wasmUrl: string;
  private node: AudioWorkletNode | null = null;

  constructor(opts: SynthWorkletOptions) {
    this.context = opts.context;
    this.workletUrl = opts.workletUrl;
    this.wasmUrl = opts.wasmUrl;
  }

  async start(): Promise<AudioWorkletNode> {
    // Cache-bust so a context built after `npm run wasm` picks up new bytes.
    const sep = this.wasmUrl.includes("?") ? "&" : "?";
    const fetchUrl = `${this.wasmUrl}${sep}t=${Date.now()}`;
    const wasmModule = await WebAssembly.compileStreaming(fetch(fetchUrl));

    await this.context.audioWorklet.addModule(this.workletUrl);
    const node = new AudioWorkletNode(this.context, PROCESSOR_NAME, {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      processorOptions: { wasmModule },
    });
    node.connect(this.context.destination);
    this.node = node;
    return node;
  }

  current(): AudioWorkletNode {
    if (!this.node) throw new Error("[SynthWorklet] not started");
    return this.node;
  }
}
