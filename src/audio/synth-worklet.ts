/// <reference types="@types/audioworklet" />
// worklet-polyfills MUST be first — the wasm-pack glue uses TextDecoder/Encoder
// at module load and for every Rust↔JS string round-trip (e.g. set_param keys).
// A missing/no-op polyfill silently corrupts those strings. Same invariant as
// dsp-worklet.ts.
import "./worklet-polyfills";
import init, { Sequencer } from "../wasm-pkg/dsp";

type SynthInbound =
  | { type: "noteOn"; track: number; midi: number; velocity: number }
  | { type: "noteOff"; track: number; midi: number }
  | { type: "param"; track: number; key: string; value: number }
  | { type: "engine"; track: number; index: number }
  | { type: "kind"; track: number; kind: number }
  | { type: "wavetable"; track: number; data: Float32Array }
  | { type: "trackCount"; count: number }
  | { type: "tempo"; bpm: number }
  | { type: "transport"; action: "play" | "pause" | "seek"; beat?: number }
  | { type: "schedule"; events: Float32Array }
  | { type: "bends"; data: Float32Array }
  | { type: "loop"; start: number; end: number; enabled: boolean };

// Post the playhead back to the main thread every N render blocks rather than
// every block: at 128 samples / 48 kHz that's ~375 blocks/sec, far more than a
// UI playhead needs. 16 blocks ≈ 43 Hz, matching the analysis update rate.
const PLAYHEAD_POST_EVERY = 16;

class SynthProcessor extends AudioWorkletProcessor {
  private seq: Sequencer | null = null;
  private ready = false;
  // Messages can arrive before wasm finishes booting; replay them in order
  // once `seq` exists rather than dropping note triggers.
  private pending: SynthInbound[] = [];
  private blockCount = 0;

  constructor(options?: AudioWorkletNodeOptions) {
    super();
    const wasmModule = (
      options?.processorOptions as { wasmModule?: WebAssembly.Module } | undefined
    )?.wasmModule;
    if (!wasmModule) {
      throw new Error("[synth-worklet] missing wasmModule in processorOptions");
    }
    this.port.onmessage = (e) => this.onMessage(e.data as SynthInbound);
    this.boot(wasmModule);
  }

  private async boot(wasmModule: WebAssembly.Module) {
    await init({ module_or_path: wasmModule });
    // `sampleRate` is a global in AudioWorkletGlobalScope.
    this.seq = new Sequencer(sampleRate);
    this.ready = true;
    const queued = this.pending;
    this.pending = [];
    for (const msg of queued) this.onMessage(msg);
  }

  private onMessage(msg: SynthInbound) {
    if (!this.ready || !this.seq) {
      this.pending.push(msg);
      return;
    }
    switch (msg.type) {
      case "noteOn":
        this.seq.note_on(msg.track, msg.midi, msg.velocity);
        break;
      case "noteOff":
        this.seq.note_off(msg.track, msg.midi);
        break;
      case "param":
        this.seq.set_param(msg.track, msg.key, msg.value);
        break;
      case "engine":
        this.seq.set_engine(msg.track, msg.index);
        break;
      case "kind":
        this.seq.set_instrument_kind(msg.track, msg.kind);
        break;
      case "wavetable":
        this.seq.set_wavetable(msg.track, msg.data);
        break;
      case "trackCount":
        this.seq.set_track_count(msg.count);
        break;
      case "tempo":
        this.seq.set_tempo(msg.bpm);
        break;
      case "transport":
        if (msg.action === "play") this.seq.play();
        else if (msg.action === "pause") this.seq.pause();
        else if (msg.action === "seek") this.seq.seek(msg.beat ?? 0);
        break;
      case "schedule":
        this.seq.set_schedule(msg.events);
        break;
      case "bends":
        this.seq.set_bends(msg.data);
        break;
      case "loop":
        this.seq.set_loop(msg.start, msg.end, msg.enabled);
        break;
      default: {
        const _exhaustive: never = msg;
        void _exhaustive;
      }
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const out = outputs[0];
    if (!out || out.length === 0) return true;
    const channel = out[0];
    if (!this.ready || !this.seq) {
      channel.fill(0);
      return true;
    }
    this.seq.process(channel);
    // Mono synth → duplicate across any extra output channels (e.g. stereo).
    for (let c = 1; c < out.length; c++) out[c].set(channel);

    if (++this.blockCount >= PLAYHEAD_POST_EVERY) {
      this.blockCount = 0;
      this.port.postMessage({
        type: "playhead",
        beat: this.seq.playhead_beats(),
        playing: this.seq.is_playing(),
      });
    }
    return true;
  }
}

registerProcessor("synth-processor", SynthProcessor);
