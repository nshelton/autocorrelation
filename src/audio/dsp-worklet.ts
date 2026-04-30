/// <reference types="@types/audioworklet" />
import "./worklet-polyfills";
import init, { Dsp } from "../wasm-pkg/dsp";

type WorkletInbound =
  | { type: "configure"; windowSize: number; rmsHistoryLen: number }
  | { type: "param"; key: "hopSize" | "smoothingTauSecs" | "dbFloor" | "accumTauSecs"; value: number }
  // HMR: a freshly-built App needs the buffer sizes to construct line renderers,
  // but the worklet only emits `configured` on boot or on a `configure` request
  // (the latter resets Dsp state, which we want to preserve across HMR). `sync`
  // re-emits the cached `configured` payload without touching Dsp.
  | { type: "sync" };

type ConfiguredOutbound = {
  type: "configured";
  waveformLen: number;
  spectrumLen: number;
  bufferAcfLen: number;
  rmsLen: number;
  rmsAcfLen: number;
  acfPeaksLen: number;
  beatGridLen: number;
  beatPulsesLen: number;
  beatStateLen: number;
};

class DSPProcessor extends AudioWorkletProcessor {
  private window: Float32Array = new Float32Array(2048);
  private hopCounter = 0;
  private dsp: Dsp | null = null;
  private ready = false;
  private windowSize = 2048;
  private hopSize = 1024;
  private rmsHistoryLen = 512;
  private smoothingTauSecs = 0.0956;
  private accumTauSecs = 4.0;
  private dbFloor = -100;
  private pendingConfigure: { windowSize: number; rmsHistoryLen: number } | null = null;
  private lastConfigured: ConfiguredOutbound | null = null;

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
    if (msg.type === "sync") {
      // If we haven't configured yet, boot() will post `configured` once init
      // finishes — no need to retain the sync request.
      if (this.lastConfigured) this.port.postMessage(this.lastConfigured);
      return;
    }
    if (msg.type === "param") {
      if (msg.key === "hopSize") {
        this.hopSize = Math.min(msg.value, this.windowSize);
      } else if (msg.key === "smoothingTauSecs") {
        this.smoothingTauSecs = msg.value;
        if (this.ready && this.dsp) this.dsp.set_smoothing_tau(msg.value);
      } else if (msg.key === "accumTauSecs") {
        this.accumTauSecs = msg.value;
        if (this.ready && this.dsp) this.dsp.set_accum_tau_secs(msg.value);
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
    this.dsp.set_accum_tau_secs(this.accumTauSecs);
    const payload: ConfiguredOutbound = {
      type: "configured",
      waveformLen: this.windowSize,
      spectrumLen: this.windowSize / 2,
      bufferAcfLen: this.windowSize / 2,
      rmsLen: this.rmsHistoryLen,
      rmsAcfLen: this.rmsHistoryLen / 2,
      acfPeaksLen: 30, // = 3 * MAX_PEAKS in crates/dsp/src/lib.rs (interleaved [lag, mag, sharpness])
      beatGridLen: 3, // = BEAT_GRID_LEN in crates/dsp/src/lib.rs ([period, phase, score])
      beatPulsesLen: 4, // = BEAT_PULSES_LEN in crates/dsp/src/lib.rs (saw values for cycles 1, 4, 8, 16)
      beatStateLen: 4, // = BEAT_STATE_LEN in crates/dsp/src/lib.rs ([bpm, bpm_conf, beats_per_measure, measure_conf])
    };
    this.lastConfigured = payload;
    this.port.postMessage(payload);
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
      const raAccum = new Float32Array(this.dsp.rms_acf_accum());
      const peaks = new Float32Array(this.dsp.acf_peaks());
      const beatGrid = new Float32Array(this.dsp.beat_grid());
      const beatPulses = new Float32Array(this.dsp.beat_pulses());
      const beatState = new Float32Array(this.dsp.beat_state());
      const rmsLow = new Float32Array(this.dsp.low_rms_history());
      const rmsMid = new Float32Array(this.dsp.mid_rms_history());
      const rmsHigh = new Float32Array(this.dsp.high_rms_history());
      const rmsAcfLow = new Float32Array(this.dsp.low_rms_acf());
      this.port.postMessage(
        {
          type: "features",
          waveform: wf,
          spectrum: sp,
          rms,
          bufferAcf: ba,
          rmsAcf: ra,
          rmsAcfAccum: raAccum,
          acfPeaks: peaks,
          beatGrid,
          beatPulses,
          beatState,
          rmsLow,
          rmsMid,
          rmsHigh,
          rmsAcfLow,
        },
        [
          wf.buffer,
          sp.buffer,
          rms.buffer,
          ba.buffer,
          ra.buffer,
          raAccum.buffer,
          peaks.buffer,
          beatGrid.buffer,
          beatPulses.buffer,
          beatState.buffer,
          rmsLow.buffer,
          rmsMid.buffer,
          rmsHigh.buffer,
          rmsAcfLow.buffer,
        ],
      );
      this.hopCounter -= this.hopSize;
    }

    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
