import { App, type AppDeps } from "./App";
import { ParamStore } from "./params/ParamStore";
import { ParamPanel } from "./params/ParamPanel";
import { WorkletBridge } from "./params/WorkletBridge";
import { analysisSchemas } from "./params/schemas";
import { createRenderer } from "./render/Scene";
import dspWorkletUrl from "./audio/dsp-worklet?worker&url";
import dspWasmUrl from "./wasm-pkg/dsp_bg.wasm?url";
import type { AudioSourceBundle } from "./audio/AudioSource";

// Constructor refs are mutable so the HMR accept callback can swap them in.
// Vite's `accept(deps, cb)` does NOT rewire the file's static imports — the
// callback receives the new module exports and we update these refs.
let AppCtor: typeof App = App;
let ParamPanelCtor: typeof ParamPanel = ParamPanel;
let WorkletBridgeCtor: typeof WorkletBridge = WorkletBridge;

let pageDeps: AppDeps | null = null;
let app: App | null = null;
let panel: ParamPanel | null = null;
let bridge: WorkletBridge | null = null;
let initialBootstrap = true;

async function buildPageDeps(
  factory: () => Promise<AudioSourceBundle>,
): Promise<AppDeps> {
  const { context, source, stream } = await factory();

  console.log("[audio] context.sampleRate:", context.sampleRate, "Hz");
  console.log("[audio] context.baseLatency:", context.baseLatency, "s");
  console.log("[audio] source.channelCount:", source.channelCount);
  const tracks = stream?.getAudioTracks() ?? [];
  if (tracks.length === 0) {
    console.log("[audio] no MediaStream tracks (likely internal test source)");
  }
  tracks.forEach((t, i) => {
    console.log(`[audio] track ${i} label:`, t.label);
    try {
      const settings = t.getSettings();
      console.log(`[audio] track ${i} settings:`, settings);
      console.log(
        `[audio] track ${i} sampleRate:`,
        (settings as { sampleRate?: number }).sampleRate ?? "(not reported)",
      );
    } catch (err) {
      console.log(`[audio] track ${i} settings: unavailable`, err);
    }
  });

  const wasmModule = await WebAssembly.compileStreaming(fetch(dspWasmUrl));
  await context.audioWorklet.addModule(dspWorkletUrl);
  const workletNode = new AudioWorkletNode(context, "dsp-processor", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    processorOptions: { wasmModule },
  });
  source.connect(workletNode);

  const canvas = document.getElementById("app") as HTMLCanvasElement;
  const renderer = await createRenderer(canvas);

  const paramStore = new ParamStore();
  for (const s of analysisSchemas) paramStore.register(s);

  const sr = context.sampleRate;
  const fftSize = paramStore.get("dsp.windowSize");
  console.log(
    "[audio] FFT bins map: bin0=DC, bin1=" +
      (sr / fftSize).toFixed(1) +
      "Hz, " +
      `bin${fftSize / 2 - 1}=` +
      (((fftSize / 2 - 1) * sr) / fftSize).toFixed(0) +
      "Hz, " +
      "Nyquist=" +
      (sr / 2).toFixed(0) +
      "Hz",
  );

  return { canvas, renderer, audioContext: context, workletNode, paramStore };
}

function buildAppLayer(deps: AppDeps): void {
  app = new AppCtor(deps);
  panel = new ParamPanelCtor(deps.paramStore);
  bridge = new WorkletBridgeCtor(deps.paramStore, deps.workletNode.port);
  app.start();
  if (initialBootstrap) {
    bridge.bootstrap();
    initialBootstrap = false;
  }
}

function teardownAppLayer(): void {
  app?.dispose();
  panel?.dispose();
  bridge?.dispose();
  app = null;
  panel = null;
  bridge = null;
}

const startMic = document.getElementById("start-mic") as HTMLButtonElement;
const startTab = document.getElementById("start-tab") as HTMLButtonElement;
const startTest = document.getElementById(
  "start-test",
) as HTMLButtonElement | null;
const buttons = document.getElementById("start-buttons") as HTMLDivElement;

let started = false;
const onStart = async (
  factory: () => Promise<AudioSourceBundle>,
): Promise<void> => {
  if (started) return;
  started = true;
  buttons.style.display = "none";
  try {
    pageDeps = await buildPageDeps(factory);
    buildAppLayer(pageDeps);
  } catch (err) {
    started = false;
    pageDeps = null;
    buttons.style.display = "";
    console.error("[app] start failed:", err);
    alert(err instanceof Error ? err.message : String(err));
  }
};

startMic.addEventListener("click", async () => {
  const { createMicSource } = await import("./audio/AudioSource");
  await onStart(createMicSource);
});
startTab.addEventListener("click", async () => {
  const { createTabSource } = await import("./audio/AudioSource");
  await onStart(createTabSource);
});
if (startTest) {
  startTest.addEventListener("click", async () => {
    const { createTestSource } = await import("./audio/AudioSource");
    await onStart(() => createTestSource(440));
  });
}

// Keyboard shortcuts before start: T → test signal (440 Hz), any other key → mic.
window.addEventListener(
  "keydown",
  async (e) => {
    if (started) return;
    if (e.key === "t" || e.key === "T") {
      const { createTestSource } = await import("./audio/AudioSource");
      await onStart(() => createTestSource(440));
    } else {
      const { createMicSource } = await import("./audio/AudioSource");
      await onStart(createMicSource);
    }
  },
  { once: true },
);

if (import.meta.hot) {
  import.meta.hot.accept(
    ["./App", "./params/ParamPanel", "./params/WorkletBridge"],
    ([appMod, panelMod, bridgeMod]) => {
      // Each entry is `Module | undefined`; undefined means the module didn't
      // change in this update batch — keep the existing constructor ref.
      if (appMod && appMod.App) AppCtor = appMod.App;
      if (panelMod && panelMod.ParamPanel) ParamPanelCtor = panelMod.ParamPanel;
      if (bridgeMod && bridgeMod.WorkletBridge)
        WorkletBridgeCtor = bridgeMod.WorkletBridge;
      if (!pageDeps) return;
      teardownAppLayer();
      try {
        buildAppLayer(pageDeps);
      } catch (err) {
        console.error("[hmr] failed to rebuild App layer:", err);
        // app/panel/bridge stay null. Next successful save retries.
      }
    },
  );
}
