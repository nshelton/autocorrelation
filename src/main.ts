import { App, type AppDeps } from "./App";
import { ParamStore } from "./params/ParamStore";
import { ParamPanel } from "./params/ParamPanel";
import { WorkletBridge } from "./params/WorkletBridge";
import { analysisSchemas } from "./params/schemas";
import { createRenderer } from "./render/Scene";
import { DspWorklet } from "./audio/DspWorklet";
import dspWorkletUrl from "./audio/dsp-worklet?worker&url";
import dspWasmUrl from "./wasm-pkg/dsp_bg.wasm?url";
import type { AudioSourceBundle } from "./audio/AudioSource";

// Constructor refs are mutable so the HMR accept callback can swap them in.
// Vite's `accept(deps, cb)` does NOT rewire the file's static imports — the
// callback receives the new module exports and we update these refs.
let AppCtor: typeof App = App;
let ParamPanelCtor: typeof ParamPanel = ParamPanel;
let WorkletBridgeCtor: typeof WorkletBridge = WorkletBridge;

type SourceProvider = () => Promise<AudioSourceBundle>;

let pageDeps: AppDeps | null = null;
let dspWorklet: DspWorklet | null = null;
let app: App | null = null;
let panel: ParamPanel | null = null;
let bridge: WorkletBridge | null = null;
let initialBootstrap = true;
let reloading = false;

// Lifted out of `buildPageDeps` so reloads preserve user-tweaked param
// values. Created once on first start; the Bridge re-bootstraps it onto
// the fresh worklet on each reload.
let paramStore: ParamStore | null = null;
// Saved source provider, wrapped so subsequent invocations reuse the
// captured MediaStream (mic/tab) instead of re-prompting the user.
let sourceProvider: SourceProvider | null = null;

/**
 * Wrap a one-shot source factory so it can be called repeatedly: the first
 * call invokes the factory and caches its MediaStream. Every later call
 * builds a fresh AudioContext + source from the cached stream — no
 * `getUserMedia` / `getDisplayMedia` re-prompt. For test sources (no
 * stream), the factory itself is reusable, so we just call it again.
 */
function wrapAsReusable(factory: SourceProvider): SourceProvider {
  let cachedStream: MediaStream | null = null;
  let firstCall = true;
  return async () => {
    if (firstCall) {
      firstCall = false;
      const bundle = await factory();
      cachedStream = bundle.stream ?? null;
      return bundle;
    }
    if (cachedStream) {
      const context = new AudioContext();
      if (context.state === "suspended") await context.resume();
      const source = context.createMediaStreamSource(cachedStream);
      return { context, source, stream: cachedStream };
    }
    // Test source — no stream, but the factory itself is reusable.
    return factory();
  };
}

async function buildPageDeps(
  provider: SourceProvider,
  store: ParamStore,
): Promise<AppDeps> {
  const { context, source, stream } = await provider();

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

  dspWorklet = new DspWorklet({
    context,
    source,
    workletUrl: dspWorkletUrl,
    wasmUrl: dspWasmUrl,
  });
  const workletNode = await dspWorklet.start();

  const canvas = document.getElementById("app") as HTMLCanvasElement;
  const renderer = await createRenderer(canvas);

  const sr = context.sampleRate;
  const fftSize = store.get("dsp.windowSize");
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

  return { canvas, renderer, audioContext: context, workletNode, paramStore: store };
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
const onStart = async (factory: SourceProvider): Promise<void> => {
  if (started) return;
  started = true;
  buttons.style.display = "none";
  try {
    if (!paramStore) {
      paramStore = new ParamStore();
      for (const s of analysisSchemas) paramStore.register(s);
    }
    sourceProvider = wrapAsReusable(factory);
    pageDeps = await buildPageDeps(sourceProvider, paramStore);
    buildAppLayer(pageDeps);
  } catch (err) {
    started = false;
    pageDeps = null;
    sourceProvider = null;
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

/**
 * Wasm hot-reload. The wasm-pack glue inside the worklet caches its compiled
 * `wasm` instance globally per AudioContext (see `__wbg_init` short-circuit
 * in `src/wasm-pkg/dsp.js`), so swapping just the AudioWorkletNode within
 * the same context can't pick up new wasm. Instead, close the AudioContext
 * and rebuild it from a cached MediaStream — mic/tab permission survives,
 * the worklet bundle gets a fresh AudioWorkletGlobalScope, and the new
 * Dsp instance loads new wasm bytes.
 */
async function reloadWasm(): Promise<void> {
  if (!sourceProvider || !pageDeps || !paramStore || reloading) return;
  reloading = true;
  console.log("[wasm-reload] rebuilding AudioContext with fresh wasm…");
  try {
    teardownAppLayer();
    await pageDeps.audioContext.close();
    pageDeps = await buildPageDeps(sourceProvider, paramStore);
    initialBootstrap = true;
    buildAppLayer(pageDeps);
    console.log("[wasm-reload] done");
  } catch (err) {
    console.error("[wasm-reload] failed:", err);
  } finally {
    reloading = false;
  }
}

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

  // Auto-reload wasm when `npm run wasm` rewrites `src/wasm-pkg/`. The
  // `wasm-hot-reload` Vite plugin (see vite.config.ts) watches that
  // directory and emits this custom event after debouncing wasm-pack's
  // multi-file output. The plugin also returns `[]` from `handleHotUpdate`
  // to suppress Vite's default HMR cascade — without that, the change
  // would propagate through the worklet's worker bundle (which can't HMR)
  // and escalate to a full page reload.
  import.meta.hot.on("wasm:reload", () => {
    void reloadWasm();
  });

  // Backtick / tilde — manual wasm reload trigger as a fallback (e.g. when
  // running `npm run wasm` from outside the dev server's watch root).
  window.addEventListener("keydown", (e) => {
    if (e.key !== "`" && e.key !== "~") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!pageDeps) return;
    e.preventDefault();
    void reloadWasm();
  });
}
