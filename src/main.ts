import { App, type AppDeps } from "./App";
import { ParamStore } from "./params/ParamStore";
import { ParamPanel } from "./params/ParamPanel";
import { WorkletBridge } from "./params/WorkletBridge";
import { analysisSchemas } from "./params/schemas";
import { createRenderer } from "./render/Scene";
import { DspWorklet } from "./audio/DspWorklet";
import { SynthWorklet } from "./audio/SynthWorklet";
import {
  demoProject,
  flattenBends,
  flattenToSchedule,
  makeDrumTrack,
  makeSynthTrack,
  type Instrument,
  type Project,
} from "./sequencer/model";
import { DEFAULT_TUNING, tuneMidi } from "./sequencer/tuning";
import { PianoRoll } from "./sequencer/PianoRoll";
import { DrumMachine } from "./sequencer/DrumMachine";
import { AnalysisPanel } from "./sequencer/AnalysisPanel";
import { SynthPanel } from "./sequencer/SynthPanel";
import { SequencerLayout } from "./sequencer/SequencerLayout";
import { Arrangement } from "./sequencer/Arrangement";
import { loadState, saveState } from "./sequencer/persistence";
import { makeSimplexWavetable } from "./sequencer/wavetables";
import dspWorkletUrl from "./audio/dsp-worklet?worker&url";
import synthWorkletUrl from "./audio/synth-worklet?worker&url";
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
  const fftSize = store.get("dsp.windowSize") as number;
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

  return {
    canvas,
    renderer,
    audioContext: context,
    workletNode,
    paramStore: store,
  };
}

function buildAppLayer(deps: AppDeps): void {
  app = new AppCtor(deps);
  app.start();                                          // constructs modulator
  panel = new ParamPanelCtor(deps.paramStore, app.modulator);
  bridge = new WorkletBridgeCtor(deps.paramStore, deps.workletNode.port);
  app.bindUI(panel.scenes);
  app.bindCameraUI(panel.camera);
  app.bindPostUI(panel.post);
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
const startSynth = document.getElementById("start-synth") as HTMLButtonElement;
const buttons = document.getElementById("start-buttons") as HTMLDivElement;

let started = false;
const onStart = async (factory: SourceProvider): Promise<void> => {
  if (started) return;
  started = true;
  buttons.style.display = "none";
  try {
    if (!paramStore) {
      paramStore = new ParamStore();
      const { postSchemas } = await import("./params/postSchemas");
      const { cameraSchemas } = await import("./params/cameraSchemas");
      const { lightSchemas } = await import("./params/lightSchemas");
      for (const s of analysisSchemas) paramStore.register(s);
      for (const s of postSchemas) paramStore.register(s);
      for (const s of cameraSchemas) paramStore.register(s);
      for (const s of lightSchemas) paramStore.register(s);
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

// --- Synth mode (Phase 0) -------------------------------------------------
// Self-contained boot path: no audio source, no analysis App. Spins up the
// synth worklet (→ destination) and maps the computer keyboard to MIDI notes
// so we can hear the voice and prove the worklet/wasm/graph wiring end to end.
// The full sequencer, timeline UI, and scene visualization land in later phases.

// Isomorphic computer-keyboard piano (two octaves). Lower row z..m = white/black
// keys of one octave; q..i = the octave above. Values are semitone offsets from
// BASE_MIDI.
const KEY_TO_SEMITONE: Record<string, number> = {
  z: 0, s: 1, x: 2, d: 3, c: 4, v: 5, g: 6, b: 7, h: 8, n: 9, j: 10, m: 11,
  q: 12, "2": 13, w: 14, "3": 15, e: 16, r: 17, "5": 18, t: 19, "6": 20,
  y: 21, "7": 22, u: 23, i: 24,
};
const BASE_MIDI = 48; // C3

async function onStartSynth(): Promise<void> {
  if (started) return;
  started = true;
  buttons.style.display = "none";
  try {
    const context = new AudioContext();
    // When synth mode auto-boots (no click), the AudioContext starts suspended
    // under the browser autoplay policy. Building the worklet on a suspended
    // context is fine — it just stays silent until the first user gesture, when
    // we resume. (When started from a button click, the immediate resume below
    // already succeeds and the gesture listeners never fire.)
    const tryResume = () => context.resume().catch(() => {});
    tryResume();
    if (context.state !== "running") {
      const onGesture = () => {
        tryResume();
        window.removeEventListener("pointerdown", onGesture);
        window.removeEventListener("keydown", onGesture);
      };
      window.addEventListener("pointerdown", onGesture);
      window.addEventListener("keydown", onGesture);
    }
    const synth = new SynthWorklet({
      context,
      workletUrl: synthWorkletUrl,
      wasmUrl: dspWasmUrl,
    });
    const node = await synth.start();
    console.log("[synth] started; sampleRate:", context.sampleRate, "Hz");
    console.log(
      "[synth] keys z..m/q..i = live notes · Space = play/pause · drop a .mid to load",
    );
    console.log(
      "[synth] piano-roll: click-drag to paint · drag body/edge to move/resize · right-click to delete · double-click to bend",
    );
    console.log(
      "[synth] select: Shift+drag marquee · Shift+click toggle · drag selection to move · Cmd/Ctrl+C/V copy/paste · Delete to remove",
    );

    // Shared Simplex wavetable (generated here from simplex-noise); sent to
    // every track's instrument so the Simplex engine works on any track.
    const wavetable = makeSimplexWavetable();

    // Restore the saved project (notes, per-track instruments, tempo, loop) if
    // present; otherwise start from the demo.
    const saved = loadState();
    let project = saved?.project ?? demoProject();
    let selectedTrack = 0;

    // Persist on any edit, debounced so a slider/note drag doesn't thrash
    // localStorage; flush on page hide to catch the trailing debounce window.
    let saveTimer: ReturnType<typeof setTimeout> | undefined;
    const save = (): void => {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => saveState(project, selectedTrack), 300);
    };
    window.addEventListener("pagehide", () => saveState(project, selectedTrack));

    // --- Worklet senders -------------------------------------------------
    const pushInstrument = (i: number, inst: Instrument): void => {
      // Kind FIRST — it (re)builds the track's instrument, so engine/wavetable/
      // params must land on the kind that's now there.
      node.port.postMessage({ type: "kind", track: i, kind: inst.kind === "drums" ? 1 : 0 });
      node.port.postMessage({ type: "engine", track: i, index: inst.engine });
      node.port.postMessage({ type: "wavetable", track: i, data: wavetable });
      for (const [key, value] of Object.entries(inst.params)) {
        node.port.postMessage({ type: "param", track: i, key, value });
      }
    };
    const pushSchedule = (p: Project): void => {
      node.port.postMessage({ type: "schedule", events: flattenToSchedule(p) });
      // Pitch-bend curves travel beside the schedule (matched to note-ons by
      // identity); always resend so a removed bend clears worklet-side too.
      node.port.postMessage({ type: "bends", data: flattenBends(p) });
    };
    // Active loop range: the gold region when enabled, else the whole view.
    const activeLoopStart = (p: Project): number =>
      p.loopEnabled ? p.loopRegionStart : p.loopStart;
    const sendLoop = (p: Project): void => {
      node.port.postMessage({
        type: "loop",
        start: activeLoopStart(p),
        end: p.loopEnabled ? p.loopRegionEnd : p.loopEnd,
        enabled: true,
      });
    };

    // --- UI (forward references resolved by call time) -------------------
    // Which clip is open in the editor + params (null = nothing selected, panes
    // show placeholders). One clip per track for now, so clip is always 0.
    let selectedClip: { track: number; clip: number } | null = null;

    // The DAW shell: global transport bar + resizable arrangement/editor/params
    // panes. It owns the `--track-bar-h` / `--synth-panel-h` vars the editors
    // fill, so resizing a pane reflows the editor with no editor-side changes.
    const layout = new SequencerLayout({
      onPlayPause: () => toggleTransport(),
      onStop: () => stopTransport(),
      onTempo: (bpm) => setTempo(bpm),
      onLoopLength: (beats) => setLoopLength(beats),
      onLoopToggle: () => toggleLoopRegion(),
      onAddTrack: (kind) => addTrack(kind),
      onToggleTuning: () => toggleTuning(),
      onSetRoot: (pc) => setTuningRoot(pc),
    });

    const synthPanel = new SynthPanel(
      layout.paramsContainer,
      (key, value) => {
        node.port.postMessage({ type: "param", track: selectedTrack, key, value });
        save();
      },
      (index) => {
        node.port.postMessage({ type: "engine", track: selectedTrack, index });
        save();
      },
    );
    const pianoRoll = new PianoRoll(project, {
      onNotesChange: (p) => {
        pushSchedule(p);
        save();
      },
      // The gold loop-region edge drag still lives in the roll's ruler; keep the
      // global loop buttons in sync when it changes the region.
      onLoopChange: (p) => {
        sendLoop(p);
        layout.setProject(p);
        save();
      },
      // Ruler scrub seeks the transport.
      onSeek: (beat) => node.port.postMessage({ type: "transport", action: "seek", beat }),
      // Transport/tempo now come from the global bar, not the roll's toolbar.
      // Persist this track's zoom/scroll (per-track view); save is debounced.
      onViewChange: () => {
        const t = project.tracks[selectedTrack];
        if (t && t.instrument.kind !== "drums") {
          t.view = pianoRoll.getView();
          save();
        }
      },
    });

    // The drum-machine step grid — shown in place of the piano roll when the
    // selected track is a drum kit. Shares the same transport/schedule wiring;
    // a drum pattern is just notes (see DrumMachine / model.flattenToSchedule).
    const drumMachine = new DrumMachine(project, {
      onNotesChange: (p) => {
        pushSchedule(p);
        save();
      },
      onAudition: (midi) =>
        node.port.postMessage({ type: "noteOn", track: selectedTrack, midi, velocity: 0.9 }),
    });

    // The arrangement timeline (top pane) — clips as thumbnail blocks. Clicking
    // a clip opens it in the editor + params.
    const arrangement = new Arrangement(project, layout.arrangementContainer, {
      onSelectClip: (track, clip) => {
        selectClip(track, clip);
        save();
      },
      onVolume: (track, value) => setTrackVolume(track, value),
    });

    // Output scope: live waveform + spectrum of the synth's post-mix output,
    // tapped off the worklet node (own AnalyserNode + rAF, self-managed).
    new AnalysisPanel(context, node);

    // Track volume fader (arrangement) → the track's instrument gain. Drums have
    // no synth panel, so this is their only level control; for synth tracks it's
    // the same value as the panel's gain slider, so refresh that when it's open.
    const setTrackVolume = (track: number, value: number): void => {
      const t = project.tracks[track];
      if (!t) return;
      t.instrument.params.gain = value;
      node.port.postMessage({ type: "param", track, key: "gain", value });
      if (track === selectedTrack && t.instrument.kind !== "drums") {
        synthPanel.setInstrument(t.instrument, t.name);
      }
      save();
    };

    // Open a clip: route the editor (piano roll vs drum grid) and params at its
    // track. `selectedTrack` stays the worklet-facing index (live keys, params).
    const selectClip = (track: number, clip: number): void => {
      selectedClip = { track, clip };
      selectedTrack = track;
      const t = project.tracks[track];
      const isDrum = t?.instrument.kind === "drums";
      layout.setEditorPlaceholder(false);
      pianoRoll.setVisible(!isDrum);
      drumMachine.setVisible(isDrum);
      if (isDrum) {
        drumMachine.setSelectedTrack(track);
        synthPanel.setVisible(false);
        layout.setParamsPlaceholder(true, "Drum kit — no parameters yet");
      } else {
        pianoRoll.setSelectedTrack(track);
        pianoRoll.setView(t?.view); // restore this track's zoom/scroll (else auto-fit)
        if (t) synthPanel.setInstrument(t.instrument, t.name);
        synthPanel.setVisible(true);
        layout.setParamsPlaceholder(false);
      }
      arrangement.setSelected(track, clip);
    };

    // Nothing selected: editor + params show placeholders.
    const deselect = (): void => {
      selectedClip = null;
      pianoRoll.setVisible(false);
      drumMachine.setVisible(false);
      synthPanel.setVisible(false);
      layout.setEditorPlaceholder(true);
      layout.setParamsPlaceholder(true, "Select a clip in the arrangement");
      arrangement.setSelected(-1, -1);
    };

    // Tempo from the global transport bar (the editors mutate project.bpm
    // themselves; this path is for the global control).
    const setTempo = (bpm: number): void => {
      project.bpm = bpm;
      node.port.postMessage({ type: "tempo", bpm });
      save();
    };
    // Tuning toggle / root: re-flatten the schedule so the new tuning is heard.
    // (Currently-sounding notes retune on their next trigger / the next loop.)
    const toggleTuning = (): void => {
      const t = project.tuning ?? { ...DEFAULT_TUNING };
      project.tuning = { mode: t.mode === "just" ? "equal" : "just", root: t.root };
      pushSchedule(project);
      layout.setProject(project);
      save();
    };
    const setTuningRoot = (pc: number): void => {
      const t = project.tuning ?? { ...DEFAULT_TUNING };
      project.tuning = { mode: t.mode, root: ((pc % 12) + 12) % 12 };
      pushSchedule(project);
      layout.setProject(project);
      save();
    };
    // Re-push the project to every view after a loop change (loop length resizes
    // the editors' time window and the drum grid's step span).
    const resyncViews = (): void => {
      pianoRoll.setProject(project);
      drumMachine.setProject(project);
      arrangement.setProject(project);
      layout.setProject(project);
      if (selectedClip) selectClip(selectedClip.track, selectedClip.clip);
    };
    const setLoopLength = (beats: number): void => {
      project.loopEnd = project.loopStart + beats;
      project.loopRegionStart = Math.max(
        project.loopStart,
        Math.min(project.loopRegionStart, project.loopEnd),
      );
      project.loopRegionEnd = Math.max(
        project.loopRegionStart,
        Math.min(project.loopRegionEnd, project.loopEnd),
      );
      sendLoop(project);
      resyncViews();
      save();
    };
    const toggleLoopRegion = (): void => {
      project.loopEnabled = !project.loopEnabled;
      if (project.loopEnabled && !(project.loopRegionEnd > project.loopRegionStart)) {
        project.loopRegionStart = project.loopStart;
        project.loopRegionEnd = project.loopEnd;
      }
      sendLoop(project);
      resyncViews();
      save();
    };

    // Push a full project to the worklet and views. Track count FIRST so the
    // per-track instrument + schedule messages land on synths that exist.
    const loadProject = (p: Project): void => {
      project = p;
      node.port.postMessage({ type: "trackCount", count: p.tracks.length });
      node.port.postMessage({ type: "tempo", bpm: p.bpm });
      p.tracks.forEach((track, i) => pushInstrument(i, track.instrument));
      pushSchedule(p); // schedule + bends
      sendLoop(p);
      node.port.postMessage({ type: "transport", action: "seek", beat: activeLoopStart(p) });
      pianoRoll.setProject(p);
      drumMachine.setProject(p);
      arrangement.setProject(p);
      layout.setProject(p);
      deselect(); // empty editor + params until a clip is clicked
    };

    // Append a new track (the entry point for adding a drum machine to a saved
    // project), push it to the worklet, and open it.
    const addTrack = (kind: "synth" | "drums"): void => {
      const index = project.tracks.length;
      project.tracks.push(kind === "drums" ? makeDrumTrack(index) : makeSynthTrack(index));
      loadProject(project);
      selectClip(index, 0);
      save();
    };
    loadProject(project);
    // Restore the persisted selection if any (else stay on the placeholder).
    if (saved && saved.selectedTrack < project.tracks.length) {
      selectClip(saved.selectedTrack, 0);
    }

    // Drag-drop MIDI import. Parsing is lazy-loaded so @tonejs/midi only enters
    // the bundle when actually used.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", async (e) => {
      e.preventDefault();
      const file = [...(e.dataTransfer?.files ?? [])].find((f) =>
        /\.midi?$/i.test(f.name),
      );
      if (!file) return;
      try {
        const { parseMidi } = await import("./sequencer/midi");
        const p = parseMidi(await file.arrayBuffer(), file.name);
        loadProject(p);
        save();
        const noteCount = p.tracks.reduce(
          (s, t) => s + t.clips.reduce((cs, c) => cs + c.notes.length, 0),
          0,
        );
        console.log(
          `[synth] loaded ${file.name}: ${p.bpm} BPM, ${p.tracks.length} track(s), ${noteCount} notes`,
        );
      } catch (err) {
        console.error("[synth] MIDI import failed:", err);
        alert("MIDI import failed: " + (err instanceof Error ? err.message : String(err)));
      }
    });

    let lastPlaying = false;
    node.port.onmessage = (e: MessageEvent) => {
      const m = e.data as { type?: string; beat?: number; playing?: boolean };
      if (m.type !== "playhead") return;
      pianoRoll.setPlayhead(m.beat ?? 0);
      drumMachine.setPlayhead(m.beat ?? 0);
      arrangement.setPlayhead(m.beat ?? 0);
      layout.setPlayhead(m.beat ?? 0);
      layout.setPlaying(!!m.playing);
      if (m.playing === lastPlaying) return;
      lastPlaying = !!m.playing;
      console.log(
        `[synth] transport ${m.playing ? "playing" : "paused"} @ beat ${(m.beat ?? 0).toFixed(2)}`,
      );
    };

    let playing = false;
    const toggleTransport = () => {
      playing = !playing;
      node.port.postMessage({ type: "transport", action: playing ? "play" : "pause" });
      layout.setPlaying(playing); // immediate feedback; the echo reconfirms
    };
    const stopTransport = () => {
      playing = false;
      node.port.postMessage({ type: "transport", action: "pause" });
      node.port.postMessage({ type: "transport", action: "seek", beat: activeLoopStart(project) });
      layout.setPlaying(false);
    };

    // Live keys play the selected track. Track held keys so OS key-repeat
    // doesn't re-trigger, and so keyup maps back to the same MIDI note.
    const held = new Map<string, number>();
    window.addEventListener("keydown", (e) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === " ") {
        e.preventDefault(); // stop the page from scrolling
        toggleTransport();
        return;
      }
      const semi = KEY_TO_SEMITONE[e.key.toLowerCase()];
      if (semi === undefined || held.has(e.key)) return;
      // Tune pitched (synth) tracks; drum lane MIDI must stay exact. Store the
      // tuned value so keyup's note-off matches the note-on.
      const isDrum = project.tracks[selectedTrack]?.instrument.kind === "drums";
      const midi = isDrum ? BASE_MIDI + semi : tuneMidi(BASE_MIDI + semi, project.tuning);
      held.set(e.key, midi);
      node.port.postMessage({ type: "noteOn", track: selectedTrack, midi, velocity: 0.9 });
    });
    window.addEventListener("keyup", (e) => {
      const midi = held.get(e.key);
      if (midi === undefined) return;
      held.delete(e.key);
      node.port.postMessage({ type: "noteOff", track: selectedTrack, midi });
    });
  } catch (err) {
    started = false;
    buttons.style.display = "";
    console.error("[synth] start failed:", err);
    alert(err instanceof Error ? err.message : String(err));
  }
}

startSynth.addEventListener("click", () => {
  void onStartSynth();
});

// Default to synth mode so it comes up without a click. Opt into the original
// analysis app with ?mode=analysis, which restores the source chooser (mic/tab
// buttons + any-key-starts-mic) instead of auto-booting the synth.
if (new URLSearchParams(location.search).get("mode") === "analysis") {
  window.addEventListener(
    "keydown",
    async () => {
      if (started) return;
      const { createMicSource } = await import("./audio/AudioSource");
      await onStart(createMicSource);
    },
    { once: true },
  );
} else {
  void onStartSynth();
}

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
