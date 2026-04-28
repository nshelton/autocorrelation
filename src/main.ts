import { App } from "./App";
import {
  createMicSource,
  createTabSource,
  createTestSource,
  type AudioSourceBundle,
} from "./audio/AudioSource";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const startMic = document.getElementById("start-mic") as HTMLButtonElement;
const startTab = document.getElementById("start-tab") as HTMLButtonElement;
const startTest = document.getElementById("start-test") as HTMLButtonElement;
const buttons = document.getElementById("start-buttons") as HTMLDivElement;

let started = false;
const start = async (factory: () => Promise<AudioSourceBundle>) => {
  if (started) return;
  started = true;
  startMic.disabled = true;
  startTab.disabled = true;
  if (startTest) startTest.disabled = true;
  buttons.style.opacity = "0.5";
  const app = new App();
  try {
    await app.start(canvas, factory);
  } catch (err) {
    started = false;
    startMic.disabled = false;
    startTab.disabled = false;
    if (startTest) startTest.disabled = false;
    buttons.style.opacity = "1";
    console.error("[app] start failed:", err);
    alert(err instanceof Error ? err.message : String(err));
  }
};

startMic.addEventListener("click", () => start(createMicSource));
startTab.addEventListener("click", () => start(createTabSource));
if (startTest) {
  startTest.addEventListener("click", () => start(() => createTestSource(440)));
}

// Keyboard shortcuts before start: T → test signal (440 Hz), any other key → mic.
// After start, the App's keyboard handlers take over (camera presets etc.).
window.addEventListener(
  "keydown",
  (e) => {
    if (started) return;
    if (e.key === "t" || e.key === "T") {
      start(() => createTestSource(440));
    } else {
      start(createMicSource);
    }
  },
  { once: true },
);
