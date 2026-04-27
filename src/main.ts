import { App } from "./App";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;

let started = false;
const start = async () => {
  if (started) return;
  started = true;
  startBtn.disabled = true;
  startBtn.textContent = "Running…";
  const app = new App();
  await app.start(canvas);
};

startBtn.addEventListener("click", start);
window.addEventListener("keydown", start, { once: true });
