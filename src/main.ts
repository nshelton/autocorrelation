import { App } from "./App";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Running…";
  const app = new App();
  await app.start(canvas);
});
