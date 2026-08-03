// Downscale the live canvas into a small JPEG data URL for preset thumbnails.
// JPEG at low quality, ~160px wide: a few KB each, which matters because every
// preset shares one localStorage key with a ~5MB budget.
//
// Must be called right after a frame finishes rendering. A WebGPU canvas has
// no preserved drawing buffer, so reading it at an arbitrary moment gives
// black — App captures inside the RAF loop, after renderAsync() resolves.
export function snapshotCanvas(canvas: HTMLCanvasElement, width = 160): string {
  const height = Math.max(1, Math.round((width * canvas.height) / canvas.width));
  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const ctx = scratch.getContext("2d");
  if (!ctx) return "";
  ctx.drawImage(canvas, 0, 0, width, height);
  return scratch.toDataURL("image/jpeg", 0.6);
}
