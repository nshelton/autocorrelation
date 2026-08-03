import { Color, PCFSoftShadowMap, PerspectiveCamera, Scene as ThreeScene } from "three";
import { WebGPURenderer } from "three/webgpu";

/**
 * Page-lifetime: creates the WebGPU renderer and registers a
 * window-resize listener that updates renderer size. The listener is
 * intentionally never removed — `createRenderer` is called exactly
 * once per page, and the renderer lives until the page is torn down.
 */
export async function createRenderer(canvas: HTMLCanvasElement): Promise<WebGPURenderer> {
  // trackTimestamp enables GPU timestamp queries (renderer.info.render.timestamp).
  // Self-disables if the adapter lacks the timestamp-query feature.
  const renderer = new WebGPURenderer({ canvas, antialias: true, trackTimestamp: true });
  await renderer.init();
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(new Color(0x0a0a0a), 1);
  // The shadow pass only runs while a shadow-casting light is in the scene,
  // so this costs nothing when the directional light is toggled off.
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFSoftShadowMap;
  window.addEventListener("resize", () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  });
  return renderer;
}

/**
 * App-lifetime: creates a fresh Scene and PerspectiveCamera. The
 * camera-side resize listener (aspect/projection update) is registered
 * by App.start() so it can be cleanly removed by App.dispose().
 */
export function createSceneAndCamera(): { scene: ThreeScene; camera: PerspectiveCamera } {
  const scene = new ThreeScene();
  const camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  return { scene, camera };
}
