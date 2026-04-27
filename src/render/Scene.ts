import { Color, PerspectiveCamera, Scene as ThreeScene } from "three";
import { WebGPURenderer } from "three/webgpu";

export interface SceneBundle {
  scene: ThreeScene;
  camera: PerspectiveCamera;
  renderer: WebGPURenderer;
}

export async function createScene(canvas: HTMLCanvasElement): Promise<SceneBundle> {
  const renderer = new WebGPURenderer({ canvas, antialias: true });
  await renderer.init();
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(new Color(0x0a0a0a), 1);

  const scene = new ThreeScene();
  const camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);

  window.addEventListener("resize", () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  });

  return { scene, camera, renderer };
}
