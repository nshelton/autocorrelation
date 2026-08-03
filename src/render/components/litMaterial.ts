import { MeshStandardNodeMaterial } from "three/webgpu";

// Every lit component mesh shares one physically-based look, driven by the
// Environment section's roughness/metalness sliders. Creation goes through
// here so the registry can sweep live materials on a slider move and seed
// materials created later (components can spawn mid-session) with the current
// look. Standard (not Lambert) because only MeshStandardNodeMaterial picks up
// scene.environment IBL in the WebGPU node renderer.
const live = new Set<MeshStandardNodeMaterial>();
let roughness = 0.5;
let metalness = 0.0;

export function makeLitMaterial(): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  mat.roughness = roughness;
  mat.metalness = metalness;
  live.add(mat);
  return mat;
}

export function releaseLitMaterial(mat: MeshStandardNodeMaterial): void {
  live.delete(mat);
  mat.dispose();
}

export function setLitLook(r: number, m: number): void {
  roughness = r;
  metalness = m;
  for (const mat of live) {
    mat.roughness = r;
    mat.metalness = m;
  }
}
