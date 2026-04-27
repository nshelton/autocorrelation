// src/render/CameraRig.ts
import { PerspectiveCamera, Vector3 } from "three";

export interface CameraPose {
  position: Vector3;
  target: Vector3;
  fov?: number;
}

export interface GoToOptions {
  duration?: number;
  easing?: (t: number) => number;
}

export type ProceduralController = (dt: number, camera: PerspectiveCamera) => void;

export class CameraRig {
  readonly camera: PerspectiveCamera;
  private presets = new Map<string, CameraPose>();

  constructor(camera: PerspectiveCamera) {
    this.camera = camera;
  }

  addPreset(name: string, pose: CameraPose): void {
    this.presets.set(name, pose);
  }

  async goTo(name: string, opts: GoToOptions = {}): Promise<void> {
    const pose = this.presets.get(name);
    if (!pose) throw new Error(`unknown preset: ${name}`);

    const duration = opts.duration ?? 0;
    if (duration === 0) {
      this.applyPose(pose);
      return;
    }
    // Tween implementation lands in Task 4.
    throw new Error("tweening not yet implemented");
  }

  setProceduralController(_fn: ProceduralController | null): void {
    // Implemented in Task 4.
  }

  update(_dt: number): void {
    // Implemented in Task 4.
  }

  private applyPose(pose: CameraPose): void {
    this.camera.position.copy(pose.position);
    this.camera.lookAt(pose.target);
    if (pose.fov !== undefined) {
      this.camera.fov = pose.fov;
      this.camera.updateProjectionMatrix();
    }
  }
}
