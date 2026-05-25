import { PerspectiveCamera, Vector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

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

interface ActiveTween {
  from: CameraPose;
  to: CameraPose;
  elapsed: number;
  duration: number;
  easing: (t: number) => number;
  resolve: () => void;
}

const linearEase = (t: number) => t;

export class CameraRig {
  readonly camera: PerspectiveCamera;
  readonly controls: OrbitControls;
  private presets = new Map<string, CameraPose>();
  private currentTarget = new Vector3();
  private tween: ActiveTween | null = null;
  private procedural: ProceduralController | null = null;

  constructor(camera: PerspectiveCamera, domElement: HTMLElement) {
    this.camera = camera;
    this.controls = new OrbitControls(camera, domElement);
    // OrbitControls fires 'start' on pointerdown before its own enabled-check
    // gates motion, so a drag mid-tween cancels the tween even though the
    // controls are disabled at the moment of input.
    this.controls.addEventListener("start", () => {
      if (this.tween) {
        const resolve = this.tween.resolve;
        this.tween = null;
        this.controls.target.copy(this.currentTarget);
        this.controls.enabled = true;
        resolve();
      }
    });
  }

  addPreset(name: string, pose: CameraPose): void {
    this.presets.set(name, pose);
  }

  async goTo(name: string, opts: GoToOptions = {}): Promise<void> {
    const target = this.presets.get(name);
    if (!target) throw new Error(`unknown preset: ${name}`);

    const duration = opts.duration ?? 0;
    if (duration === 0) {
      this.applyPose(target);
      return;
    }

    this.procedural = null;
    this.controls.enabled = false;
    const from: CameraPose = {
      position: this.camera.position.clone(),
      target: this.currentTarget.clone(),
      fov: this.camera.fov,
    };

    return new Promise<void>((resolve) => {
      this.tween = {
        from,
        to: target,
        elapsed: 0,
        duration,
        easing: opts.easing ?? linearEase,
        resolve,
      };
    });
  }

  setProceduralController(fn: ProceduralController | null): void {
    this.procedural = fn;
    this.controls.enabled = fn === null;
  }

  update(dt: number): void {
    if (this.tween) {
      this.tween.elapsed += dt;
      const raw = Math.min(1, this.tween.elapsed / this.tween.duration);
      const t = this.tween.easing(raw);
      this.camera.position.lerpVectors(this.tween.from.position, this.tween.to.position, t);
      const tgt = new Vector3().lerpVectors(this.tween.from.target, this.tween.to.target, t);
      this.camera.lookAt(tgt);
      this.currentTarget.copy(tgt);
      if (this.tween.from.fov !== undefined && this.tween.to.fov !== undefined) {
        this.camera.fov = this.tween.from.fov + (this.tween.to.fov - this.tween.from.fov) * t;
        this.camera.updateProjectionMatrix();
      }
      if (raw >= 1) {
        const resolve = this.tween.resolve;
        this.tween = null;
        this.controls.target.copy(this.currentTarget);
        this.controls.enabled = this.procedural === null;
        resolve();
      }
      return;
    }

    if (this.procedural) {
      this.procedural(dt, this.camera);
      return;
    }

    this.controls.update();
  }

  dispose(): void {
    this.controls.dispose();
  }

  getPose(): CameraPose {
    return {
      position: this.camera.position.clone(),
      target: this.currentTarget.clone(),
      fov: this.camera.fov,
    };
  }

  setPose(pose: CameraPose): void {
    this.applyPose(pose);
  }

  private applyPose(pose: CameraPose): void {
    this.camera.position.copy(pose.position);
    this.camera.lookAt(pose.target);
    this.currentTarget.copy(pose.target);
    if (pose.fov !== undefined) {
      this.camera.fov = pose.fov;
      this.camera.updateProjectionMatrix();
    }
    this.controls.target.copy(pose.target);
    this.controls.enabled = this.procedural === null;
  }
}
