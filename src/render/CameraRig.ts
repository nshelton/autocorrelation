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

export interface SwimConfig {
  enabled: boolean;
  posRoughness: number;   // time rate of the position wander (rad/s base freq)
  posAmplitude: number;   // world units
  rotRoughness: number;   // time rate of the orientation wander
  rotAmplitude: number;   // radians (App converts from degrees)
}

const SWIM_OFF: SwimConfig = {
  enabled: false, posRoughness: 0, posAmplitude: 0, rotRoughness: 0, rotAmplitude: 0,
};

// Smooth fractional-brownian noise in ~[-1, 1]: three octaves of sine, summed
// and normalized. `phase` decorrelates channels so each axis wanders on its
// own. Non-integer octave ratio keeps the octaves from phase-aligning into an
// obvious beat. Stateless — depends only on (t, phase).
function swimNoise(t: number, phase: number): number {
  let v = 0, amp = 1, freq = 1, norm = 0;
  for (let o = 0; o < 3; o++) {
    v += amp * Math.sin(t * freq + phase * (o + 1) * 1.7);
    norm += amp;
    amp *= 0.5;
    freq *= 2.07;
  }
  return v / norm;
}

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

  // Additive "swim" wander layered on top of the base pose each frame.
  // swimPos holds the offset applied last frame so we can strip it before the
  // base update runs — OrbitControls reads camera.position as ground truth, so
  // a leaked offset would corrupt the orbit. Rotation needs no strip: every
  // base path (controls/tween/procedural) rewrites the quaternion fresh.
  private swim: SwimConfig = SWIM_OFF;
  private swimTime = 0;
  private swimPos = new Vector3();

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

  setSwim(cfg: SwimConfig): void {
    this.swim = cfg;
  }

  // Rescales the offset from currentTarget to camera.position to `distance`,
  // preserving the current orbit direction. Strips swimPos first (like
  // getPose) so a live swim wobble doesn't get baked into the new radius —
  // OrbitControls picks up the new position as ground truth on its next
  // update(), same as applyPose/tween writes.
  setDistance(distance: number): void {
    const dir = this.camera.position.clone().sub(this.swimPos).sub(this.currentTarget);
    if (dir.lengthSq() < 1e-8) dir.set(0, 0, 1);
    else dir.normalize();
    this.camera.position.copy(this.currentTarget).addScaledVector(dir, distance);
  }

  setAutorotate(degPerSec: number): void {
    // Use OrbitControls' native autoRotate so mouse drag still works and the
    // spin only pauses while the user is actively dragging (state !== NONE).
    // At update(dt) the orbit advances 2π/60·autoRotateSpeed rad/s, so
    // autoRotateSpeed = degPerSec / 6.
    this.controls.autoRotate = degPerSec !== 0;
    this.controls.autoRotateSpeed = degPerSec / 6;
  }

  update(dt: number): void {
    // Strip last frame's additive position swim so the base update (and any
    // state derived from camera.position) sees the clean pose.
    this.camera.position.sub(this.swimPos);
    this.swimPos.set(0, 0, 0);
    this.updateBase(dt);
    this.applySwim(dt);
  }

  // Additive wander on top of the base pose. Position is offset by swimPos
  // (stripped next frame); orientation is tilted in place (regenerated next
  // frame by the base update, so it can't accumulate).
  private applySwim(dt: number): void {
    if (!this.swim.enabled) return;
    this.swimTime += dt;
    const a = this.swim.posAmplitude;
    if (a > 0) {
      const t = this.swimTime * this.swim.posRoughness;
      this.swimPos.set(swimNoise(t, 0), swimNoise(t, 2.4), swimNoise(t, 4.8)).multiplyScalar(a);
      this.camera.position.add(this.swimPos);
    }
    const ra = this.swim.rotAmplitude;
    if (ra > 0) {
      const t = this.swimTime * this.swim.rotRoughness;
      this.camera.rotateX(swimNoise(t, 7.2) * ra);
      this.camera.rotateY(swimNoise(t, 9.6) * ra);
      this.camera.rotateZ(swimNoise(t, 12.0) * ra);
    }
  }

  private updateBase(dt: number): void {
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
        this.controls.enabled = true;
        resolve();
      }
      return;
    }

    // Pass dt so native autoRotate is frame-rate independent. Keep currentTarget
    // synced from controls (drag/pan moves it) so getPose/saved pose stay accurate.
    this.controls.update(dt);
    this.currentTarget.copy(this.controls.target);
  }

  dispose(): void {
    this.controls.dispose();
  }

  getPose(): CameraPose {
    return {
      // Subtract the live swim offset so a saved/persisted pose is the base
      // pose, not a wandered one.
      position: this.camera.position.clone().sub(this.swimPos),
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
    this.controls.enabled = true;
  }
}
