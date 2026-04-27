import { Vector3 } from "three";
import { createScene } from "./render/Scene";
import { CameraRig } from "./render/CameraRig";
import { LineRenderer } from "./render/LineRenderer";

export class App {
  private rig!: CameraRig;
  private line!: LineRenderer;
  private buffer = new Float32Array(2048);
  private last = 0;

  async start(canvas: HTMLCanvasElement): Promise<void> {
    const { scene, camera, renderer } = await createScene(canvas);

    this.rig = new CameraRig(camera);
    this.rig.addPreset("front", {
      position: new Vector3(0, 0, 3),
      target: new Vector3(0, 0, 0),
    });
    this.rig.addPreset("side", {
      position: new Vector3(3, 1, 0),
      target: new Vector3(0, 0, 0),
    });
    await this.rig.goTo("front", { duration: 0 });

    this.line = new LineRenderer({
      source: () => this.buffer,
      color: 0x66ffcc,
    });
    scene.add(this.line.object3d);

    let toggled = false;
    window.addEventListener("keydown", (e) => {
      if (e.key !== " ") return;
      toggled = !toggled;
      this.rig.goTo(toggled ? "side" : "front", { duration: 0.8 });
    });

    const loop = (now: number) => {
      const dt = this.last === 0 ? 0 : (now - this.last) / 1000;
      this.last = now;

      // Synthetic sine wave for v1 rendering proof.
      const t = now / 1000;
      for (let i = 0; i < this.buffer.length; i++) {
        const phase = (i / this.buffer.length) * Math.PI * 4;
        this.buffer[i] = 0.5 * Math.sin(phase + t * 2);
      }

      this.rig.update(dt);
      this.line.update();
      renderer.render(scene, camera);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
