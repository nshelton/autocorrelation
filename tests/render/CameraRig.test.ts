// tests/render/CameraRig.test.ts
import { describe, it, expect } from "vitest";
import { PerspectiveCamera, Vector3 } from "three";
import { CameraRig } from "../../src/render/CameraRig";

describe("CameraRig", () => {
  const makeRig = () => new CameraRig(new PerspectiveCamera(60, 1, 0.1, 100));

  it("registers and immediately moves to a preset with duration 0", async () => {
    const rig = makeRig();
    rig.addPreset("front", {
      position: new Vector3(0, 0, 5),
      target: new Vector3(0, 0, 0),
    });

    await rig.goTo("front", { duration: 0 });

    expect(rig.camera.position.x).toBeCloseTo(0);
    expect(rig.camera.position.y).toBeCloseTo(0);
    expect(rig.camera.position.z).toBeCloseTo(5);
  });

  it("throws when goTo references an unknown preset", async () => {
    const rig = makeRig();
    await expect(rig.goTo("nope", { duration: 0 })).rejects.toThrow(/unknown preset/i);
  });

  it("tweens between presets when duration > 0", async () => {
    const rig = makeRig();
    rig.addPreset("front", { position: new Vector3(0, 0, 5), target: new Vector3() });
    rig.addPreset("side", { position: new Vector3(5, 0, 0), target: new Vector3() });

    await rig.goTo("front", { duration: 0 });

    const tween = rig.goTo("side", { duration: 1 });
    rig.update(0.5); // halfway with linear easing
    expect(rig.camera.position.x).toBeGreaterThan(0);
    expect(rig.camera.position.x).toBeLessThan(5);
    rig.update(0.5); // arrive
    await tween;
    expect(rig.camera.position.x).toBeCloseTo(5, 4);
    expect(rig.camera.position.z).toBeCloseTo(0, 4);
  });

  it("invokes the procedural controller on update when set", () => {
    const rig = makeRig();
    rig.addPreset("front", { position: new Vector3(0, 0, 5), target: new Vector3() });
    let called = 0;
    rig.setProceduralController((dt) => {
      called += 1;
      rig.camera.position.x += dt;
    });
    rig.update(0.1);
    rig.update(0.1);
    expect(called).toBe(2);
    expect(rig.camera.position.x).toBeCloseTo(0.2, 4);
  });

  it("clearing the procedural controller stops calls", () => {
    const rig = makeRig();
    let called = 0;
    rig.setProceduralController(() => { called += 1; });
    rig.update(0.1);
    rig.setProceduralController(null);
    rig.update(0.1);
    expect(called).toBe(1);
  });

  it("starting a tween clears any active procedural controller", async () => {
    const rig = makeRig();
    rig.addPreset("front", { position: new Vector3(0, 0, 5), target: new Vector3() });
    rig.addPreset("side", { position: new Vector3(5, 0, 0), target: new Vector3() });
    await rig.goTo("front", { duration: 0 });
    let proceduralRan = false;
    rig.setProceduralController(() => { proceduralRan = true; });
    const tween = rig.goTo("side", { duration: 1 });
    rig.update(0.5);
    rig.update(0.5);
    await tween;
    proceduralRan = false;
    rig.update(0.1);
    expect(proceduralRan).toBe(false);
  });
});
