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
});
