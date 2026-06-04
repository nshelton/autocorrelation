import { describe, expect, it } from "vitest";
import { PerspectiveCamera, Vector3 } from "three";
import { CameraRig } from "../../src/render/CameraRig";

function makeRig(): CameraRig {
  const camera = new PerspectiveCamera(60, 1, 0.1, 100);
  return new CameraRig(camera, document.createElement("div"));
}

const xzRadius = (p: Vector3, c: Vector3) => Math.hypot(p.x - c.x, p.z - c.z);
const azimuth = (p: Vector3, c: Vector3) => Math.atan2(p.x - c.x, p.z - c.z);

describe("CameraRig autorotate", () => {
  it("advances azimuth ~90° in 1s at 90 deg/s, preserving radius and elevation", () => {
    const rig = makeRig();
    const center = new Vector3(0, 1, 0);
    rig.setPose({ position: new Vector3(0, 2, 4), target: center });
    rig.setAutorotate(90);
    rig.update(1);

    expect(Math.abs(azimuth(rig.camera.position, center))).toBeCloseTo(Math.PI / 2, 3);
    expect(rig.camera.position.y).toBeCloseTo(2, 6); // elevation unchanged
    expect(xzRadius(rig.camera.position, center)).toBeCloseTo(4, 4); // radius unchanged
  });

  it("orbits around the live controls target", () => {
    const rig = makeRig();
    const center = new Vector3(0, 1, 0);
    rig.setPose({ position: new Vector3(0, 0, 3), target: center });
    rig.setAutorotate(45);
    rig.update(0.5);
    expect(rig.controls.target.y).toBeCloseTo(1, 6);
    expect(xzRadius(rig.camera.position, center)).toBeCloseTo(3, 4);
  });

  it("toggles native autoRotate while leaving mouse controls enabled", () => {
    const rig = makeRig();
    rig.setAutorotate(30);
    expect(rig.controls.autoRotate).toBe(true);
    expect(rig.controls.autoRotateSpeed).toBeCloseTo(5, 6); // 30 / 6
    expect(rig.controls.enabled).toBe(true); // drag still works while spinning
    rig.setAutorotate(0);
    expect(rig.controls.autoRotate).toBe(false);
  });
});
