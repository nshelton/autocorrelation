import { describe, it, expect } from "vitest";
import { OrbitalVolume } from "../../src/render/components/OrbitalVolume";

describe("OrbitalVolume static schema", () => {
  it("has the expected identity strings", () => {
    expect(OrbitalVolume.id).toBe("orbitalVolume");
    expect(OrbitalVolume.label).toBe("Orbital Volume");
    expect(OrbitalVolume.paramPrefix).toBe("orbitalVolume");
  });

  it("declares all four documented params with defaults", () => {
    const d = OrbitalVolume.paramDefaults!;
    expect(d.volumeSteps).toBe(48);
    expect(d.shadowSteps).toBe(8);
    expect(d.density).toBe(50);
    expect(d.boundsRadius).toBe(8);
  });

  it("marks volumeSteps and shadowSteps as discrete with the spec'd options", () => {
    expect(OrbitalVolume.paramKinds?.volumeSteps).toBe("discrete");
    expect(OrbitalVolume.paramKinds?.shadowSteps).toBe("discrete");
    expect(OrbitalVolume.paramDiscreteOptions?.volumeSteps).toEqual(
      [16, 32, 48, 64, 96, 128],
    );
    expect(OrbitalVolume.paramDiscreteOptions?.shadowSteps).toEqual(
      [0, 4, 8, 16, 24],
    );
  });

  it("marks density and boundsRadius as continuous with the spec'd ranges", () => {
    const opts = OrbitalVolume.paramOpts!;
    expect(opts.density.min).toBe(0.1);
    expect(opts.density.max).toBe(500);
    expect(opts.boundsRadius.min).toBe(1);
    expect(opts.boundsRadius.max).toBe(20);
  });
});
