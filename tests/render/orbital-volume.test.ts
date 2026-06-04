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
    // Cheapest config to start; crank up via panel. Heads off the hang
    // that ambitious defaults caused on Retina + integrated GPU.
    expect(d.volumeSteps).toBe(8);
    expect(d.shadowSteps).toBe(0);
    expect(d.density).toBe(50);
    expect(d.boundsRadius).toBe(8);
  });

  it("marks volumeSteps and shadowSteps as discrete with the spec'd options", () => {
    expect(OrbitalVolume.paramKinds?.volumeSteps).toBe("discrete");
    expect(OrbitalVolume.paramKinds?.shadowSteps).toBe("discrete");
    expect(OrbitalVolume.paramDiscreteOptions?.volumeSteps).toEqual(
      [8, 16, 24, 32],
    );
    expect(OrbitalVolume.paramDiscreteOptions?.shadowSteps).toEqual(
      [0, 2, 4, 8],
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
