//! Wavetable generators for the synth engines. Run on the main thread (not
//! realtime), so they can use the JS `simplex-noise` dependency; the resulting
//! Float32Array is shipped to the Rust `Sequencer` via a `wavetable` message
//! and read there with linear interpolation.

import { createNoise2D } from "simplex-noise";

/**
 * Tiny deterministic PRNG (mulberry32) so a given seed always yields the same
 * waveform — stable timbre across reloads, and a hook for a future "reseed"
 * control. Mirrors how curl-noise.ts seeds its simplex fields.
 */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A seamless single-cycle wavetable sampled from 2D simplex noise around a
 * circle (start and end meet, so the cycle is periodic), then DC-removed and
 * peak-normalized to [-1, 1]. The organic, vocal-ish waveform is what gives the
 * "Simplex" engine its character — distinct from the saw's bright buzz.
 *
 * `radius` controls how much of the noise field one cycle traverses: larger →
 * more wiggles per cycle → brighter / more harmonically dense.
 */
export function makeSimplexWavetable(len = 2048, radius = 1.6, seed = 1): Float32Array {
  const noise2D = createNoise2D(mulberry32(seed));
  const table = new Float32Array(len);
  let mean = 0;
  for (let i = 0; i < len; i++) {
    const theta = (i / len) * Math.PI * 2;
    const v = noise2D(Math.cos(theta) * radius, Math.sin(theta) * radius);
    table[i] = v;
    mean += v;
  }
  mean /= len;

  let peak = 0;
  for (let i = 0; i < len; i++) {
    table[i] -= mean; // remove DC so the waveform is centered
    peak = Math.max(peak, Math.abs(table[i]));
  }
  if (peak > 0) {
    for (let i = 0; i < len; i++) table[i] /= peak;
  }
  return table;
}
