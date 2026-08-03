import { DataTexture, RedFormat, UnsignedByteType, RepeatWrapping, NearestFilter } from "three";

// Void-and-cluster (Ulichney 1993) blue-noise mask generator. Ranks every
// pixel of a toroidal grid by the order it would join/leave an evenly
// dispersed "prototype" pattern, so thresholding the result at ANY level
// yields a blue-noise (no clumping, no low-frequency energy) distribution.
// Runs once (result is cached) — no baked texture asset needed.
export const BLUE_NOISE_SIZE = 64;

const SIGMA = 1.5;
const KERNEL_RADIUS = 5;
const SEED_DENSITY = 0.1;

type Kernel = { dx: number; dy: number; w: number }[];

function buildKernel(): Kernel {
  const k: Kernel = [];
  for (let dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
    for (let dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {
      k.push({ dx, dy, w: Math.exp(-(dx * dx + dy * dy) / (2 * SIGMA * SIGMA)) });
    }
  }
  return k;
}

// Toroidal Gaussian energy field: incremental add/remove of one point's
// contribution, wrapped so the pattern tiles seamlessly.
function addEnergy(energy: Float32Array, kernel: Kernel, x: number, y: number, sign: number): void {
  for (const { dx, dy, w } of kernel) {
    const ex = (x + dx + BLUE_NOISE_SIZE) % BLUE_NOISE_SIZE;
    const ey = (y + dy + BLUE_NOISE_SIZE) % BLUE_NOISE_SIZE;
    energy[ey * BLUE_NOISE_SIZE + ex] += sign * w;
  }
}

function tightestCluster(pattern: Uint8Array, energy: Float32Array): number {
  let best = -1, bestE = -Infinity;
  for (let i = 0; i < pattern.length; i++) {
    if (pattern[i] === 1 && energy[i] > bestE) { bestE = energy[i]; best = i; }
  }
  return best;
}

function largestVoid(pattern: Uint8Array, energy: Float32Array): number {
  let best = -1, bestE = Infinity;
  for (let i = 0; i < pattern.length; i++) {
    if (pattern[i] === 0 && energy[i] < bestE) { bestE = energy[i]; best = i; }
  }
  return best;
}

// mulberry32 — deterministic so the mask is stable across reloads.
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Prototype {
  pattern: Uint8Array;
  energy: Float32Array;
  count: number;
}

// Phase 1: seed a random pattern at SEED_DENSITY, then repeatedly relocate
// the tightest cluster to the largest void until the swap is a no-op.
function buildPrototype(kernel: Kernel, rng: () => number): Prototype {
  const n = BLUE_NOISE_SIZE * BLUE_NOISE_SIZE;
  const pattern = new Uint8Array(n);
  const energy = new Float32Array(n);
  const count = Math.round(n * SEED_DENSITY);

  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  for (let i = 0; i < count; i++) {
    const idx = indices[i];
    pattern[idx] = 1;
    addEnergy(energy, kernel, idx % BLUE_NOISE_SIZE, (idx / BLUE_NOISE_SIZE) | 0, 1);
  }

  // Bounded defensively — converges in practice within a few hundred swaps,
  // this just guards against a float-precision tie oscillating forever.
  for (let iter = 0; iter < 50 * n; iter++) {
    const cluster = tightestCluster(pattern, energy);
    pattern[cluster] = 0;
    addEnergy(energy, kernel, cluster % BLUE_NOISE_SIZE, (cluster / BLUE_NOISE_SIZE) | 0, -1);

    const empty = largestVoid(pattern, energy);
    if (empty === cluster) {
      pattern[cluster] = 1;
      addEnergy(energy, kernel, cluster % BLUE_NOISE_SIZE, (cluster / BLUE_NOISE_SIZE) | 0, 1);
      break;
    }
    pattern[empty] = 1;
    addEnergy(energy, kernel, empty % BLUE_NOISE_SIZE, (empty / BLUE_NOISE_SIZE) | 0, 1);
  }

  return { pattern, energy, count };
}

// Phase 2: rank every pixel — below the seed density by repeatedly removing
// the tightest cluster from the prototype, above it by repeatedly filling
// the largest void. The combined order is the blue-noise mask.
function rankPixels(kernel: Kernel, proto: Prototype): Uint32Array {
  const n = BLUE_NOISE_SIZE * BLUE_NOISE_SIZE;
  const rank = new Uint32Array(n);

  const pattern = proto.pattern.slice();
  const energy = proto.energy.slice();
  for (let r = proto.count - 1; r >= 0; r--) {
    const idx = tightestCluster(pattern, energy);
    pattern[idx] = 0;
    addEnergy(energy, kernel, idx % BLUE_NOISE_SIZE, (idx / BLUE_NOISE_SIZE) | 0, -1);
    rank[idx] = r;
  }

  pattern.set(proto.pattern);
  energy.set(proto.energy);
  for (let r = proto.count; r < n; r++) {
    const idx = largestVoid(pattern, energy);
    pattern[idx] = 1;
    addEnergy(energy, kernel, idx % BLUE_NOISE_SIZE, (idx / BLUE_NOISE_SIZE) | 0, 1);
    rank[idx] = r;
  }

  return rank;
}

let cached: DataTexture | null = null;

export function getBlueNoiseTexture(): DataTexture {
  if (cached) return cached;

  const kernel = buildKernel();
  const proto = buildPrototype(kernel, mulberry32(1));
  const rank = rankPixels(kernel, proto);

  const n = BLUE_NOISE_SIZE * BLUE_NOISE_SIZE;
  const data = new Uint8Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.floor((rank[i] / n) * 256);

  const tex = new DataTexture(data, BLUE_NOISE_SIZE, BLUE_NOISE_SIZE, RedFormat, UnsignedByteType);
  tex.wrapS = RepeatWrapping;
  tex.wrapT = RepeatWrapping;
  tex.magFilter = NearestFilter;
  tex.minFilter = NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;

  cached = tex;
  return tex;
}
