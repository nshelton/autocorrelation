import { Vector3 } from "three";
import type { LineLayoutFn } from "./LineRenderer";

/**
 * Linear x-axis layout: maps sample index linearly to x in [-1, 1],
 * value to y in [yOffset, yOffset + height]. Used for waveform and
 * RMS history.
 */
export function linearLayout(yOffset: number, height: number): LineLayoutFn {
  return (i, n, value) => {
    const x = n <= 1 ? 0 : (i / (n - 1)) * 2 - 1;
    return new Vector3(x, yOffset + value * height, 0);
  };
}

/**
 * Log-frequency layout for FFT spectrum: bin index → x via log2,
 * value → y in [yOffset, yOffset + height]. Bin 0 maps to x = -1,
 * bin (n-1) maps to x = +1.
 */
export function logSpectrumLayout(yOffset: number, height: number): LineLayoutFn {
  return (i, n, value) => {
    if (n <= 1) return new Vector3(0, yOffset + value * height, 0);
    // log2(i + 1) / log2(n) ∈ [0, log2(n) / log2(n)] for i in [0, n-1]
    // Want [0, 1] → [-1, 1]
    const t = Math.log2(i + 1) / Math.log2(n);
    return new Vector3(t * 2 - 1, yOffset + value * height, 0);
  };
}
