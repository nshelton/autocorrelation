import { Vector3 } from "three";

export type LineLayoutFn = (i: number, n: number, value: number) => Vector3;
export type LineLayout = "linear" | "log" | "logRight" | "logY";

const linearLayout: LineLayoutFn = (i, n, value) => {
  const x = n <= 1 ? 0 : i / (n - 1);
  return new Vector3(x, value, 0);
};

const logYLayout: LineLayoutFn = (i, n, value) => {
  const x = n <= 1 ? 0 : i / (n - 1);
  const y = Math.log2(value + 1);
  return new Vector3(x, y, 0);
};

const logLeftLayout: LineLayoutFn = (i, n, value) => {
  if (n <= 1) return new Vector3(0, value, 0);
  // log2(i + 1) / log2(n) ∈ [0, 1] for i in [0, n-1]
  const t = Math.log2(i + 1) / Math.log2(n);
  const y = Math.log2(value + 1);
  return new Vector3(t, y, 0);
};

const logRightLayout: LineLayoutFn = (i, n, value) => {
  if (n <= 1) return new Vector3(0, value, 0);
  const t = Math.log2(n - i + 1) / Math.log2(n);
  const y = Math.log2(value + 1);
  return new Vector3(t, y, 0);
};

export const layouts: Record<LineLayout, LineLayoutFn> = {
  linear: linearLayout,
  log: logLeftLayout,
  logRight: logRightLayout,
  logY: logYLayout,
};
