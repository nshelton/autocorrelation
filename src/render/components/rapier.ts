import RAPIER from "@dimforge/rapier3d-compat";

// rapier3d-compat's init() is NOT idempotent — each call re-instantiates the
// wasm module and rebinds the module-level instance. Components init
// concurrently, so a second call swaps the memory out from under any world or
// body the first one already created: their handles now index a different
// instance and the next setter panics ("RuntimeError: unreachable" out of
// rbSetLinearDamping et al). Memoize the promise so every component awaits the
// same single instantiation.
let ready: Promise<void> | null = null;

export function initRapier(): Promise<void> {
  return (ready ??= RAPIER.init());
}
