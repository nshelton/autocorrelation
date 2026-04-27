// AudioWorkletGlobalScope is missing TextDecoder/TextEncoder. The wasm-pack
// `--target web` glue instantiates a TextDecoder at module-load time for
// decoding Rust panic strings. Our passthrough never panics, so a stub that
// returns an empty string is sufficient to let the WASM module finish loading.
//
// This file must be imported BEFORE any wasm-pack glue. ES module imports are
// evaluated in source order before the importing module's body runs, so
// importing this file first guarantees the polyfill is in place when the WASM
// glue evaluates.

const g = globalThis as unknown as {
  TextDecoder?: unknown;
  TextEncoder?: unknown;
};

if (typeof g.TextDecoder === "undefined") {
  g.TextDecoder = class {
    decode(_buf?: BufferSource): string {
      return "";
    }
  };
}

if (typeof g.TextEncoder === "undefined") {
  g.TextEncoder = class {
    encode(_s?: string): Uint8Array {
      return new Uint8Array(0);
    }
  };
}
