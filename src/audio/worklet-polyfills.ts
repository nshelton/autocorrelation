// Polyfills for older browsers where AudioWorkletGlobalScope lacks TextDecoder/
// TextEncoder. wasm-pack `--target web` glue uses both at module-load time and
// for every Rust↔JS string round-trip (e.g. `Dsp::buffer_names()`), so a no-op
// stub silently corrupts strings to "". Real UTF-8 codecs only — wasm-bindgen
// always passes Uint8Array.
//
// Must be imported BEFORE any wasm-pack glue so the polyfills land before that
// module's top-level `new TextDecoder(...)` evaluates.

const g = globalThis as unknown as {
  TextDecoder?: unknown;
  TextEncoder?: unknown;
};

if (typeof g.TextDecoder === "undefined") {
  g.TextDecoder = class {
    // wasm-bindgen does a warmup `decode()` with no args at module load.
    decode(bytes?: Uint8Array): string {
      if (!bytes) return "";
      let out = "";
      for (let i = 0; i < bytes.length; ) {
        const b0 = bytes[i++];
        if (b0 < 0x80) {
          out += String.fromCharCode(b0);
        } else if (b0 < 0xc0) {
          out += "�";
        } else if (b0 < 0xe0) {
          out += String.fromCharCode(((b0 & 0x1f) << 6) | (bytes[i++] & 0x3f));
        } else if (b0 < 0xf0) {
          const b1 = bytes[i++] & 0x3f;
          const b2 = bytes[i++] & 0x3f;
          out += String.fromCharCode(((b0 & 0x0f) << 12) | (b1 << 6) | b2);
        } else {
          const b1 = bytes[i++] & 0x3f;
          const b2 = bytes[i++] & 0x3f;
          const b3 = bytes[i++] & 0x3f;
          const cp = (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3) - 0x10000;
          out += String.fromCharCode(0xd800 | (cp >> 10), 0xdc00 | (cp & 0x3ff));
        }
      }
      return out;
    }
  };
}

if (typeof g.TextEncoder === "undefined") {
  g.TextEncoder = class {
    encode(s?: string): Uint8Array {
      if (!s) return new Uint8Array(0);
      const bytes: number[] = [];
      for (let i = 0; i < s.length; i++) {
        const cp = s.charCodeAt(i);
        if (cp < 0x80) {
          bytes.push(cp);
        } else if (cp < 0x800) {
          bytes.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
        } else {
          bytes.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
        }
      }
      return new Uint8Array(bytes);
    }
  };
}
