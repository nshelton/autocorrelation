import { defineConfig, type Plugin } from "vite";
import { spawn } from "node:child_process";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

/**
 * Watches `src/wasm-pkg/` and pushes a custom `wasm:reload` event to the
 * dev client whenever wasm-pack rewrites the output. Suppresses Vite's
 * default HMR cascade (returning `[]` from `handleHotUpdate`) — the worklet
 * bundle imports the wasm-pack glue, and worklets can't HMR, so without
 * this the cascade escalates to a full page reload. Debounced because
 * wasm-pack writes several files in quick succession (`dsp_bg.wasm`,
 * `dsp.js`, two `.d.ts`s); we only want one reload per build.
 */
function wasmHotReload(): Plugin {
  let timer: ReturnType<typeof setTimeout> | null = null;
  return {
    name: "wasm-hot-reload",
    handleHotUpdate({ file, server }) {
      if (!file.includes("/src/wasm-pkg/")) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        server.ws.send({ type: "custom", event: "wasm:reload" });
      }, 100);
      return [];
    },
  };
}

/**
 * Watches the `crates/dsp/` Rust sources and re-runs `npm run wasm` on
 * change. wasm-pack's output writes then trip the `wasmHotReload` plugin
 * above, which sends `wasm:reload` to the client. End-to-end: save a .rs
 * file → wasm rebuilds → page reloads wasm without losing mic permission.
 *
 * Builds are debounced (200ms) and serialized — saves during an in-flight
 * build queue exactly one rebuild after it completes.
 */
function cargoWatch(): Plugin {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let building = false;
  let queued = false;

  async function runBuild() {
    if (building) {
      queued = true;
      return;
    }
    building = true;
    try {
      do {
        queued = false;
        console.log("[cargo-watch] rebuilding wasm…");
        const code = await new Promise<number>((resolve) => {
          const child = spawn("npm", ["run", "wasm"], {
            stdio: "inherit",
            shell: true,
          });
          child.on("close", (c) => resolve(c ?? -1));
        });
        if (code !== 0) {
          console.error(`[cargo-watch] wasm build failed (exit ${code})`);
        } else {
          console.log("[cargo-watch] wasm build done");
        }
      } while (queued);
    } finally {
      building = false;
    }
  }

  return {
    name: "cargo-watch",
    configureServer(server) {
      // Add Rust sources to Vite's existing chokidar watcher. Direct
      // `change` listener bypasses Vite's HMR pipeline — we just want
      // file events, no module-graph reasoning.
      server.watcher.add(["crates/dsp/**/*.rs", "crates/dsp/Cargo.toml"]);
      server.watcher.on("change", (file) => {
        const isRust =
          file.endsWith(".rs") || file.endsWith("Cargo.toml");
        if (!isRust || !file.includes("crates/dsp")) return;
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => {
          timer = null;
          void runBuild();
        }, 200);
      });
    },
  };
}

export default defineConfig({
  plugins: [wasm(), topLevelAwait(), wasmHotReload(), cargoWatch()],
  worker: {
    plugins: () => [wasm()],
  },
  resolve: {
    dedupe: ["three"],
  },
  server: {
    port: 5173,
    headers: {
      // Cross-origin isolation unlocks high-resolution performance.now() in
      // AudioWorkletGlobalScope (~5 µs in Chrome vs 1 ms otherwise). Used by
      // crates/dsp/src/perf.rs for dspPerf measurements.
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
