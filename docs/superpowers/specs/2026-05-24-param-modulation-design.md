# Param Modulation Design

**Date:** 2026-05-24
**Status:** Approved, awaiting plan.

## Goal

Let any continuous param in the render/post/component/camera/light folders be modulated at runtime by an audio signal — band-energy levels (rmsLow / rmsMid / rmsHigh) or one of the four beat saws (beatPulses[0..3]). The slider remains the user-set "rest" value; a `depth` knob (0..1) controls how strongly the audio pulls the param across its `[min, max]` range.

DSP params (`dsp.*`) are excluded — modulating analysis params from audio creates a feedback loop (audio drives analyzer → analyzer drives modulator → modulator drives analyzer).

## Architecture

Three new pieces, no replacement of existing structures:

### 1. `Modulator` — `src/params/Modulator.ts`

The runtime engine. Holds a `Map<paramKey, { source: string, depth: number }>` of active bindings. Each frame:

1. For each binding, look up the source descriptor in `MOD_SOURCES`.
2. Read the latest sample from the named buffer in `FeatureStore`.
3. Read the base value from `ParamStore`, plus the schema's `min`/`max`.
4. Compute `effective = lerp(base, lerp(min, max, src), depth)`.
5. Call `paramStore.notify(key, effective)` — fires existing subscribers without persisting or mutating store state.

Public surface:

```ts
class Modulator {
  constructor(store: ParamStore, features: FeatureStore);
  setBinding(key: string, binding: { source: string; depth: number } | null): void;
  getBinding(key: string): { source: string; depth: number } | null;
  tick(): void;                                  // called per RAF frame
  subscribe(fn: (key: string) => void): () => void; // for UI two-way sync
  dispose(): void;
}
```

`setBinding(key, null)` removes the binding AND fires one final `notify(key, base)` so the param snaps back to slider position.

### 2. `ParamStore.notify(key, value)` — new method

```ts
notify(key: string, value: ParamValue): void {
  for (const fn of this.subscribers) fn(key, value);
}
```

No validation, no persistence, no mutation of `this.values`. The Modulator uses this to inject effective values through the channel existing consumers already subscribe on — **no consumer-side migration**.

The slider stays anchored to the base (still read via `store.get(key)`); the scene state reflects the effective value.

### 3. `bindParam()` helper — `src/params/bindParam.ts`

Wraps the existing pattern of `addBinding(...).on("change", ...)` plus the optional inline mod sub-folder.

```ts
function bindParam(
  folder: FolderApi,
  store: ParamStore,
  modulator: Modulator,
  schema: ParamSchema,
): BindingApi;
```

Behavior:
- Always adds the normal widget (slider / dropdown / checkbox) and wires its `change` event to `store.set(schema.key, value)`.
- For continuous schemas whose key does NOT start with `dsp.`, also appends a collapsed sub-folder titled `↳ mod` containing:
  - **Source** dropdown with the 8 entries from `MOD_SOURCES` (plus `none` first).
  - **Depth** slider 0..1, step 0.01, default 0.
- Wires the sub-folder widgets to `modulator.setBinding(schema.key, ...)`. When source is `none`, the binding is removed; otherwise stored.
- Subscribes to `modulator.subscribe` so persisted bindings loaded at startup show up correctly in the UI on initial render.

Call-sites to migrate:
- `ParamPanel.addWidget` — refactor to use `bindParam` (no-op for `dsp.*` continuous params and discrete/boolean — but unifies the path).
- `App.bindCameraUI` — replace hand-rolled `addBinding(...).on("change", ...)` for `camera.fov` with `bindParam`. `camera.preset` (discrete) and `light.directional.enabled` (boolean) bypass the mod sub-folder automatically.
- `App.bindPostUI` — replace the existing post-folder bindings (delegate path lives inside `PostStack.bindUI`).
- `ComponentManager.bindUI` — same refactor for component continuous params.

## Modulation Sources

Static table in `Modulator.ts`. Same string keys are used in the UI dropdown and in persistence.

| Key | Buffer | Sample |
|---|---|---|
| `none` | — | (sentinel: no modulation) |
| `rms.low` | `rmsLow` | `buf[buf.length-1]` |
| `rms.mid` | `rmsMid` | `buf[buf.length-1]` |
| `rms.high` | `rmsHigh` | `buf[buf.length-1]` |
| `beat.1x` | `beatPulses` | `buf[0]` |
| `beat.2x` | `beatPulses` | `buf[1]` |
| `beat.4x` | `beatPulses` | `buf[2]` |
| `beat.8x` | `beatPulses` | `buf[3]` |

```ts
const MOD_SOURCES: Record<string, { buffer: string; read: (b: Float32Array) => number }> = {
  "rms.low":  { buffer: "rmsLow",     read: latest },
  "rms.mid":  { buffer: "rmsMid",     read: latest },
  "rms.high": { buffer: "rmsHigh",    read: latest },
  "beat.1x":  { buffer: "beatPulses", read: (b) => b[0] ?? 0 },
  "beat.2x":  { buffer: "beatPulses", read: (b) => b[1] ?? 0 },
  "beat.4x":  { buffer: "beatPulses", read: (b) => b[2] ?? 0 },
  "beat.8x":  { buffer: "beatPulses", read: (b) => b[3] ?? 0 },
};

function latest(b: Float32Array): number {
  return b.length === 0 ? 0 : b[b.length - 1];
}
```

NaN guard in `tick()`: `if (Number.isNaN(src)) src = 0`. Beat values are NaN in silence — that should reduce to `effective = base`, not crash.

## Data Flow

```
audio frame      →  FeatureStore.set(name, buf)         (per ~47Hz worklet message)

render loop tick →  modulator.tick()
                       for each (key, {source, depth}):
                         src = MOD_SOURCES[source].read( FeatureStore.get(buf) )
                         src = isNaN(src) ? 0 : src
                         base = store.get(key)
                         { min, max } = store.schemaFor(key)
                         eff = base + (lerp(min, max, src) - base) * depth
                         store.notify(key, eff)          ← fires existing subscribers
                                                              ↓
                                                  camera.fov / uniform / etc

user slider     →  store.set(key, base)                 (persists, fires subscribers with base)
                                                         (next frame, modulator re-fires with eff)

remove binding  →  modulator.setBinding(key, null)
                     ↳ deletes the entry
                     ↳ one-shot notify(key, base) so consumers snap back to slider position
```

`ParamStore.schemaFor(key)` is implicitly already available via the private `schemas` map; expose a public getter:

```ts
schemaFor(key: string): ParamSchema | undefined { return this.schemas.get(key); }
```

## Persistence

Bindings live in their own localStorage key, separate from param values:

- **Key:** `autocorrelation.modulation.v1`
- **Shape:** `{ [paramKey: string]: { source: string; depth: number } }`
- **Write:** `Modulator.setBinding` writes after every change (cheap; localStorage is fast for this size).
- **Read:** `Modulator` constructor loads once. Bindings whose `source` is no longer in `MOD_SOURCES`, or whose `paramKey` doesn't have a schema, are dropped silently.
- **Reset:** A small "Reset modulation" button next to the existing "Reset to defaults" button. Clears the localStorage key and all in-memory bindings, then for each previously bound key fires `notify(key, base)` once.

## Lifecycle / HMR

- `Modulator` is constructed in `App.start()` alongside `ComponentManager`/`PostStack`, after `ParamStore` and `FeatureStore` exist.
- `Modulator.tick()` is called from the existing RAF loop in `App.start()`, immediately before `components.update()` so consumers see the modulated value on the same frame.
- `App.dispose()` calls `modulator.dispose()`, which clears subscribers but leaves localStorage intact.
- On HMR, the new `App.start()` instantiates a fresh `Modulator`, which re-loads bindings from localStorage. No round-trip needed.

## Failure Modes & Non-Obvious Invariants

- **NaN source values → 0.** Beat outputs are NaN in silence; the modulator collapses to `effective = base` rather than propagating NaN into uniforms.
- **Empty source buffer → 0.** Before the first features message arrives, `FeatureStore.get` returns the shared empty `Float32Array`. `latest()` returns 0; modulation contributes nothing.
- **Slider shows base, scene shows effective.** Intentional. The slider is the user's resting value; the audio drives behavior on top. A user moving the slider during active modulation sets base; next frame the modulator overrides with the new effective.
- **Unknown source string in persisted binding → dropped.** Could happen after a `MOD_SOURCES` table rename. Don't crash, don't log noisily — just drop on load.
- **Unknown param key in persisted binding → dropped.** Same reasoning.
- **`dsp.*` params are filtered at the helper.** `bindParam` checks `schema.key.startsWith("dsp.")` and skips the sub-folder. Defense in depth: if someone hand-writes a persisted binding for a `dsp.*` key, `tick()` still processes it (no second filter). Acceptable — there's no UI path to create it, and the consequences are recoverable (clear the localStorage key).
- **`notify()` callers don't update `this.values`.** Subscribers that read `store.get(key)` from inside their own callback will see the base value, NOT the effective value they were just called with. The only existing subscriber that does this is `WorkletBridge.handleChange` — and it only subscribes to `dsp.*` keys, which are never notified by the modulator. Safe.

## Out of Scope

- **Discrete or boolean param modulation.** Continuous only. A "step modulation" for discrete params (e.g. cycle a preset on each beat) could be a future addition.
- **Modulating dsp.\* params.** Feedback loop risk; excluded.
- **Multiple sources per param.** One source per param. A future "modulation matrix" / additive combiner could come later.
- **Source shaping** (curve, smoothing τ, attack/release envelope on the source side). Sources are consumed raw. If a beat saw is too snappy, smoothing belongs in the DSP crate, not the modulator.
- **Per-source inversion.** Considered and declined — depth could be made signed instead if asked for later, but keeping it unipolar 0..1 matches the "depth = how much audio takes over" mental model.
- **Modulation of the modulation** (LFO-on-amount, etc.). Not now.

## File Touch List

New:
- `src/params/Modulator.ts`
- `src/params/bindParam.ts`
- `tests/params/Modulator.test.ts`

Edited:
- `src/params/ParamStore.ts` — add `notify()` and `schemaFor()`.
- `src/params/ParamPanel.ts` — refactor `addWidget` call-site to use `bindParam`; add "Reset modulation" button.
- `src/App.ts` — construct `Modulator`, call `modulator.tick()` in RAF loop, pass to consumers' `bindUI`, dispose on teardown. Migrate camera/light bindings to `bindParam`.
- `src/render/post/PostStack.ts` — accept `modulator` in `bindUI`, migrate post-effect param bindings to `bindParam`.
- `src/render/components/ComponentManager.ts` — accept `modulator` in `bindUI`, migrate component param bindings to `bindParam`.

## Testing

`tests/params/Modulator.test.ts`:
- `tick()` with no bindings fires nothing.
- `tick()` with depth=0 fires `notify` with `base`.
- `tick()` with depth=1 fires `notify` with `lerp(min, max, src)`.
- `tick()` with NaN source fires `notify` with `base`.
- `tick()` with empty source buffer fires `notify` with `base`.
- `setBinding(key, null)` fires one `notify(key, base)` then no more on subsequent ticks.
- `setBinding` persists; new `Modulator` instance on same localStorage reads it back.
- Persisted binding with unknown source key is dropped on load.

No DSP-side or wasm-side changes; existing tests stay green.
