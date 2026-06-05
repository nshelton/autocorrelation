# Audio-triggered buttons + per-binding source monitor

**Date:** 2026-06-04
**Status:** Approved

## Goal

Let component action buttons (e.g. "Randomize SH") fire from the audio
modulation system the same way continuous params do — but instead of a `depth`
that continuously lerps a value, a button uses a `threshold`: when the selected
source rises across the threshold, the button's action fires once. Also add a
live 0–1 debug graph of the selected source to every modulation UI so the
threshold/depth is easy to tune against the audio.

## Decisions (from brainstorming)

- **Scope:** component-declared `paramButtons` only (currently "Randomize SH").
  The housekeeping Reset buttons stay plain.
- **Debug bar:** a Tweakpane readonly **graph** monitor (`view: 'graph'`, 0–1).
  Idiomatic and built-in; shows recent history. A literal single-fill bar would
  need a custom DOM widget and is out of scope for now.
- **Monitor reach:** added to BOTH the existing continuous `↳ mod` folders and
  the new button `↳ trig` folders.
- **Visibility:** the monitor is a child of the `↳ mod`/`↳ trig` sub-folder,
  which is collapsed (`expanded: false`) by default — only visible when the
  user expands that sub-folder.

## Trigger semantics

Rising-edge, single-fire, self-re-arming:

- Per trigger key, track an `armed` boolean.
- On first sample: set `armed = (v < threshold)` (no fire on load).
- When `armed && v >= threshold`: fire the callback, set `armed = false`.
- When `!armed && v < threshold`: set `armed = true` (re-arm).
- NaN / silence reads as 0 (via `readSource`), which keeps it armed.

## Architecture

### `Modulator` (src/params/Modulator.ts)

Triggers reuse the existing source-reading, per-frame tick, persistence, and
UI-sync infrastructure, so they live in `Modulator` rather than a parallel
class.

New state:
- `interface TriggerBinding { source: string; threshold: number }`
- `triggers: Map<string, TriggerBinding>` — persisted.
- `triggerCallbacks: Map<string, () => void>` — attached by the UI at bind
  time; NOT persisted (functions can't serialize), reattached on every
  rebuild/HMR.
- `triggerArmed: Map<string, boolean>` — edge-detection state.

New methods:
- `setTrigger(key, TriggerBinding | null)` — set/delete, persist, notify UI.
- `getTrigger(key): TriggerBinding | null`.
- `registerTriggerCallback(key, fn)` — UI attaches the action.
- `readSource(source): number` — finite source read (NaN→0); used by both the
  trigger tick and the monitor getter.

`tick()` gains a second loop after the continuous bindings that evaluates each
trigger with a registered callback using the edge logic above.

Persistence: triggers under a **separate** localStorage key
(`autocorrelation.triggers.v1`) so the existing
`autocorrelation.modulation.v1` (continuous bindings) needs no migration.
`dispose()` clears `triggerCallbacks` (stale closures); `triggers` data is kept
and reloaded by the next constructor like continuous bindings.

### UI helpers (src/params/bindParam.ts)

- `addSourceMonitor(folder, modulator, getSource)` — adds a readonly graph
  binding bound to `{ get value() { return modulator.readSource(getSource()) } }`,
  `view: 'graph'`, min 0, max 1, `interval ≈ 33`. Tweakpane polls the getter;
  no per-frame wiring, nothing to dispose.
- `bindTrigger(folder, modulator, triggerKey)` — adds a `↳ trig` sub-folder
  (collapsed) with: source dropdown (incl. "none" sentinel → `setTrigger(null)`),
  threshold slider (0–1, step 0.01), and `addSourceMonitor`. Seeds from
  `getTrigger`, writes via `setTrigger`, two-way syncs via `modulator.subscribe`.
- The existing continuous `↳ mod` folder also calls `addSourceMonitor` (reads
  the live `modProxy.source`).

### Button wiring (src/render/components/ComponentManager.ts)

For each `paramButton`:
- `fire = () => { btn.onClick(paramStore); folder.refresh(); }` wired to both
  the manual button click and `modulator.registerTriggerCallback(triggerKey, fire)`.
- `triggerKey = ${prefix}.button.${slug(btn.title)}` where slug strips
  whitespace (e.g. `orbitalCloud.button.RandomizeSH`).
- `bindTrigger(folder, modulator, triggerKey)`.

`bindUI` already receives the `Modulator`.

## Tests (tests/params/Modulator.test.ts)

- Rising edge fires the callback exactly once.
- Holding above threshold does NOT re-fire (stays disarmed).
- Falling below then rising again re-fires.
- No fire on the first sample when the source starts above threshold.
- `readSource` returns finite values and maps NaN/none → 0.

## Out of scope

- Custom single-fill bar widget (graph only for now).
- Triggers on Reset buttons.
- Trigger persistence surviving a button title rename.
