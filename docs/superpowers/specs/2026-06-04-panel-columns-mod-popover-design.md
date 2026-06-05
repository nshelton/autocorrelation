# Multi-column tweak panel + inline mod button → floating popover

**Date:** 2026-06-04
**Status:** Approved

## Goal

The single tweakpane column gets too long. Two changes:
1. Split the panel into **top-anchored columns** (one pane per section, flex-wrap).
2. Replace each modulatable param's always-on `↳ mod` sub-folder (a second row)
   with a **small inline button** on the right of the slider that opens a
   **floating popover** holding the mod controls. Triggers (`↳ trig`) get the
   same treatment.

## Decisions

- Columns: one pane per section (Analysis, Scenes, Camera, Post), flex-wrap,
  each column `max-height: 100vh; overflow-y: auto` (independent scroll).
- Mod reveal: inline button → single shared floating popover (one open at a
  time), closed by re-click / click-outside.
- Triggers also convert to the popover.
- Global Reset buttons move to the bottom of the Analysis column.

## Architecture

### `ModPopover` (new, src/params/ModPopover.ts) — module singleton

Like `shTween`: `export const modPopover = new ModPopover()`. bindParam /
bindTrigger import it directly, so **no signature threading** through
App/PostStack/main.ts.

- Owns a floating `<div>` (appended to body, `position: fixed; z-index` above
  panes, hidden by default) + one `Pane` inside it.
- `toggle(anchor: HTMLElement, build: (pane: Pane) => void)`: if already open
  for `anchor` → close; else clear pane, run `build`, show, position next to
  `anchor` (right-aligned under it, clamped to viewport).
- `close()`: hide + dispose pane children.
- A capture-phase `pointerdown` listener on document closes the popover unless
  the target is inside the popover or is the current anchor (so the anchor's
  own click toggles cleanly).
- Singleton persists for the page lifetime; `ParamPanel.dispose()` calls
  `modPopover.close()` on teardown. The `build` closure captures the live
  store/modulator from the current panel build, so HMR stays correct.

### `bindParam` (src/params/bindParam.ts)

- Extract the current mod-control building into
  `buildModControls(folder, store, modulator, key)` (source / depth / power /
  smoothing / level graph + wiring). Reused as the popover `build`.
- Replace the inline `↳ mod` folder with an injected button:
  - `binding.element.style.position = 'relative'`; append a `.mod-btn` button
    abs-positioned at the right; pad the value cell (`.tp-lblv_v`, guarded) so
    the slider doesn't underlap.
  - Click → `modPopover.toggle(btn, (pane) => buildModControls(pane.addFolder(
    {title: schema.label, expanded: true}), store, modulator, schema.key))`.
  - Button gets an `.active` tint when `modulator.getBinding(key)` is set,
    refreshed via `modulator.subscribe`.
- The live-value slider refresh + mid-drag `interacting` guard are unchanged.

### `bindTrigger` (src/params/bindParam.ts)

- Signature changes to take the host `ButtonApi` (so it can inject into the
  button's row element) instead of a folder.
- Extract `buildTriggerControls(folder, modulator, triggerKey)` (source +
  threshold + raw-source graph).
- Inject a `.mod-btn` button into the button row; click opens the popover with
  `buildTriggerControls`; `.active` tint via `getTrigger`.

### `ComponentManager.bindUI`

- `bindTrigger(b, modulator, triggerKey)` — pass the `ButtonApi` instead of the
  folder.

### `ParamPanel` (src/params/ParamPanel.ts)

- Build a top-anchored flex-wrap container `<div>` (`pointer-events: none`),
  appended to body; each column `<div>` is `pointer-events: auto`,
  `max-height: 100vh; overflow-y: auto`.
- One `Pane` per column in its own `<div>`. Sections Analysis / Scenes /
  Camera / Post each become a titled folder inside their pane, so the public
  `scenes` / `camera` / `post` stay `FolderApi` and **main.ts is unchanged**.
- Move the two global Reset buttons to the bottom of the Analysis column.
- `dispose()` disposes all panes, removes the container, and calls
  `modPopover.close()`.

### CSS

A one-time injected `<style>` for `.mod-btn` (small square, right-aligned,
dimmed) and `.mod-btn.active` (tinted). Column/container layout via inline
styles in ParamPanel.

## Testing

Modulator logic is unchanged → existing unit tests still cover behavior. This
change is DOM/visual; **verification is manual** (`npm run dev`): columns
wrap/scroll, mod button opens/positions/closes the popover, click-outside
closes, active params tint, modulation works through the popover, triggers work
through their button. No brittle tweakpane-DOM tests added.

## Risks

- Popover positioning + click-outside + z-index above panes.
- Inline button relies on tweakpane 4.0.5 row internals (`binding.element`,
  `.tp-lblv_v`) — pinned version, guarded querySelector.

## Out of scope

- Persisting column positions / collapse state beyond tweakpane defaults.
- Custom tweakpane plugin/blade for a truly native inline button.
