# SEQUENCER.md

Architecture of **synth mode** — the synthesizer / sequencer / mini-DAW. This is
the *inverse* of the audio-analysis pipeline documented in
[CLAUDE.md](CLAUDE.md): instead of `audio in → features → visuals`, it is
`score in → synthesis → audio out` (plus a custom DOM/Canvas2D editor UI).

## Two modes, one app

The app boots into **synth mode by default**. Append `?mode=analysis` to the URL
to get the original audio-analysis visualizer (mic/tab → DSP → Three.js). The
two share almost no runtime state:

- **Analysis mode** uses the `Dsp` wasm export, the `dsp-worklet`, the Three.js
  `Scene`/`App`/`ParamStore`/`WorkletBridge` — all in CLAUDE.md.
- **Synth mode** uses the `Sequencer` wasm export, the `synth-worklet`, and a
  self-contained custom UI under `src/sequencer/`. It does **not** touch the
  analysis App, Scene, or ParamStore.

Both wasm structs (`Dsp` and `Sequencer`) come from the **same crate** (`crates/dsp`)
and the same `npm run wasm` build — they're just two `#[wasm_bindgen]` surfaces.

Boot lives in [`src/main.ts`](src/main.ts): `onStartSynth()` is the synth path;
`?mode=analysis` falls back to the original source-chooser. Because synth mode can
auto-boot without a click, its `AudioContext` may start *suspended* under the
autoplay policy — the worklet is built anyway and resumes on the first user
gesture (pointer/key).

## Signal flow

```
                         main thread                         │   audio thread (wasm)
                                                             │
 SequencerLayout / Arrangement /     ──messages──▶  synth-worklet.ts  ──▶  Sequencer
 PianoRoll / DrumMachine / SynthPanel   (port)     (AudioWorkletProcessor)   (Rust)
  (custom DOM + Canvas2D UI)
        ▲                                                    │                 │
        │   playhead {beat, playing}  ◀──────post (~43 Hz)───┘          Vec<PolySynth>
        │                                                    │          (one per track)
   persistence.ts (localStorage)                             ▼                 │
                                                        destination  ◀──mix + tanh──┘
```

`SynthWorklet.ts` (main thread) builds an `AudioWorkletNode` with
`numberOfInputs: 0, numberOfOutputs: 1` and connects it straight to
`context.destination` — the mirror of the analysis node (1 in / 0 out).

## Rust DSP (`crates/dsp/src/synth.rs` + `sequencer.rs`)

### `synth.rs` — the instrument

- **`Voice`** — one sounding note: oscillator → Cytomic TPT state-variable
  lowpass filter → linear ADSR amp envelope. The oscillator is chosen by an
  `Engine`:
  - `Engine::Subtractive` — naive sawtooth (`2·phase − 1`); aliases above a few
    kHz (PolyBLEP is a future refinement).
  - `Engine::Simplex` — single-cycle **wavetable** lookup (linear-interpolated).
- **`PolySynth`** — a fixed pool of `MAX_VOICES = 16` voices + a note-stealing
  allocator (prefer an idle voice, else steal the quietest). Holds the `engine`
  and the shared `wavetable`. `render_add(out)` sums active voices **additively**
  (no clear, no clip) so the sequencer can mix many instruments under one master
  soft-clip. `process()` (clear + render + `tanh`) is **`#[cfg(test)]`-only** —
  the realtime path uses `render_add`.
- Params (per voice, set across all of a `PolySynth` via `set_param`) — the Rust
  keys the UI mirrors (and `INSTRUMENT_DEFAULTS` must match the `Voice` defaults):
  - **Osc tuning / unison:** `octave`, `semi`, `fine` (cents) shift every note's
    pitch; `detune` (cents) adds a second detuned oscillator for a fatter unison.
  - **Filter:** `cutoff`, `resonance`.
  - **Filter envelope:** `filterEnvAmount` (signed; scales cutoff by up to ±6
    octaves at full envelope) + its own ADSR `fAttack`/`fDecay`/`fSustain`/`fRelease`.
  - **Amp envelope:** `attack`, `decay`, `sustain`, `release`.
  - **LFO:** `lfoRate` (Hz), `lfoDepth`, `lfoTarget` (0 = pitch, 1 = cutoff,
    2 = amp), `lfoShape` (0 = sine, 1 = triangle, 2 = square, 3 = saw). Per-voice,
    retriggered at note-on.
  - **Drive:** `drive` (waveshaper amount) + `driveMode` (0 = pre-filter,
    1 = post-filter).
  - **Output:** `gain`.
  The added params all default to **inert** values (offsets/amounts/depths/drive =
  0), so a default voice is exactly the original saw → filter → amp-ADSR and skips
  the per-sample modulation work (detune adds a second oscillator, the filter-env
  and LFO-cutoff paths recompute SVF coefficients per sample, etc.) until used.
  Discrete selectors (`lfoTarget`/`lfoShape`/`driveMode`) ride the same `set_param`
  float path — rounded to an index in Rust — so no new message type is needed.

### `drums.rs` — the drum kit (ReDrum-style)

- **`DrumKit`** — a fixed bank of one `DrumVoice` per lane (kick, snare, closed/
  open hat, clap, rim, three toms, cymbal), addressed by **MIDI note**: the step
  grid emits a note at the lane's MIDI number and `note_on` routes it back to its
  lane. Lane→MIDI mapping **MUST agree with `src/sequencer/drumkit.ts`** (the drum
  analogue of how `INSTRUMENT_DEFAULTS` mirrors the `Voice` defaults).
- **`DrumVoice`** — synthesized (no samples): a sine "tone" component with an
  optional fast pitch sweep (kick/toms) + a high-passed white-noise component
  (snare/hats/clap/cymbal), each with its own exponential decay. One-shots:
  `note_on` retriggers, **`note_off` is ignored** (decay mode — a hit always
  rings its full envelope regardless of note length). Voices below ~−60 dBFS
  deactivate. Closed/open hat share a **choke group** (closed cuts open).
- Like `PolySynth`, exposes `note_on` / `note_off` / `release_all` / `set_param`
  (only `gain`) / `render_add` so the sequencer treats both uniformly via
  `TrackInstrument` (below). Deterministic xorshift noise (per-lane seed) avoids
  pulling wasm RNG into the build.

### `TrackInstrument` (sequencer.rs) — synth vs drums

`Sequencer::synths` is `Vec<TrackInstrument>`, an enum of `Poly(PolySynth)` |
`Drum(DrumKit)`. Every per-track call routes through it, so the transport /
scheduler / `reconcile` stay instrument-agnostic — **a drum pattern is just
notes** routed to a `Drum` track; nothing else in the pipeline changes.
`set_instrument_kind(track, kind)` (0 = synth, 1 = drums) rebuilds a track's
instrument; it's a guarded no-op when the kind is unchanged (so a redundant
resend doesn't cut a sounding voice). Engine/wavetable are synth-only (drums
ignore them).

### `sequencer.rs` — transport + multi-track scheduler (the wasm surface)

`Sequencer` is the `#[wasm_bindgen]` struct the worklet drives. It owns:

- `synths: Vec<PolySynth>` — **one instrument per track**, sized by
  `set_track_count`.
- `schedule: Vec<NoteEvent>` — flat, sorted note events. `NoteEvent { beat, midi,
  vel, track }`; `vel == 0` is a note-off.
- `active: Vec<Vec<i32>>` — per-track set of midis **currently sounding from the
  schedule** (not live keys). The basis of stuck-note prevention.
- A **beat-domain** playhead. `beats_per_sample = bpm / 60 / sample_rate`.
- Loop bounds (`loop_start`, `loop_end`, `loop_enabled`).

`process(out)` mixes the block:

1. `render_mix` = clear `out`, each `PolySynth.render_add(out)`, then a single
   master `tanh` soft-clip.
2. When playing, the block is **split at event boundaries** (sample-accurate
   dispatch): each scheduled event routes to `synths[ev.track]` at its exact
   sample offset, and updates `active[ev.track]`.
3. At the loop point the playhead wraps to `loop_start` and `reconcile()` runs.

**`reconcile()` — the stuck/missing-note fix.** Any time the playhead or schedule
jumps (`play`, `seek`, `set_schedule` while playing, loop-wrap), the on/off
pairing in the flat schedule can get out of sync — a deleted note's on already
fired but its off is gone (stuck), or playback starts mid-note so the on was
skipped (silent). `reconcile()` computes `schedule_state_at(playhead)` (the notes
that *should* be sounding) and: releases `active` notes not wanted, triggers
wanted notes not already sounding. **Live-keyboard voices aren't in `active`**, so
they're never disturbed by edits/seeks. This is the single place to hook any new
playhead/schedule discontinuity.

Why beats (not samples/seconds): the natural domain for MIDI + the piano-roll,
tempo-independent, and f32-precise (a 10-min song at 120 BPM is ~1200 beats).

## Worklet ↔ main message protocol (`synth-worklet.ts`)

Down (main → worklet), all forwarded to the `Sequencer`:

| message | → Rust |
|---|---|
| `{ trackCount, count }` | `set_track_count` — **send before schedule/instruments** |
| `{ tempo, bpm }` | `set_tempo` |
| `{ transport, action: "play"\|"pause"\|"seek", beat? }` | `play` / `pause` / `seek` |
| `{ schedule, events: Float32Array }` | `set_schedule` — **stride-4** `[beat, midi, vel, track]` |
| `{ loop, start, end, enabled }` | `set_loop` |
| `{ param, track, key, value }` | `set_param` |
| `{ engine, track, index }` | `set_engine` (0 = subtractive, 1 = simplex) |
| `{ kind, track, kind }` | `set_instrument_kind` (0 = synth, 1 = drum kit) — **send before** engine/params, it rebuilds the instrument |
| `{ wavetable, track, data: Float32Array }` | `set_wavetable` |
| `{ noteOn\|noteOff, track, midi, velocity? }` | live keys → `note_on` / `note_off` |

Up (worklet → main): `{ type: "playhead", beat, playing }`, posted every 16
render blocks (~43 Hz — the UI playhead/position readout, not every block).

Messages that arrive before wasm finishes booting are queued and replayed in
order (note triggers aren't dropped).

## Data model (`src/sequencer/model.ts`)

The main-thread source of truth. Times are in **beats**.

```ts
Note  { start, duration, midi, velocity }   // start relative to its clip
Clip  { id, start, length, notes: Note[] }  // placed on the timeline
Instrument { kind?: "synth"|"drums", engine, params }  // kind defaults "synth"
Track { id, name, color, instrument, clips: Clip[], view? }  // view = saved PR zoom/scroll
Project {
  bpm,
  loopStart, loopEnd,          // the VIEW window + base loop (4/8/16/32 buttons)
  loopEnabled,                 // gold loop toggle
  loopRegionStart, loopRegionEnd,  // the draggable sub-loop (when enabled)
  tracks: Track[],
}
```

`flattenToSchedule(project)` → the stride-4 `Float32Array` the `Sequencer` wants:
for each track index, each clip, each note → an on event at `clip.start +
note.start` and an off at `+ duration`, tagged with the track index, sorted by
beat (note-off before note-on on a tie so a re-struck pitch releases first).

`INSTRUMENT_DEFAULTS` is the single source of synth-param defaults, shared with
the panel; it mirrors the Rust `Voice` defaults so UI and DSP start in agreement.

**Loop fields are overloaded by name:** `loopStart`/`loopEnd` are really the
*view window* (and the loop when the region is off); `loopRegionStart`/`End` are
the gold sub-loop. `main.ts` picks the active range (`sendLoop`).

## Tuning (`src/sequencer/tuning.ts`)

Pitch → frequency is pure 12-TET in the DSP (`midi_to_hz` = `440·2^((m−69)/12)`),
and the whole pitch path is **f32**, so a tuning system is just a **remap of the
base pitch applied at flatten time** — no DSP change. `tuneMidi(midi, tuning)`:

- **`equal`** — identity (12-tone equal temperament; the default).
- **`just`** — 5-limit just intonation relative to a movable **root** (tonic): a
  fixed 12-entry cents table indexed by `(round(midi) − root) mod 12`, added to
  the ET pitch (cents ÷ 100 → fractional semitones). Pure intervals from the root
  (fifth +2¢, major third −14¢); changing `root` re-derives every interval.

`Project.tuning = { mode, root }` (persisted; `normalize` backfills `equal`).
Applied in **`flattenToSchedule` + `flattenBends`** (the bend base is tuned too,
so a bent note glides from its tuned start) and to **live keys** in `main.ts`.
**Drum tracks are skipped** — their MIDI is a lane index, not a pitch. The
**TUNE** group in the transport bar toggles JI and cycles the root; a change
re-flattens the schedule, so it takes effect on the next note / loop wrap (held
notes aren't retuned mid-sound). The editor still stores/show integer semitones —
only the *sounding* pitch is tuned. Caveat: fixed-root JI has a "wolf" interval or
two per key (e.g. C → the D–A fifth is ~22¢ flat); that's expected, not a bug.

## MIDI import (`src/sequencer/midi.ts`)

`parseMidi(ArrayBuffer)` uses `@tonejs/midi` (lazy-imported, so it's a separate
bundle chunk loaded only on first drag-drop). Note positions convert ticks →
beats via `ticks / ppq` (tempo-independent). Each MIDI track becomes a `Track`
with one `Clip` and a default `Instrument`. Limitations: only the first tempo is
used; multi-track files play in full but every track defaults to the same timbre
until tweaked.

## Frontend UI (`src/sequencer/`, all custom — no UI framework)

A DAW shell glued by `main.ts`: a global transport bar, then three resizable
stacked panes — **arrangement** (top), **editor** (middle), **params** (bottom).

- **`SequencerLayout.ts`** (the shell) — builds the **global transport bar**
  (play/stop, `bar.beat` readout, draggable **BPM**, **LOOP** length 4/8/16/32,
  gold **⟳** toggle, **TUNE** = JI toggle + root, **+ Synth / + Drums**) and the
  arrangement + params pane
  containers, with two draggable **splitters** between the three regions. It is
  the **sole publisher** of `--track-bar-h` (= transport + arrangement + splitter)
  and `--synth-panel-h` (= params + splitter), repurposed as the **editor band's
  top/bottom insets**. Because `PianoRoll`/`DrumMachine` are `position:fixed` and
  already fill `[--track-bar-h, 100dvh − --synth-panel-h]`, a splitter drag just
  re-publishes the vars and they reflow — **no editor edits required**. Vars are
  set from explicit drag / `resize` math only (never a self-observing
  ResizeObserver), so there's no resize-feedback loop. Pane heights persist under
  the separate `autocorrelation.ui` localStorage key (clamped so the editor band
  never collapses below a minimum). Shows a "select a clip" placeholder over the
  editor band, and a params placeholder, when nothing is selected / for drum tracks.
- **`Arrangement.ts`** (top pane, replaces the old `TrackBar`) — a Canvas2D
  timeline (same DPR + rAF + ResizeObserver pattern as `PianoRoll`) with one lane
  per track, a left header column (name + color + a **volume fader**), and each
  clip drawn as a block with a **mini thumbnail** (note dots for synth, step dots
  for drums via `DRUM_LANES`). Click a clip (or a track's header / empty lane) →
  `onSelectClip` → the host opens it in the editor + params. The header fader
  drag → `onVolume(track, 0..1)`, which sets the track's **instrument gain** (the
  per-track level — drums' only volume control; for synths it mirrors the panel's
  gain slider, which the host re-syncs). Read live each frame, so panel-side gain
  edits show up here with no wiring. Playhead line synced to transport.
- **`PianoRoll.ts`** (editor pane) — a Canvas2D overlay sized
  `calc(100dvh - var(--track-bar-h) - var(--synth-panel-h))`. Owns a lot:
  - **Note editing** of the *selected track's* clip (others render dimmed):
    click-drag to paint, drag body to move, drag right edge to resize,
    right-click to delete. 1/16 grid snap. Notes stored clip-relative.
  - **Pitch view** = center + octave count (pan ▲▼, zoom −/+, wheel to pan).
  - **Per-track view persistence** — each track remembers its own zoom/scroll in
    `Track.view`. Opening a track's clip restores it via `setView` (no saved view →
    auto-fit to that track's content); pan/zoom fires `onViewChange`, and `main.ts`
    stores `getView()` back onto the selected track (debounced save).
  - **Toolbar** (top-left): only the view controls — **OCT** (pitch pan ▼▲) and
    **ZOOM** (octave count −/+). Transport / BPM / loop length / the **⟳** toggle
    moved to the global transport bar. (The gold loop *region* is still dragged in
    the ruler — `onLoopChange` — and ruler scrub still seeks — `onSeek`.)
  - **Scrub** — drag the red playhead line or click/drag the top ruler band.
  - **Gold loop region** — when `loopEnabled`, a gold band with draggable edge
    handles (in the ruler); playback loops just that section.
  - Re-syncs its backing store via a `ResizeObserver` on the canvas (catches both
    window resize and the CSS-var height changes).
- **`DrumMachine.ts`** (replaces the piano roll for a **drum** track) — a custom
  DOM step grid: `DRUM_LANES.length` rows × `DRUM_STEPS` columns spanning the
  loop window (default 4-beat loop → classic 16-step bar). Click/drag to
  paint steps, right-click to clear; an **H/M/S level** selector sets the
  velocity each new step gets (the ReDrum 3-level dynamic), shown as cell
  brightness. Own compact transport toolbar (play/stop/BPM) and a playhead
  column highlight. Edits mutate the track's clip notes and report via
  `onNotesChange` — same path as the piano roll. Clicking a lane name auditions
  that drum. `drumkit.ts` holds the canonical lane→MIDI table + the H/M/S
  velocity scheme (shared with `model.ts` starter patterns), mirroring
  `drums.rs`. Transport/tempo moved to the global bar, so this toolbar now carries
  only **VEL** (H/M/S) + **clear**. **+ Synth / + Drums** track-add buttons live in
  the transport bar.
- **`SynthPanel.ts`** (params pane) — a *view of the selected track's instrument*.
  Engine selector (Subtractive/Simplex) + **labeled control groups** (OSC, FILTER,
  F.ENV, AMP, LFO, DRIVE, OUT) that wrap to fit the pane. Most rows are custom
  sliders (log-scaled cutoff / LFO rate, step-snapped octave/semi/fine/detune);
  the discrete params (LFO target/shape, drive mode) are **segmented selectors**
  whose chosen index is the stored value (mirroring `engine`), reported through the
  same `onParam(key, number)` callback so the worklet path is uniform. The whole
  set is driven from one `GROUPS`/`ParamSpec` table; `INSTRUMENT_DEFAULTS` is the
  single source of defaults (also used for double-click / "reset"). `setInstrument`
  loads a track's values; edits mutate that instrument in place and report via
  callbacks. The title cell is sticky so the engine/reset stay put while the groups
  scroll. **Mounts into the layout's params pane** (constructor `mount` arg) and
  fills it; the layout owns `--synth-panel-h` now (the panel no longer publishes it).

`wavetables.ts` generates the Simplex single-cycle table on the main thread
(2D simplex sampled around a circle → seamless, DC-removed, peak-normalized;
seeded `mulberry32` for stability) and ships it per-track.

## Persistence (`src/sequencer/persistence.ts`)

The **whole `Project`** (notes, per-track instruments, tempo, loop, +
selectedTrack) is one JSON blob in localStorage. `save()` in `main.ts` is
debounced (300 ms) and called on every mutation — note edits, slider/engine
changes, loop/tempo, track selection, MIDI import — plus a `pagehide` flush.
`loadState` validates shape + version and **`normalize`s** missing fields so
saves from before a field existed still load (preferred over a version bump that
would discard the user's set).

**View state** (the resizable pane heights) persists *separately* under
`autocorrelation.ui`, owned by `SequencerLayout` — kept out of the project blob
so the project schema/version logic stays untouched. Clamped to the viewport on
load so a save from a tall window can't break a short one.

## `main.ts` orchestration (synth path)

Holds `project` and `selectedTrack` and ties the panes together:

- `loadProject(p)` — pushes to the worklet in order: **`trackCount` first**, then
  tempo, per-track instruments (**kind** → engine + wavetable + params), schedule,
  loop, seek; then `setProject` on the views.
  loop, seek; then `setProject` on the views, then `deselect()` (empty editor +
  params until a clip is clicked).
- `addTrack(kind)` — append a synth/drum track (via the transport bar buttons),
  push it, and open it.
- `selectClip(track, clip)` — open a clip: **swaps the editor** (drum track →
  `DrumMachine`, else piano roll + synth panel) and points params at the track.
  `selectedClip` drives the view; `selectedTrack` stays the worklet-facing index
  (live keys, params). `deselect()` shows placeholders. One clip/track for now,
  so `clip` is always 0.
- `setLoopLength`/`toggleLoopRegion` (from the transport bar) mutate the loop,
  `sendLoop`, then `resyncViews()` (re-push the project to the editors, since loop
  length resizes their time window / the drum step span) and re-open the clip.
- `sendLoop(p)` — sends the active loop range (the gold region when enabled, else
  the whole view).
- Drag-drop a `.mid` file → `parseMidi` → `loadProject`.

## Key invariants / pitfalls

- **Schedule is stride-4** `[beat, midi, vel, track]` (the analysis `candidates`
  buffer is stride-3 — don't confuse them).
- **`reconcile()` is the one place** that fixes stuck/missing notes at playhead or
  schedule discontinuities. Hook any new jump there. Live keys are intentionally
  not tracked in `active`, so editing/seeking never cuts them.
- **`trackCount` must precede** schedule/instrument messages so events route to
  synths that exist; out-of-range track indices are guarded (ignored) defensively.
- **Everything is in beats**, tempo-independent. `beats_per_sample` converts.
- **Two wasm exports, one crate/build**: `Dsp` (analysis) and `Sequencer` (synth).
- **A drum track is just notes.** A `Drum` `TrackInstrument` + a step-grid editor;
  the schedule/transport/`reconcile`/persistence are unchanged. Drum lane→MIDI
  numbers are duplicated in `drums.rs` and `drumkit.ts` and **must stay in sync**.
- **Drums ignore `note_off`** (one-shot decay mode), so `reconcile` can't cut a
  drum voice — fine, decays are short; `release_all` (pause/stop) chokes them.
- **Tuning is a flatten-time pitch remap, not a DSP feature.** The synth is pure
  12-TET (`midi_to_hz`); JI is applied in `flattenToSchedule`/`flattenBends` (and
  live keys) by nudging the base MIDI a fraction of a semitone. Apply it in *all*
  pitch paths or notes won't pair / will glitch — and **never to drum tracks**.
- **`SequencerLayout` is the sole publisher of `--track-bar-h`/`--synth-panel-h`.**
  They mean the editor band's insets now, not a bar's height. Don't re-add a
  second publisher (the old `TrackBar` did) or the editor band breaks. Set them
  from explicit drag/`resize` math only — never a ResizeObserver on the layout's
  own panes (feedback loop).
- **Synth mode is decoupled** from the analysis App/Scene/ParamStore — keep it
  that way; its UI is plain DOM/Canvas2D for that reason.
- **`Voice::process` is `#[cfg(test)]`-only**; realtime mixing uses `render_add`.
- **Persistence: prefer `normalize` backfill over version bumps** so users don't
  lose saved sets when the model grows a field.

## Current state / what's next

The UI is now a **resizable 3-pane DAW shell** (`SequencerLayout`): global
transport bar · arrangement (`Arrangement`) · editor (`PianoRoll`/`DrumMachine`) ·
params (`SynthPanel`). Clicking a clip opens it in the editor + params; panes
resize by dragging the splitters and the sizes persist.

Each track still holds **exactly one clip** (start 0), so the arrangement renders
clips and selects them but doesn't yet **move / add / duplicate / delete** them —
that's the next phase (it needs the editor to target an arbitrary clip, i.e. a
`PianoRoll` API change). Independent clips (copy = duplicate notes) was the chosen
model. (Transport is now solely in the global bar — the PianoRoll toolbar is just
OCT/ZOOM, the DrumMachine toolbar just VEL + clear — and the editors use `100dvh`.)
The synth voice grew **osc tuning/unison, a filter envelope, an LFO, and drive**
(see the `synth.rs` params list above), surfaced as grouped controls in
`SynthPanel`. Other open ideas: per-track distinct default timbres, a true
*second oscillator* (independent waveform/level, beyond the unison-detune pair),
a filter-env *velocity* mod, a free-running/tempo-synced LFO option, Simplex
wavetable morph (multi-frame), time-signature support.

The **drum machine** landed as a "core groovebox" (synth drums + 16-step grid +
H/M/S velocity, locked to the transport). Deferred ReDrum features: per-channel
tune / decay / pan / tone (pan needs a **stereo** rewrite of `Sequencer::process`
— currently mono), pattern length / resolution controls, shuffle/swing, flam,
and the pattern **bank** (A–D × 1–8) chained into a song.
