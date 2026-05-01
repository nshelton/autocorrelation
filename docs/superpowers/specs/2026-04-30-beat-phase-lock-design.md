# Beat Phase Lock-In: Top-K φ Candidates + Faster PLL

**Status:** approved, ready for plan
**Owner:** nick
**Affects:** `crates/dsp/src/beat.rs`

## Problem

The smoothed beat phase (`phase_smoothed` in `BeatState`) jiggles visibly in the
beat-grid renderer when the source contains content that is not part of the
groove (vocals, sustained chords, fills). On clean drum loops the grid is
stable. BPM (`tau_smoothed`) is unaffected — the issue is purely the φ argmax
returned by `score_phase_for_tau` wandering by a handful of hops between
frames as non-rhythmic onsets move which φ wins.

The PLL on top of the observation runs at α=0.01, slow enough to mask sample
noise but not slow enough to reject these multi-hop excursions cleanly. It
slowly drags `phase_smoothed` off the true beat then back, which the user sees
as "sliding."

## Goal

Make `phase_smoothed` lock harder to the true beat phase on imperfect material,
without sacrificing reactivity to genuine tempo/phase changes.

Non-goals: downbeat detection, comb history weighting, sub-hop φ resolution.
Those remain on the roadmap; this spec narrowly fixes the wandering-φ problem.

## Approach

Two changes, both in `beat.rs`:

### 1. Replace integer argmax with top-K local maxima + snap-to-prediction

Today `score_phase_for_tau` returns the single best φ. Replace with a top-K
list of *local maxima* of the per-φ correlation curve. The PLL picks the
candidate closest to its predicted phase, falling back to the global max only
when it has no prediction yet (cold start).

**`score_phase_for_tau` changes:**

- Compute the per-φ correlation array as today (Φ₁ + Φ₂ + Φ₁.₅, N=4 pulses).
- Walk the array; a φ is a local maximum iff `corr[φ] > corr[φ-1]` and
  `corr[φ] > corr[φ+1]` (and `> 0`). Endpoints are not eligible (matches the
  existing peak-picker convention in `pick_candidates`).
- Insertion-sort into a fixed-size top-K buffer by `corr` descending.
- Return type:
  `([(usize, f32); PHI_CANDIDATE_COUNT], f32, f32, usize)` —
  the top-K array followed by `(sum, sum_sq, n_phases)` (same as the existing
  trailing triple). Drops the separate `best_phi`/`best_corr` returns; callers
  read `candidates[0]`. Empty slots have `corr == 0.0`; `phi` in empty slots
  is unspecified (callers iterate until `corr == 0.0`).

`PHI_CANDIDATE_COUNT = 5`.

No allocation. The top-K buffer is a stack array. Caller-side iteration is
`for (phi, corr) in &candidates { if corr <= 0.0 { break; } ... }`.

### 2. PLL picks nearest candidate above a relative-correlation floor

In `update_tea`, after computing `tau` and `expected_phase`:

- If `phase_smoothed` is not initialized — current condition is
  `!self.phase_smoothed.is_finite() || self.phase_updated_at == 0` — pick the
  top-K entry with highest `corr` (i.e. the global max — current behavior).
- Otherwise: among the K candidates, keep only those with
  `corr >= PHI_CANDIDATE_MIN_RATIO * top_corr`. From the survivors, pick the
  one minimizing `|signed_phase_delta(expected_phase, candidate_phi, τ)|`.

`PHI_CANDIDATE_MIN_RATIO = 0.7`. This stops a tiny secondary peak from winning
just because it happens to sit closer to the prediction. The dominant peak
gets to override the prediction if it's much stronger.

The chosen φ becomes `observed_phase`. The existing predict-correct step is
unchanged in shape:

```rust
let correction = PHASE_CORRECTION_ALPHA
    * signed_phase_delta(expected_phase, observed_phase, tau);
self.phase_smoothed = wrap_phase(expected_phase + correction, tau);
```

### 3. Bump α and drop dead clamp

- `PHASE_CORRECTION_ALPHA`: **0.01 → 0.05** (time constant ≈ 20 frames ≈
  0.4 s at 47 Hz). With top-K snap doing the heavy lifting on disambiguation,
  the PLL only needs to absorb sub-hop noise. A faster α makes the grid feel
  responsive when it does need to move.
- `PHASE_CORRECTION_MAX_HOPS = 20.0`: dead code with α=0.01 and `signed_phase_delta`
  bounded by ±τ/2. Delete the constant and the `.clamp()` call.

### 4. `score_beat_hypotheses` consumer

`score_beat_hypotheses` already calls `score_phase_for_tau` per beat hypothesis
and uses only `best_phi`, `best_corr`, `sum`, `sum_sq`. Update the call site
to read `candidates[0].0` (top φ) and `candidates[0].1` (top corr) into the
existing `pulse_phi[i]` and `pulse_x[i]` slots. Per-hypothesis scoring is
unchanged.

## Constants summary

| Name | Value | Where |
|---|---|---|
| `PHI_CANDIDATE_COUNT` | 5 | new |
| `PHI_CANDIDATE_MIN_RATIO` | 0.7 | new |
| `PHASE_CORRECTION_ALPHA` | 0.01 → **0.05** | bumped |
| `PHASE_CORRECTION_MAX_HOPS` | — | **deleted** |

## Test plan

- Unit test: synthesize an `onset[]` with a clean periodic kick at τ=20 plus
  one off-beat distractor at τ/3. Verify `score_phase_for_tau` returns the
  kick φ in slot 0 and the distractor φ in a later slot.
- Unit test: same synthetic input, run two consecutive frames. Verify
  `phase_smoothed` after the second frame snaps to the kick φ even when the
  distractor's correlation grows past the kick's on the second frame (top-K
  rescue).
- Integration test (existing `lib.rs` beat tests): re-run; expect no behavior
  change on the clean periodic test cases — the new code path with cold start
  reduces to the old behavior.
- Manual: run `npm run dev`, play a track with non-drum content (the
  problematic material the user already has loaded). Visually confirm
  `BeatGridMarkers` lines stop sliding on sustained chords / fills.

## Out of scope

- Downbeat detection (φ ↔ φ + N·τ disambiguation)
- Sub-hop φ refinement (parabolic interp on the φ correlation)
- Comb-history exponential decay
- Confidence-gated PLL (option (b)/(c) from the brainstorming session)

These remain in `ROADMAP.md` candidates if the symptom persists after this
change.
