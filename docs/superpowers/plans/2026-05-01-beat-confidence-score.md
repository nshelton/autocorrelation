# Beat Confidence Score Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current relative beat `score_inst` output with a calibrated-ish `[0, 1]` beat confidence that consumers can gate on. `beatGrid[2]` and `beatState[1]` should mean “safe to trust the beat detector” rather than “best candidate relative to this frame’s candidates.”

**Architecture:** Keep the existing internal candidate ranking score for TEA voting, but stop exposing it as the public confidence. Add separate confidence components in `crates/dsp/src/beat.rs`: activity confidence, comb confidence, phase confidence, and TEA lock confidence. Combine them into an instantaneous confidence, smooth it, and write the smoothed confidence to `beatGrid[2]` / `beatState[1]`.

**Tech Stack:** Rust (`cargo test -p dsp`). Main implementation in `crates/dsp/src/beat.rs`; tests in `crates/dsp/src/lib.rs`.

---

## Current Problem

The current public score is derived in `score_beat_hypotheses` as an internal ranking value:

- `xn = pulse_x[i] / sum_x`
- `vn = pulse_v[i] / sum_v`
- `cn = beat_comb_score[i] / sum_comb`
- `pulse_score[i] = (xn + vn) * cn`
- `score_inst = max(pulse_score)`

That is useful for choosing which hypotheses vote into TEA, but it is not a stable confidence. It is relative to whatever candidates happen to exist in the current frame, so weak/noisy candidates can still produce a deceptively high winner.

---

## Desired Public Semantics

After this change:

- `beatGrid[0]`: smoothed beat period in hop/lag units.
- `beatGrid[1]`: smoothed phase in hop units since latest beat.
- `beatGrid[2]`: beat confidence in `[0, 1]`.
- `beatState[0]`: BPM derived from `beatGrid[0]`.
- `beatState[1]`: same beat confidence in `[0, 1]`.
- `beatState[2]` / `beatState[3]`: remain reserved for future meter confidence.

Suggested consumer thresholds after smoothing:

- Enable/use beat detector above `0.6`.
- Disable/ignore below `0.35`.
- Keep previous enabled state in between.

The Rust side should only produce confidence; hysteresis can live in TS consumers unless we later decide the DSP should export a binary lock state.

---

## Design Overview

Keep two different score concepts:

1. **Internal vote score**
   - Existing `pulse_score[i]`.
   - Relative candidate ranking.
   - Used only for picking `period_inst` and weighting top-N TEA votes.

2. **Public confidence**
   - New `confidence_inst` / `confidence_smoothed` fields.
   - Absolute-ish `[0, 1]` value.
   - Written to public buffers.

Compute public confidence from four factors:

1. **Activity confidence**: enough onset energy exists.
2. **Comb confidence**: selected period explains the full ACF peak lattice.
3. **Phase confidence**: selected period has a strong and distinct phase.
4. **TEA confidence**: tempo accumulator has a clear stable peak.

Use product/geometric combination so one bad component pulls confidence down.

---

## Task 1: Rename/Separate Internal Score From Public Confidence

**Files:**

- Modify: `crates/dsp/src/beat.rs`
- Test: `crates/dsp/src/lib.rs`

- [ ] Add fields to `BeatState`:
  - `vote_score_inst: f32` or keep `score_inst` temporarily for internal vote ranking.
  - `confidence_inst: f32`.
  - `confidence_smoothed: f32`.

- [ ] Prefer a small semantic refactor:
  - Rename `score_inst` to `vote_score_inst` internally if the edit is manageable.
  - If too invasive, keep `score_inst` for one commit but add comments that it is internal vote score, not public confidence.

- [ ] Update `write_beat_outputs` so public score slots will eventually come from `confidence_smoothed`, not `score_inst`.

- [ ] Keep `update_beat_pulses` using internal validity plus confidence:
  - Initial option: keep existing `score_inst > 0.0` behavior to avoid changing pulse behavior while confidence work lands.
  - Later option: require `confidence_smoothed > 0.35` if we want DSP-side suppression.

- [ ] Add/adjust tests asserting public confidence is within `[0, 1]` once Task 6 lands.

---

## Task 2: Add Activity Confidence

**Goal:** Silence/no onset content should force public confidence toward zero.

**Implementation idea:**

- Add helper `activity_confidence(onset: &[f32]) -> f32`.
- Compute a simple positive-energy statistic from the onset history:
  - mean positive onset
  - or RMS of onset
  - maybe ignore NaNs defensively

Suggested first formula:

- `onset_rms = sqrt(mean(onset[i]^2))`
- `activity_conf = onset_rms / (onset_rms + ACTIVITY_KNEE)`

Start with:

- `ACTIVITY_KNEE = 0.02` or `0.05`

Tune based on observed `onset` buffer magnitudes.

**Steps:**

- [ ] Add constants:
  - `ACTIVITY_KNEE: f32`

- [ ] Add helper:
  - `fn activity_confidence(onset: &[f32]) -> f32`

- [ ] Unit test:
  - silence returns near `0.0`.
  - a synthetic impulse train returns greater than silence.

---

## Task 3: Add Absolute Comb Confidence

**Goal:** The selected period should get high confidence if its multiples hit multiple raw ACF peaks, especially useful metrical multiples like `2τ`, `4τ`, and `8τ`.

Current `comb_score_for_tau` returns a useful ranking score but does not expose coverage details. Extend or add a second helper that returns a normalized confidence.

Suggested helper result:

- `comb_strength`: weighted matched peak strength mapped to `[0, 1]`.
- `comb_coverage`: weighted fraction of checked multiples that hit.
- `comb_conf = sqrt(comb_strength * comb_coverage)`.

Important: preserve the current `comb_score_for_tau` behavior for ranking unless tests show this new confidence can replace it safely.

**Steps:**

- [ ] Add small struct, private to `beat.rs`:
  - `CombEvidence { strength: f32, coverage: f32, confidence: f32 }`

- [ ] Add helper:
  - `fn comb_evidence_for_tau(&self, tau: f32, candidates: &[f32], raw_count: usize) -> CombEvidence`

- [ ] Reuse the same multiple weights and tolerances as `comb_score_for_tau`.

- [ ] Strength mapping proposal:
  - Accumulate weighted best hits exactly like the comb score.
  - Normalize by weighted possible mass.
  - Apply saturating knee: `strength = raw / (raw + COMB_CONF_KNEE)`.

- [ ] Coverage proposal:
  - `coverage = hit_weight / total_weight`.

- [ ] Confidence:
  - `confidence = sqrt(strength * coverage)`.

- [ ] Unit tests:
  - One isolated far peak produces lower confidence than a lattice with `τ`, `2τ`, `4τ`.
  - Strong `4τ` anchor plus weaker `τ`/`2τ` support gives meaningful confidence for `τ`.

---

## Task 4: Add Phase Confidence

**Goal:** A beat should be trusted only if the selected period has a clear phase, not just a broad/ambiguous phase response.

`score_phase_for_tau` already returns:

- top phase candidates
- `sum`
- `sum_sq`
- `n_phi`

Use those to compute absolute-ish phase confidence for the selected hypothesis.

Suggested components:

- `top_corr = phi_cands[0].1`
- `mean = sum / n_phi`
- `var = max(sum_sq / n_phi - mean * mean, 0)`
- `std = sqrt(var)`
- `z = (top_corr - mean) / (std + eps)`

Then:

- `phase_strength = top_corr / (top_corr + PHASE_STRENGTH_KNEE)`
- `phase_contrast = clamp((z - PHASE_Z_LOW) / (PHASE_Z_HIGH - PHASE_Z_LOW), 0, 1)`
- `phase_conf = phase_strength * phase_contrast`

Suggested initial constants:

- `PHASE_STRENGTH_KNEE = 0.1`
- `PHASE_Z_LOW = 1.0`
- `PHASE_Z_HIGH = 4.0`
- `PHASE_EPS = 1e-6`

**Steps:**

- [ ] Add helper:
  - `fn phase_confidence(top_corr: f32, sum: f32, sum_sq: f32, n_phi: usize) -> f32`

- [ ] Store per-hypothesis phase confidence in a new array:
  - `pulse_conf: [f32; MAX_PEAKS]`

- [ ] In `score_beat_hypotheses`, compute phase confidence for each hypothesis.

- [ ] Unit tests:
  - clear impulse train phase gives higher confidence than flat onset/noise.
  - ambiguous phase curve gives lower confidence than a single clear phase.

---

## Task 5: Add TEA Lock Confidence

**Goal:** Public confidence should rise only when the tempo accumulator has a clear stable winner.

After TEA update and argmax:

- `best_v`: TEA value at selected bin.
- Find `second_v`: strongest TEA bin outside a guard region around `best_i`.

Suggested guard:

- `guard = ceil(2 * tea_sigma)`

Suggested formulas:

- `peak_conf = clamp(best_v, 0, 1)` because TEA vote input is normalized to roughly `[0, 1]`.
- `margin = (best_v - second_v) / (best_v + eps)`.
- `margin_conf = clamp(margin, 0, 1)`.
- `tea_conf = sqrt(peak_conf * margin_conf)`.

**Steps:**

- [ ] Add helper:
  - `fn tea_lock_confidence(&self, tea: &[f32], best_i: usize, best_v: f32, upper: usize) -> f32`

- [ ] Use `beat_tau_min..=upper` as the TEA confidence search domain.

- [ ] Exclude bins within `guard` of `best_i` when finding `second_v`.

- [ ] Unit tests:
  - one clear TEA peak gives high confidence.
  - two equal separated TEA peaks gives low margin confidence.
  - all-zero TEA gives zero confidence.

---

## Task 6: Combine Components Into Public Confidence

**Goal:** Produce final `[0, 1]` public beat confidence.

Suggested combination:

- `confidence_inst = activity_conf * sqrt(comb_conf * phase_conf * tea_conf)`

Alternative weighted product if tuning needs more control:

- `confidence_inst = activity_conf * comb_conf.powf(0.35) * phase_conf.powf(0.35) * tea_conf.powf(0.30)`

Start simple with the square-root product.

Which candidate’s comb/phase confidence should be used?

- Use the hypothesis nearest `tau_smoothed` if available.
- Simpler first version: use the instantaneous best hypothesis selected in `score_beat_hypotheses`.
- If `tau_smoothed` differs meaningfully from `period_inst`, confidence should be reduced by an agreement factor.

Agreement factor proposal:

- `period_delta = abs(tau_smoothed - period_inst)`.
- `agreement = exp(-0.5 * (period_delta / tea_sigma)^2)`.
- `confidence_inst *= agreement`.

**Steps:**

- [ ] Add fields:
  - `comb_conf_inst: f32`
  - `phase_conf_inst: f32`
  - `tea_conf_inst: f32`
  - optional debug-only values if needed for tests.

- [ ] In `score_beat_hypotheses`, store selected hypothesis comb and phase confidence.

- [ ] In `update_tea`, compute TEA lock confidence after argmax.

- [ ] Add helper:
  - `fn update_public_confidence(&mut self, onset: &[f32], period_inst: f32)` or inline near end of `update_tea`.

- [ ] Clamp final value defensively:
  - `confidence_inst = confidence_inst.clamp(0.0, 1.0)`.

- [ ] Write `confidence_smoothed`, not `score_inst`, to public outputs.

---

## Task 7: Smooth Confidence

**Goal:** Avoid frame-to-frame flicker around threshold values.

Use asymmetric smoothing:

- faster attack
- slower release

Suggested constants:

- `CONFIDENCE_ATTACK_ALPHA = 0.2`
- `CONFIDENCE_RELEASE_ALPHA = 0.03`

Implementation:

- If `confidence_inst > confidence_smoothed`, use attack alpha.
- Else use release alpha.

Formula:

- `confidence_smoothed += alpha * (confidence_inst - confidence_smoothed)`

**Steps:**

- [ ] Add constants.

- [ ] Add field initialization in `BeatState::new`.

- [ ] Reset confidence to zero when there is no valid candidate/TEA peak.

- [ ] Test:
  - confidence rises after repeated periodic input.
  - confidence decays after silence.
  - confidence remains in `[0, 1]`.

---

## Task 8: Public Buffer Output Changes

**Files:**

- Modify: `crates/dsp/src/beat.rs`
- Maybe update: `src/render/DebugLabels.ts` copy if labels assume raw score semantics.

**Steps:**

- [ ] Change `write_beat_outputs`:
  - invalid/no confidence: `beatGrid[2] = 0.0`, `beatState[1] = 0.0`.
  - valid: write `confidence_smoothed` to both slots.

- [ ] Consider whether `update_beat_pulses` should use confidence:
  - Option A: keep pulses based on period validity and let renderer/consumer gate externally.
  - Option B: NaN-fill pulses below `confidence_smoothed < 0.35`.

Recommendation: start with Option A so consumers can choose thresholds.

- [ ] Update comments/docs in `CLAUDE.md` later if this behavior is kept.

---

## Task 9: Tests

Add tests in `crates/dsp/src/lib.rs`.

Required tests:

- [ ] `beat_confidence_silent_input_is_zero`
  - Process silence.
  - Assert `beatGrid[2] == 0.0` and `beatState[1] == 0.0`.

- [ ] `beat_confidence_periodic_input_rises`
  - Use existing periodic-envelope test harness.
  - After convergence, assert confidence is above a conservative floor, e.g. `> 0.25` initially.
  - Tune expected floor after observing values.

- [ ] `beat_confidence_is_bounded`
  - Run silence, periodic input, and maybe random-ish synthetic signal.
  - Assert confidence is finite and `0.0 <= confidence <= 1.0`.

- [ ] `beat_confidence_prefers_lattice_support`
  - Synthetic ACF peaks at `τ`, `2τ`, and strong `4τ`.
  - Synthetic onset pulses at `τ`.
  - Assert confidence for the beat-level lock becomes meaningfully positive.

- [ ] Existing tests should continue to pass:
  - `cargo test -p dsp`.

---

## Task 10: Tuning/Inspection Pass

After tests pass, run the app and inspect:

- silence
- internal test tone
- a steady 120 BPM source
- content with strong 120/60/30 lattice
- half-time section
- non-drum melodic content

Record observed confidence ranges:

- silence
- weak/noisy content
- plausible but uncertain rhythm
- strong lock

Use those ranges to tune:

- `ACTIVITY_KNEE`
- `COMB_CONF_KNEE`
- `PHASE_STRENGTH_KNEE`
- `PHASE_Z_LOW` / `PHASE_Z_HIGH`
- confidence attack/release alphas

Target practical interpretation:

- `< 0.25`: ignore beat detector.
- `0.25..0.6`: maybe use for subtle visuals only.
- `> 0.6`: safe for beat-synced visuals.

---

## Non-Goals

- Do not replace TEA in this pass.
- Do not add a binary lock output yet.
- Do not add meter/downbeat detection yet.
- Do not change public buffer shapes.
- Do not remove the internal relative `pulse_score`; TEA still needs it.

---

## Open Questions

- Should confidence gate `beatPulses` inside DSP, or should TS consumers decide?
- Should we expose debug confidence components as separate buffers later?
- Should `beatState[2]` / `beatState[3]` become `beats_per_measure` / `meter_confidence` once the comb confidence is stable?
- Should the confidence smoothing time constants become runtime params?
