# Beat Phase Lock-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the beat-grid from sliding on non-drum content by replacing the integer-argmax phi observation with a top-K local-maxima list, then having the PLL snap to the candidate nearest its prediction.

**Architecture:** Three small changes in `crates/dsp/src/beat.rs`. (1) `score_phase_for_tau` returns top-K local maxima of the per-φ correlation curve instead of a single argmax. (2) A new pure helper `pick_snapped_phi` chooses among them based on PLL prediction. (3) `PHASE_CORRECTION_ALPHA` bumped 0.01 → 0.05 and the dead `PHASE_CORRECTION_MAX_HOPS` clamp removed.

**Tech Stack:** Rust (`cargo test -p dsp`). All edits in one file; no JS/wasm rebuild required for tests.

**Spec:** `docs/superpowers/specs/2026-04-30-beat-phase-lock-design.md`

---

## File Structure

All changes in **`crates/dsp/src/beat.rs`**:
- Modify `score_phase_for_tau` signature (returns top-K)
- Modify `score_beat_hypotheses` call site (line ~369)
- Modify `update_tea` call site + observation logic (line ~507)
- Add new constants `PHI_CANDIDATE_COUNT`, `PHI_CANDIDATE_MIN_RATIO`
- Add new pure helper `pick_snapped_phi`
- Bump `PHASE_CORRECTION_ALPHA`, delete `PHASE_CORRECTION_MAX_HOPS`

Tests added to existing **`crates/dsp/src/lib.rs`** `mod tests` block (matches project convention — `score_phase_for_tau` is `pub` and `pick_snapped_phi` will be `pub(crate)` so both reachable from `crate::beat::...`).

---

## Task 1: Refactor `score_phase_for_tau` to return top-K local maxima

**Files:**
- Modify: `crates/dsp/src/beat.rs:1-56` (signature, body)
- Modify: `crates/dsp/src/beat.rs:369` (call in `score_beat_hypotheses`)
- Modify: `crates/dsp/src/beat.rs:507` (call in `update_tea`)
- Test: `crates/dsp/src/lib.rs` (add to existing `mod tests`)

- [ ] **Step 1: Write the failing test**

Add at the end of `mod tests` in `crates/dsp/src/lib.rs` (just before the closing `}` of the `mod tests` block):

```rust
#[test]
fn score_phase_top_k_returns_local_maxima() {
    use crate::beat::{score_phase_for_tau, PHI_CANDIDATE_COUNT};
    // n=189, tau=20.0; pulses at i=0,20,...,180 → true phi=8 for the kick
    // (last=188, 188-180=8). Distractor pulses at i=15,35,...,175 → true
    // phi=13 (188-175=13). Both phi values are interior to [0,20), so the
    // local-max picker (which skips endpoints) can see them.
    let n = 189usize;
    let tau = 20.0f32;
    let mut onset = vec![0.0f32; n];
    for i in (0..n).step_by(20) { onset[i] = 1.0; }
    for i in (15..n).step_by(20) { onset[i] = 0.4; }

    let (cands, _sum, _sum_sq, _n_phi) = score_phase_for_tau(&onset, tau);
    assert_eq!(cands.len(), PHI_CANDIDATE_COUNT);
    assert_eq!(cands[0].0, 8, "kick phi=8 expected at slot 0, got {:?}", cands);
    assert!(cands[0].1 > 0.0);
    let distractor = cands.iter().skip(1).find(|(phi, c)| *c > 0.0 && *phi == 13);
    assert!(
        distractor.is_some(),
        "distractor phi=13 expected somewhere in top-K, got {:?}", cands,
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsp score_phase_top_k_returns_local_maxima`
Expected: compile error (`PHI_CANDIDATE_COUNT` not defined / wrong return type for `score_phase_for_tau`).

- [ ] **Step 3: Add constants and refactor `score_phase_for_tau`**

In `crates/dsp/src/beat.rs`, replace the `PULSE_N` const block at the top with:

```rust
//! Candidate picking → phase scoring → TEA → beat outputs.

/// Number of pulses per train in `score_phase_for_tau`.
pub const PULSE_N: usize = 4;

/// Number of top φ local maxima returned by `score_phase_for_tau`. The PLL
/// in `update_tea` picks among these by snapping to its prediction; a fixed
/// stack-allocated array keeps the hot path allocation-free.
pub const PHI_CANDIDATE_COUNT: usize = 5;
```

Then replace the entire `score_phase_for_tau` function (`crates/dsp/src/beat.rs:6-56`) with:

```rust
/// Score one tempo lag `tau` against the OSS by sweeping integer phases
/// `phi ∈ [0, ceil(tau))`. Returns the top-`PHI_CANDIDATE_COUNT` local maxima
/// of the per-phase correlation (sorted descending by `corr`; empty slots
/// have `corr == 0.0` and `phi` unspecified) plus `(sum, sum_sq, n_phases)`
/// summary stats. Pulse-train is the paper's combined `Φ₁ (w=1.0) + Φ₂
/// (w=0.5) + Φ₁.₅ (w=0.5)` with N=4 pulses each. Local maxima skip endpoints
/// of the φ range (matches `pick_candidates`).
pub fn score_phase_for_tau(
    onset: &[f32],
    tau: f32,
) -> ([(usize, f32); PHI_CANDIDATE_COUNT], f32, f32, usize) {
    let mut top = [(0usize, 0.0f32); PHI_CANDIDATE_COUNT];
    let n = onset.len();
    if n == 0 || tau < 1.0 {
        return (top, 0.0, 0.0, 0);
    }
    let last = (n - 1) as i32;
    let phi_max = (tau.ceil() as usize).max(1);

    // Compute per-phi correlations into a scratch vec so we can scan for
    // local maxima after the fact. phi_max is bounded by tau_max ≈ 70 hops
    // so the allocation is small and short-lived.
    let mut corr = vec![0.0f32; phi_max];
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;

    for phi in 0..phi_max {
        let mut c = 0.0f32;
        for k in 0..PULSE_N {
            let off = (k as f32 * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                c += onset[pos as usize];
            }
        }
        for k in 0..PULSE_N {
            let off = (k as f32 * 2.0 * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                c += 0.5 * onset[pos as usize];
            }
        }
        for k in 0..PULSE_N {
            let off = ((k as f32 + 0.5) * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                c += 0.5 * onset[pos as usize];
            }
        }
        corr[phi] = c;
        sum += c;
        sum_sq += c * c;
    }

    // Insertion-sort local maxima into the top-K buffer (descending by corr).
    // Endpoints are skipped: phi=0 has no left neighbor, phi=phi_max-1 no
    // right neighbor. Equal-valued plateaus are not flagged (strict `>`).
    if phi_max >= 3 {
        for phi in 1..phi_max - 1 {
            let c = corr[phi];
            if c <= 0.0 || c <= corr[phi - 1] || c <= corr[phi + 1] {
                continue;
            }
            // Insertion sort into top[].
            let mut insert_at = PHI_CANDIDATE_COUNT;
            for i in 0..PHI_CANDIDATE_COUNT {
                if c > top[i].1 {
                    insert_at = i;
                    break;
                }
            }
            if insert_at < PHI_CANDIDATE_COUNT {
                for j in (insert_at + 1..PHI_CANDIDATE_COUNT).rev() {
                    top[j] = top[j - 1];
                }
                top[insert_at] = (phi, c);
            }
        }
    }

    (top, sum, sum_sq, phi_max)
}
```

- [ ] **Step 4: Update `score_beat_hypotheses` call site**

In `crates/dsp/src/beat.rs`, find the loop at `score_beat_hypotheses` (around line 367-380):

```rust
            let (phi, x, sum, sum_sq, n_phi) = score_phase_for_tau(onset, lag);
```

Replace with:

```rust
            let (phi_cands, sum, sum_sq, n_phi) = score_phase_for_tau(onset, lag);
            let phi = phi_cands[0].0;
            let x = phi_cands[0].1;
```

- [ ] **Step 5: Update `update_tea` call site (minimal — keep behavior identical for now)**

In `crates/dsp/src/beat.rs`, find `update_tea` (around line 507):

```rust
        let (phi, _, _, _, _) = score_phase_for_tau(onset, tau);
        let observed_phase = phi as f32;
```

Replace with:

```rust
        let (phi_cands, _, _, _) = score_phase_for_tau(onset, tau);
        let observed_phase = phi_cands[0].0 as f32;
```

(Snap-to-prediction logic is added in Task 2.)

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p dsp score_phase_top_k_returns_local_maxima`
Expected: PASS.

- [ ] **Step 7: Run all dsp tests to verify no regressions**

Run: `cargo test -p dsp`
Expected: all existing tests PASS (the call sites still read `phi_cands[0]`, which equals the previous `best_phi`).

- [ ] **Step 8: Commit**

```bash
git add crates/dsp/src/beat.rs crates/dsp/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(dsp): score_phase_for_tau returns top-K φ local maxima

Foundation for snap-to-prediction phase locking. Replaces single
argmax with PHI_CANDIDATE_COUNT=5 local maxima of the per-φ
correlation curve, sorted descending. Both callers updated to read
slot 0 (= previous best_phi), so behavior is unchanged at this step.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `pick_snapped_phi` helper and integrate into the PLL

**Files:**
- Modify: `crates/dsp/src/beat.rs` (add helper near other free functions, modify `update_tea`)
- Test: `crates/dsp/src/lib.rs` (add to existing `mod tests`)

- [ ] **Step 1: Write the failing tests**

Add at the end of `mod tests` in `crates/dsp/src/lib.rs`:

```rust
#[test]
fn pick_snapped_phi_picks_nearest_above_floor() {
    use crate::beat::{pick_snapped_phi, PHI_CANDIDATE_COUNT};
    let mut cands = [(0usize, 0.0f32); PHI_CANDIDATE_COUNT];
    cands[0] = (10, 1.0);
    cands[1] = (5, 0.8);
    cands[2] = (15, 0.4);
    let phi = pick_snapped_phi(&cands, 6.0, 20.0);
    assert_eq!(phi, 5.0, "phi=5 (corr 0.8 ≥ floor 0.7) is closest to expected=6");
}

#[test]
fn pick_snapped_phi_keeps_strongest_when_alternatives_below_floor() {
    use crate::beat::{pick_snapped_phi, PHI_CANDIDATE_COUNT};
    let mut cands = [(0usize, 0.0f32); PHI_CANDIDATE_COUNT];
    cands[0] = (10, 1.0);
    cands[1] = (15, 0.5);
    cands[2] = (12, 0.4);
    let phi = pick_snapped_phi(&cands, 11.0, 20.0);
    assert_eq!(phi, 10.0, "no candidate above floor 0.7 except slot 0");
}

#[test]
fn pick_snapped_phi_returns_nan_when_no_candidates() {
    use crate::beat::{pick_snapped_phi, PHI_CANDIDATE_COUNT};
    let cands = [(0usize, 0.0f32); PHI_CANDIDATE_COUNT];
    let phi = pick_snapped_phi(&cands, 5.0, 20.0);
    assert!(phi.is_nan());
}

#[test]
fn pick_snapped_phi_uses_circular_distance() {
    use crate::beat::{pick_snapped_phi, PHI_CANDIDATE_COUNT};
    // Expected=1, tau=20. Circular distance to phi=19 is 2 (wraps via 0);
    // linear distance is 18. Circular distance to slot 0 phi=15 is 6. With
    // circular distance the wrap-neighbor slot 1 wins despite slot 0 having
    // higher corr; with linear distance slot 0 would falsely win. This
    // test fails if signed_phase_delta is replaced with naive subtraction.
    let mut cands = [(0usize, 0.0f32); PHI_CANDIDATE_COUNT];
    cands[0] = (15, 1.0);
    cands[1] = (19, 0.9);
    let phi = pick_snapped_phi(&cands, 1.0, 20.0);
    assert_eq!(phi, 19.0);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p dsp pick_snapped_phi`
Expected: compile error (`pick_snapped_phi` not defined).

- [ ] **Step 3: Add the constant and the helper**

In `crates/dsp/src/beat.rs`, just below the `PHI_CANDIDATE_COUNT` const, add:

```rust
/// Relative-correlation floor for the snap-to-prediction picker. A φ
/// candidate must have `corr ≥ PHI_CANDIDATE_MIN_RATIO * top_corr` to be
/// eligible to override the global max — stops tiny secondary peaks from
/// winning just because they happen to sit closer to the prediction.
pub(crate) const PHI_CANDIDATE_MIN_RATIO: f32 = 0.7;
```

Then add the helper. Place it right after `signed_phase_delta` (around line 90 in the current file):

```rust
/// Pick the φ candidate closest to `expected` (circular distance mod `tau`),
/// among those with correlation at least `PHI_CANDIDATE_MIN_RATIO` of the
/// top correlation. Returns NaN if all slots are empty (`corr == 0.0`).
/// `candidates` must be sorted descending by correlation (as produced by
/// `score_phase_for_tau`).
pub(crate) fn pick_snapped_phi(
    candidates: &[(usize, f32); PHI_CANDIDATE_COUNT],
    expected: f32,
    tau: f32,
) -> f32 {
    let top_corr = candidates[0].1;
    if top_corr <= 0.0 {
        return f32::NAN;
    }
    let floor = PHI_CANDIDATE_MIN_RATIO * top_corr;
    let mut best_phi = candidates[0].0 as f32;
    let mut best_dist = signed_phase_delta(expected, best_phi, tau).abs();
    for &(phi, corr) in &candidates[1..] {
        if corr < floor || corr <= 0.0 {
            break;
        }
        let phi_f = phi as f32;
        let d = signed_phase_delta(expected, phi_f, tau).abs();
        if d < best_dist {
            best_dist = d;
            best_phi = phi_f;
        }
    }
    best_phi
}
```

- [ ] **Step 4: Wire `pick_snapped_phi` into `update_tea`**

In `crates/dsp/src/beat.rs`, replace the call site you edited in Task 1 Step 5:

```rust
        let (phi_cands, _, _, _) = score_phase_for_tau(onset, tau);
        let observed_phase = phi_cands[0].0 as f32;
```

with:

```rust
        let (phi_cands, _, _, _) = score_phase_for_tau(onset, tau);
        // Cold start: no prediction available, take the global max.
        // Tracking: snap to the candidate nearest the predicted phase.
        let observed_phase = if self.phase_smoothed.is_finite() && self.phase_updated_at != 0 {
            let predicted = wrap_phase(
                self.phase_smoothed
                    + self
                        .frame_index
                        .saturating_sub(self.phase_updated_at)
                        .max(1) as f32,
                tau,
            );
            pick_snapped_phi(&phi_cands, predicted, tau)
        } else {
            phi_cands[0].0 as f32
        };
```

Note: `observed_phase` may now be NaN (if all top-K slots are empty). The
existing predict-correct block already handles non-finite `phase_smoothed`
on the cold-start side, but `signed_phase_delta(expected, NaN, tau)` would
produce NaN and corrupt the smoother. Guard the correction:

Find the existing block (around line 509-521):

```rust
        self.phase_smoothed = if self.phase_smoothed.is_finite() && self.phase_updated_at != 0 {
            let elapsed_hops = self
                .frame_index
                .saturating_sub(self.phase_updated_at)
                .max(1) as f32;
            let expected_phase = wrap_phase(self.phase_smoothed + elapsed_hops, tau);
            let correction = (PHASE_CORRECTION_ALPHA
                * signed_phase_delta(expected_phase, observed_phase, tau))
            .clamp(-PHASE_CORRECTION_MAX_HOPS, PHASE_CORRECTION_MAX_HOPS);
            wrap_phase(expected_phase + correction, tau)
        } else {
            observed_phase
        };
```

Replace with (note `observed_phase.is_finite()` short-circuit and dropped clamp — the clamp is removed in Task 3, but we change shape here too):

```rust
        self.phase_smoothed = if self.phase_smoothed.is_finite() && self.phase_updated_at != 0 {
            let elapsed_hops = self
                .frame_index
                .saturating_sub(self.phase_updated_at)
                .max(1) as f32;
            let expected_phase = wrap_phase(self.phase_smoothed + elapsed_hops, tau);
            if observed_phase.is_finite() {
                let correction = (PHASE_CORRECTION_ALPHA
                    * signed_phase_delta(expected_phase, observed_phase, tau))
                .clamp(-PHASE_CORRECTION_MAX_HOPS, PHASE_CORRECTION_MAX_HOPS);
                wrap_phase(expected_phase + correction, tau)
            } else {
                expected_phase
            }
        } else {
            observed_phase
        };
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p dsp pick_snapped_phi`
Expected: 4 tests PASS.

- [ ] **Step 6: Run all dsp tests to verify no regressions**

Run: `cargo test -p dsp`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/dsp/src/beat.rs crates/dsp/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(dsp): PLL snaps φ observation to nearest prediction

Adds pick_snapped_phi helper. update_tea uses it to choose among the
top-K local maxima from score_phase_for_tau, picking the one closest
to the predicted phase (above 0.7 relative-corr floor). Stops the beat
grid from being yanked off-beat by spurious onsets in non-drum content.
Cold start still takes the global max.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Bump α and drop dead clamp

**Files:**
- Modify: `crates/dsp/src/beat.rs` (constants near top, `update_tea`)

- [ ] **Step 1: Edit constants**

In `crates/dsp/src/beat.rs`, find:

```rust
const PHASE_CORRECTION_ALPHA: f32 = 0.01;
const PHASE_CORRECTION_MAX_HOPS: f32 = 20.0;
```

Replace with:

```rust
// PLL gain on the per-frame phase observation. With top-K snap-to-prediction
// doing observation disambiguation, α only needs to absorb sub-hop noise; a
// faster α makes the grid responsive to genuine drift. ~20-frame TC at 47 Hz.
const PHASE_CORRECTION_ALPHA: f32 = 0.05;
```

(Delete `PHASE_CORRECTION_MAX_HOPS` entirely.)

- [ ] **Step 2: Drop the `.clamp()` call in `update_tea`**

In `crates/dsp/src/beat.rs`, find the block edited in Task 2 Step 4:

```rust
            if observed_phase.is_finite() {
                let correction = (PHASE_CORRECTION_ALPHA
                    * signed_phase_delta(expected_phase, observed_phase, tau))
                .clamp(-PHASE_CORRECTION_MAX_HOPS, PHASE_CORRECTION_MAX_HOPS);
                wrap_phase(expected_phase + correction, tau)
            } else {
```

Replace with (clamp removed; `signed_phase_delta` is already bounded by ±τ/2):

```rust
            if observed_phase.is_finite() {
                let correction = PHASE_CORRECTION_ALPHA
                    * signed_phase_delta(expected_phase, observed_phase, tau);
                wrap_phase(expected_phase + correction, tau)
            } else {
```

- [ ] **Step 3: Run all dsp tests**

Run: `cargo test -p dsp`
Expected: all tests PASS. (The α change affects steady-state convergence speed but no test asserts on PLL settling time.)

- [ ] **Step 4: Manual smoke test in browser**

Run: `npm run wasm && npm run dev`
Open the dev URL, start a tab/mic source, play music with non-drum content (sustained chords, vocals, fills). Visually confirm `BeatGridMarkers` lines stay locked rather than sliding when interfering content appears. On clean drum loops, behavior should be unchanged.

If the grid still slides on weird material, capture the symptom and consider the gating options (b)/(c) deferred from the spec.

- [ ] **Step 5: Commit**

```bash
git add crates/dsp/src/beat.rs
git commit -m "$(cat <<'EOF'
tune(dsp): bump beat PLL alpha 0.01 → 0.05, drop dead clamp

PHASE_CORRECTION_MAX_HOPS=20 was unreachable with α=0.01 and the
±τ/2 bound on signed_phase_delta. With top-K snap-to-prediction
handling observation disambiguation, α can be faster — ~20-frame TC
at 47 Hz lets the grid track real drift without lagging.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (for the implementer)

Before declaring done:

- `cargo test -p dsp` shows all tests passing.
- `grep PHASE_CORRECTION_MAX_HOPS crates/dsp/` returns nothing (constant fully removed).
- `grep "best_phi.*best_corr" crates/dsp/src/beat.rs` returns nothing in `score_phase_for_tau` (old return idiom gone).
- The four new tests appear in `crates/dsp/src/lib.rs`'s `mod tests`.
- Manual smoke test was performed (Task 3 Step 4) and the symptom is improved.
