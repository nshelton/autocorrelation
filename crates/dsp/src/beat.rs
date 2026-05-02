//! Candidate picking → phase scoring → TEA → beat outputs.

/// Number of pulses per train in `score_phase_for_tau`.
pub const PULSE_N: usize = 4;

/// Number of top φ local maxima returned by `score_phase_for_tau`. The PLL
/// in `update_tea` picks among these by snapping to its prediction; a fixed
/// stack-allocated array keeps the hot path allocation-free.
pub const PHI_CANDIDATE_COUNT: usize = 5;

/// Relative-correlation floor for the snap-to-prediction picker. A φ
/// candidate must have `corr ≥ PHI_CANDIDATE_MIN_RATIO * top_corr` to be
/// eligible to override the global max — stops tiny secondary peaks from
/// winning just because they happen to sit closer to the prediction.
pub(crate) const PHI_CANDIDATE_MIN_RATIO: f32 = 0.7;

/// Upper bound on `phi_max` (= ceil(tau)). At BEAT_TRACKER_MIN_BPM=40 and
/// dt=1024/48000, tau_max ≈ 70 hops, so 80 leaves headroom. Used to size
/// the per-φ correlation scratch buffer as a stack array — keeps
/// `score_phase_for_tau` allocation-free.
const PHI_MAX_CAP: usize = 80;

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
    let phi_max = (tau.ceil() as usize).max(1).min(PHI_MAX_CAP);

    // Stack-allocated per-φ correlation scratch. Sized by PHI_MAX_CAP which
    // exceeds the project's BPM-bound tau_max with headroom; phi_max above
    // is clamped to PHI_MAX_CAP as a defensive bound on out-of-spec callers.
    let mut corr = [0.0f32; PHI_MAX_CAP];
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

/// Maximum number of tempo candidates tracked per hop. Shared with `Buffers`,
/// which sizes `candidates` as `3 * MAX_PEAKS` triples (lag, mag, sharpness).
pub(crate) const MAX_PEAKS: usize = 50;
const MIN_PEAK_SPACING: usize = 3;
const MIN_BEAT_HYPOTHESIS_SPACING: f32 = 1.0;
const PEAK_SCAN_MIN_BPM: f32 = 1.0;
const BEAT_HYPOTHESIS_MIN_BPM: f32 = 1.0;
const BEAT_TRACKER_MAX_BPM: f32 = 220.0;
const BEAT_PULSE_CYCLES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const TEA_GAUSSIAN_SIGMA_DEFAULT: f32 = 5.0;
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;
const TEA_CANDIDATE_VOTE_COUNT: usize = 50;
const COMB_MAX_DIVISOR: usize = 8;
const COMB_MAX_MULTIPLE: usize = 16;
const COMB_ALIGNMENT_TOLERANCE: f32 = 2.0;
const COMB_CONF_KNEE: f32 = 0.05;
const ACTIVITY_KNEE: f32 = 0.01;
const PHASE_STRENGTH_KNEE: f32 = 0.1;
const PHASE_Z_LOW: f32 = 1.0;
const PHASE_Z_HIGH: f32 = 4.0;
const PHASE_EPS: f32 = 1e-6;
const CONFIDENCE_ATTACK_ALPHA: f32 = 0.2;
const CONFIDENCE_RELEASE_ALPHA: f32 = 0.03;
// Default time constant for the phase-smoothing PLL. EMA gain is derived as
// `α = 1 − exp(−dt / τ)`. Larger τ ⇒ slower α ⇒ harder lock. Settable at
// runtime via `set_phase_lock_tau` (param key "phaseLock").
const PHASE_LOCK_DEFAULT_SECS: f32 = 1.0;

/// Fit a parabola through `(−1, ym), (0, y0), (+1, yp)` and return the
/// vertex's signed offset from the center sample, in `[-0.5, +0.5]`. Returns
/// 0.0 if the triple isn't a strict local maximum (vertex would land outside
/// the bracket — bogus refinement).
fn parabolic_refine(ym: f32, y0: f32, yp: f32) -> f32 {
    let denom = ym - 2.0 * y0 + yp;
    if denom.abs() < 1e-9 {
        return 0.0;
    }
    let delta = 0.5 * (ym - yp) / denom;
    if delta.abs() <= 1.0 {
        delta
    } else {
        0.0
    }
}

/// Fold `tau` by powers of 2 into `[lo, hi)`. Every period has a unique fold
/// because `hi = 2 * lo` makes the range exactly one octave wide. Returns NaN
/// for non-positive or non-finite `tau` (would otherwise spin forever).
fn fold_to_canonical(tau: f32, lo: f32, hi: f32) -> f32 {
    if !tau.is_finite() || tau <= 0.0 || lo <= 0.0 || hi <= lo {
        return f32::NAN;
    }
    let mut t = tau;
    while t < lo {
        t *= 2.0;
    }
    while t >= hi {
        t *= 0.5;
    }
    t
}

fn wrap_phase(phase: f32, period: f32) -> f32 {
    if period > 0.0 && period.is_finite() {
        phase.rem_euclid(period)
    } else {
        f32::NAN
    }
}

fn signed_phase_delta(from: f32, to: f32, period: f32) -> f32 {
    let mut delta = (to - from).rem_euclid(period);
    if delta > 0.5 * period {
        delta -= period;
    }
    delta
}

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
        if corr < floor {
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

fn comb_weight(multiple: usize) -> f32 {
    match multiple {
        1 => 1.0,
        2 => 0.9,
        3 => 0.55,
        4 => 1.25,
        8 => 0.8,
        _ => 1.0 / (multiple as f32).sqrt(),
    }
}

fn comb_tolerance(multiple: usize) -> f32 {
    (COMB_ALIGNMENT_TOLERANCE + 0.25 * (multiple as f32).sqrt()).min(6.0)
}

fn knee_confidence(x: f32, knee: f32) -> f32 {
    if !x.is_finite() || x <= 0.0 {
        0.0
    } else {
        (x / (x + knee)).clamp(0.0, 1.0)
    }
}

fn activity_confidence(onset: &[f32]) -> f32 {
    if onset.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0f32;
    let mut count = 0usize;
    for &x in onset {
        if x.is_finite() {
            sum_sq += x.max(0.0) * x.max(0.0);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        knee_confidence((sum_sq / count as f32).sqrt(), ACTIVITY_KNEE)
    }
}

fn phase_confidence(top_corr: f32, sum: f32, sum_sq: f32, n_phi: usize) -> f32 {
    if n_phi == 0 || !top_corr.is_finite() || top_corr <= 0.0 {
        return 0.0;
    }
    let n = n_phi as f32;
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);
    let z = (top_corr - mean) / (var.sqrt() + PHASE_EPS);
    let strength = knee_confidence(top_corr, PHASE_STRENGTH_KNEE);
    let contrast = ((z - PHASE_Z_LOW) / (PHASE_Z_HIGH - PHASE_Z_LOW)).clamp(0.0, 1.0);
    (strength * contrast).clamp(0.0, 1.0)
}

#[derive(Clone, Copy)]
struct CombEvidence {
    score: f32,
    confidence: f32,
}

pub struct BeatState {
    tau_min: usize,
    tau_max: usize,
    beat_tau_min: usize,
    beat_tau_max: usize,
    tau_scores: Vec<f32>,
    phase_scores: Vec<f32>,
    cand_scratch: Vec<(usize, f32)>,
    beat_scratch: Vec<(f32, f32, f32)>,

    onset_windowed: Vec<f32>,
    beat_lag: [f32; MAX_PEAKS],
    beat_comb_score: [f32; MAX_PEAKS],
    beat_comb_conf: [f32; MAX_PEAKS],
    beat_count: usize,
    pulse_x: [f32; MAX_PEAKS],
    pulse_v: [f32; MAX_PEAKS],
    pulse_phi: [f32; MAX_PEAKS],
    pulse_score: [f32; MAX_PEAKS],
    pulse_conf: [f32; MAX_PEAKS],
    period_inst: f32,
    phase_inst: f32,
    score_inst: f32,
    comb_conf_inst: f32,
    phase_conf_inst: f32,
    tea_conf_inst: f32,
    confidence_inst: f32,
    confidence_smoothed: f32,
    tea_alpha: f32,
    tea_sigma: f32,
    phase_correction_alpha: f32,
    tau_smoothed: f32,
    phase_smoothed: f32,
    canonical_lo: usize,
    canonical_hi: usize,
    tau_score: Vec<f32>,
    phase_score_inst: Vec<f32>,
    frame_index: u64,
    phase_updated_at: u64,
    beat_position: f32,
}

impl BeatState {
    pub fn new(rms_history_len: usize, dt: f32) -> Self {
        let onset_acf_len = rms_history_len / 2;
        let tau_min = 30_usize;
        let tau_max = 100_usize;
        let beat_tau_min = tau_min;
        let beat_tau_max_unbounded = ((60.0 / BEAT_HYPOTHESIS_MIN_BPM) / dt).ceil() as usize;
        let beat_tau_max = beat_tau_max_unbounded.min(onset_acf_len.saturating_sub(2));
        let tea_alpha = 1.0 - (-dt / TEA_TAU_DEFAULT_SECS).exp();
        let phase_correction_alpha = 1.0 - (-dt / PHASE_LOCK_DEFAULT_SECS).exp();
        let canonical_lo = tau_min;
        let canonical_hi = 2 * tau_min;
        let tau_score = vec![0.0_f32; canonical_hi - canonical_lo];
        let phase_score_inst = vec![0.0_f32; tau_max];
        Self {
            tau_min,
            tau_max,
            beat_tau_min,
            beat_tau_max,
            tau_scores: vec![0.0_f32; onset_acf_len],
            phase_scores: vec![0.0_f32; onset_acf_len],
            cand_scratch: Vec::with_capacity(onset_acf_len / 2 + 1),
            beat_scratch: Vec::with_capacity(MAX_PEAKS * COMB_MAX_DIVISOR),
            onset_windowed: vec![0.0; rms_history_len],
            beat_lag: [f32::NAN; MAX_PEAKS],
            beat_comb_score: [0.0; MAX_PEAKS],
            beat_comb_conf: [0.0; MAX_PEAKS],
            beat_count: 0,
            pulse_x: [0.0; MAX_PEAKS],
            pulse_v: [0.0; MAX_PEAKS],
            pulse_phi: [0.0; MAX_PEAKS],
            pulse_score: [0.0; MAX_PEAKS],
            pulse_conf: [0.0; MAX_PEAKS],
            period_inst: f32::NAN,
            phase_inst: f32::NAN,
            score_inst: 0.0,
            comb_conf_inst: 0.0,
            phase_conf_inst: 0.0,
            tea_conf_inst: 0.0,
            confidence_inst: 0.0,
            confidence_smoothed: 0.0,
            tea_alpha,
            tea_sigma: TEA_GAUSSIAN_SIGMA_DEFAULT,
            phase_correction_alpha,
            tau_smoothed: 0.0_f32,
            phase_smoothed: 0.0_f32,
            canonical_lo,
            canonical_hi,
            tau_score,
            phase_score_inst,
            frame_index: 0,
            phase_updated_at: 0,
            beat_position: 0.0,
        }
    }

    pub fn set_tea_tau(&mut self, tau_secs: f32, dt: f32) {
        let tau = tau_secs.clamp(0.05, 60.0);
        self.tea_alpha = 1.0 - (-dt / tau).exp();
    }

    pub fn set_tea_sigma(&mut self, sigma: f32) {
        self.tea_sigma = sigma.clamp(0.1, 100.0);
    }

    pub fn set_phase_lock_tau(&mut self, tau_secs: f32, dt: f32) {
        let tau = tau_secs.clamp(0.01, 60.0);
        self.phase_correction_alpha = 1.0 - (-dt / tau).exp();
    }

    /// Run one beat-tracker frame: pick candidates, score phases, update TEA,
    /// write public outputs (`candidates`, `tea`, `beatGrid`, `beatState`,
    /// `beatPulses`).
    pub fn process(
        &mut self,
        onset: &[f32],
        onset_acf_enhanced: &[f32],
        candidates: &mut [f32],
        tea: &mut [f32],
        beat_grid: &mut [f32],
        beat_state: &mut [f32],
        beat_pulses: &mut [f32],
        dt: f32,
    ) {
        self.frame_index = self.frame_index.wrapping_add(1);

        // estimate tau scores
        let mut best_tau = 0.0_f32;
        let mut best_score = 0.0_f32;
        for candidate_tau in self.tau_min..self.tau_max {
            self.tau_scores[candidate_tau] = 0.0_f32;

            let mut idx = candidate_tau;
            let mut taps = 0.0_f32;

            while idx < onset_acf_enhanced.len() {
                self.tau_scores[candidate_tau] += onset_acf_enhanced[idx];

                idx += candidate_tau;
                taps += 1.0_f32;
            }

            if taps > 0.0_f32 {
                self.tau_scores[candidate_tau] /= taps.sqrt();
            }

            if self.tau_scores[candidate_tau] > best_score {
                best_score = self.tau_scores[candidate_tau];
                best_tau = candidate_tau as f32;
            }
        }

        let bi = best_tau as usize;
        if bi > self.tau_min && bi + 1 < self.tau_max {
            best_tau += parabolic_refine(
                self.tau_scores[bi - 1],
                self.tau_scores[bi],
                self.tau_scores[bi + 1],
            );
        }

        self.tau_smoothed = best_tau * self.tea_alpha + self.tau_smoothed * (1.0 - self.tea_alpha);
        let tau = self.tau_smoothed;

        // estimate phase

        let mut phase_measured: f32 = 0.0;
        let mut best_phase_score = -1.0_f32;

        for phase in 0..tau.round() as usize {
            let mut s = 0.0_f32;
            let last = onset.len() as isize - 1;
            let mut k = 0_isize;

            loop {
                let pos = last - phase as isize - (k as f32 * tau).round() as isize;
                if pos < 1 {
                    break;
                }
                s += onset[pos as usize] - onset[pos as usize - 1];
                k += 1;
            }

            if phase < self.phase_score_inst.len() {
                self.phase_score_inst[phase] = s;
            }
            if s > best_phase_score {
                best_phase_score = s;
                phase_measured = phase as f32;
            }
        }

        let mut phase_pred =
            self.phase_smoothed + (self.frame_index - self.phase_updated_at) as f32;

        if phase_pred > tau {
            phase_pred -= tau;
        }

        // Shortest signed arc on the circle so lerping doesn't swim backwards
        // through the full cycle when phase wraps at 0/tau.
        let mut delta = phase_measured - phase_pred;
        if delta > 0.5 * tau {
            delta -= tau;
        }
        if delta < -0.5 * tau {
            delta += tau;
        }

        self.phase_smoothed = (phase_pred + self.phase_correction_alpha * delta);

        self.phase_updated_at = self.frame_index;

        self.confidence_smoothed = 1.0_f32;
        self.score_inst = 1.0_f32;

        self.write_beat_outputs(beat_grid, beat_state, dt);
        self.update_beat_pulses(beat_pulses);
    }

    fn pick_candidates(&mut self, onset_acf_enhanced: &[f32], candidates: &mut [f32]) {
        for slot in candidates.iter_mut() {
            *slot = f32::NAN;
        }
        if self.tau_max < self.tau_min + 1 {
            return;
        }

        self.cand_scratch.clear();
        let upper = self.tau_max.min(onset_acf_enhanced.len() - 1);
        for k in self.tau_min..upper {
            let y = onset_acf_enhanced[k];
            if y > 0.0 && y > onset_acf_enhanced[k - 1] && y > onset_acf_enhanced[k + 1] {
                self.cand_scratch.push((k, y));
            }
        }

        self.cand_scratch
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut accepted: [u32; MAX_PEAKS] = [0; MAX_PEAKS];
        let mut count: usize = 0;
        for &(k, _) in &self.cand_scratch {
            if count == MAX_PEAKS {
                break;
            }
            let too_close = accepted[..count]
                .iter()
                .any(|&j| ((k as i32 - j as i32).unsigned_abs() as usize) < MIN_PEAK_SPACING);
            if !too_close {
                accepted[count] = k as u32;
                count += 1;
            }
        }

        for i in 0..count {
            let k = accepted[i] as usize;
            let y0 = onset_acf_enhanced[k - 1];
            let y1 = onset_acf_enhanced[k];
            let y2 = onset_acf_enhanced[k + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let (lag_frac, mag) = if denom.abs() < 1e-12 {
                (k as f32, y1)
            } else {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                (k as f32 + delta, y1 - 0.25 * (y0 - y2) * delta)
            };
            candidates[3 * i] = lag_frac;
            candidates[3 * i + 1] = mag;
            candidates[3 * i + 2] = -denom;
        }
    }

    fn raw_candidate_count(candidates: &[f32]) -> usize {
        let candidate_count = (candidates.len() / 3).min(MAX_PEAKS);
        for i in 0..candidate_count {
            if candidates[3 * i].is_nan() {
                return i;
            }
        }
        candidate_count
    }

    fn comb_evidence_for_tau(
        &self,
        tau: f32,
        candidates: &[f32],
        raw_count: usize,
    ) -> CombEvidence {
        let mut score = 0.0f32;
        let mut total_weight = 0.0f32;
        let mut hit_weight = 0.0f32;
        let mut hit_count = 0usize;

        for multiple in 1..=COMB_MAX_MULTIPLE {
            let target = tau * multiple as f32;
            if target > self.tau_max as f32 {
                break;
            }
            let weight = comb_weight(multiple);
            let tolerance = comb_tolerance(multiple);
            let mut best_hit = 0.0f32;

            for i in 0..raw_count {
                let peak_lag = candidates[3 * i];
                let peak_mag = candidates[3 * i + 1];
                if !peak_lag.is_finite() || !peak_mag.is_finite() || peak_mag <= 0.0 {
                    continue;
                }
                let dist = (peak_lag - target).abs();
                if dist <= tolerance {
                    let x = dist / tolerance;
                    let alignment = (-0.5 * x * x).exp();
                    best_hit = best_hit.max(peak_mag * alignment);
                }
            }

            total_weight += weight;
            if best_hit > 0.0 {
                hit_count += 1;
                hit_weight += weight;
                score += weight * best_hit;
            }
        }

        if hit_count == 0 || total_weight <= 0.0 {
            CombEvidence {
                score: 0.0,
                confidence: 0.0,
            }
        } else {
            let normalized_strength = score / total_weight;
            let strength = knee_confidence(normalized_strength, COMB_CONF_KNEE);
            let coverage = (hit_weight / total_weight).clamp(0.0, 1.0);
            CombEvidence {
                score: score / total_weight.sqrt(),
                confidence: (strength * coverage).sqrt().clamp(0.0, 1.0),
            }
        }
    }

    fn build_beat_hypotheses(&mut self, candidates: &[f32]) {
        self.beat_scratch.clear();
        self.beat_lag.fill(f32::NAN);
        self.beat_comb_score.fill(0.0);
        self.beat_comb_conf.fill(0.0);
        self.beat_count = 0;

        let raw_count = Self::raw_candidate_count(candidates);
        if raw_count == 0 || self.beat_tau_max < self.beat_tau_min {
            return;
        }

        for i in 0..raw_count {
            let peak_lag = candidates[3 * i];
            if !peak_lag.is_finite() {
                continue;
            }

            for divisor in 1..=COMB_MAX_DIVISOR {
                let tau = peak_lag / divisor as f32;
                if tau < self.beat_tau_min as f32 || tau > self.beat_tau_max as f32 {
                    continue;
                }

                let evidence = self.comb_evidence_for_tau(tau, candidates, raw_count);
                if evidence.score <= 0.0 {
                    continue;
                }

                let mut merged = false;
                for existing in &mut self.beat_scratch {
                    if (existing.0 - tau).abs() < MIN_BEAT_HYPOTHESIS_SPACING {
                        if evidence.score > existing.1 {
                            *existing = (tau, evidence.score, evidence.confidence);
                        }
                        merged = true;
                        break;
                    }
                }
                if !merged {
                    self.beat_scratch
                        .push((tau, evidence.score, evidence.confidence));
                }
            }
        }

        self.beat_scratch
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let count = self.beat_scratch.len().min(MAX_PEAKS);
        for i in 0..count {
            self.beat_lag[i] = self.beat_scratch[i].0;
            self.beat_comb_score[i] = self.beat_scratch[i].1;
            self.beat_comb_conf[i] = self.beat_scratch[i].2;
        }
        self.beat_count = count;
    }

    fn score_beat_hypotheses(&mut self, onset: &[f32], candidates: &[f32]) {
        self.build_beat_hypotheses(candidates);
        let count = self.beat_count;
        for i in 0..count {
            let lag = self.beat_lag[i];
            let (phi_cands, sum, sum_sq, n_phi) = score_phase_for_tau(onset, lag);
            let phi = phi_cands[0].0;
            let x = phi_cands[0].1;
            let n = n_phi as f32;
            let mean = if n > 0.0 { sum / n } else { 0.0 };
            let var = if n > 0.0 {
                (sum_sq / n - mean * mean).max(0.0)
            } else {
                0.0
            };
            self.pulse_x[i] = x;
            self.pulse_v[i] = var;
            self.pulse_phi[i] = phi as f32;
            self.pulse_conf[i] = phase_confidence(x, sum, sum_sq, n_phi);
        }
        for i in count..MAX_PEAKS {
            self.pulse_x[i] = 0.0;
            self.pulse_v[i] = 0.0;
            self.pulse_phi[i] = 0.0;
            self.pulse_score[i] = 0.0;
            self.pulse_conf[i] = 0.0;
        }
        if count == 0 {
            self.period_inst = f32::NAN;
            self.phase_inst = f32::NAN;
            self.score_inst = 0.0;
            self.comb_conf_inst = 0.0;
            self.phase_conf_inst = 0.0;
            return;
        }

        let sum_x: f32 = self.pulse_x[..count].iter().sum();
        let sum_v: f32 = self.pulse_v[..count].iter().sum();
        let sum_comb: f32 = self.beat_comb_score[..count].iter().sum();
        let mut best_i = 0usize;
        let mut best_score = -1.0f32;
        for i in 0..count {
            let xn = if sum_x > 0.0 {
                self.pulse_x[i] / sum_x
            } else {
                0.0
            };
            let vn = if sum_v > 0.0 {
                self.pulse_v[i] / sum_v
            } else {
                0.0
            };
            let cn = if sum_comb > 0.0 {
                self.beat_comb_score[i] / sum_comb
            } else {
                0.0
            };
            let s = (xn + vn) * cn;
            self.pulse_score[i] = s;
            if s > best_score {
                best_score = s;
                best_i = i;
            }
        }
        if best_score <= 0.0 {
            self.period_inst = f32::NAN;
            self.phase_inst = f32::NAN;
            self.score_inst = 0.0;
            self.comb_conf_inst = 0.0;
            self.phase_conf_inst = 0.0;
        } else {
            self.period_inst = self.beat_lag[best_i];
            self.phase_inst = self.pulse_phi[best_i];
            self.score_inst = best_score;
            self.comb_conf_inst = self.beat_comb_conf[best_i];
            self.phase_conf_inst = self.pulse_conf[best_i];
        }
    }

    fn tea_lock_confidence(&self, tea: &[f32], best_i: usize, best_v: f32, upper: usize) -> f32 {
        if !best_v.is_finite() || best_v <= 0.0 || tea.is_empty() || upper < self.beat_tau_min {
            return 0.0;
        }

        let guard = (2.0 * self.tea_sigma).ceil().max(1.0) as usize;
        let mut second_v = 0.0f32;
        for i in self.beat_tau_min..=upper {
            let dist = if i > best_i { i - best_i } else { best_i - i };
            if dist <= guard {
                continue;
            }
            let v = tea[i];
            if v.is_finite() && v > second_v {
                second_v = v;
            }
        }

        let peak_conf = best_v.clamp(0.0, 1.0);
        let margin_conf = ((best_v - second_v) / (best_v + PHASE_EPS)).clamp(0.0, 1.0);
        (peak_conf * margin_conf).sqrt().clamp(0.0, 1.0)
    }

    fn update_public_confidence(&mut self, onset: &[f32]) {
        let target = if self.period_inst.is_finite()
            && self.tau_smoothed.is_finite()
            && self.score_inst > 0.0
        {
            let activity = activity_confidence(onset);
            let evidence =
                (self.comb_conf_inst * self.phase_conf_inst * self.tea_conf_inst).clamp(0.0, 1.0);
            let agreement = if self.tea_sigma > 0.0 {
                let delta = self.tau_smoothed - self.period_inst;
                (-0.5 * delta * delta / (self.tea_sigma * self.tea_sigma)).exp()
            } else {
                0.0
            };
            activity * evidence.sqrt() * agreement.clamp(0.0, 1.0)
        } else {
            0.0
        }
        .clamp(0.0, 1.0);

        self.confidence_inst = target;
        let alpha = if target > self.confidence_smoothed {
            CONFIDENCE_ATTACK_ALPHA
        } else {
            CONFIDENCE_RELEASE_ALPHA
        };
        self.confidence_smoothed = (self.confidence_smoothed
            + alpha * (target - self.confidence_smoothed))
            .clamp(0.0, 1.0);
    }

    fn update_tea(&mut self, onset: &[f32], tea: &mut [f32]) {
        let alpha = self.tea_alpha;
        let mut top_indices = [usize::MAX; TEA_CANDIDATE_VOTE_COUNT];
        let mut top_scores = [0.0f32; TEA_CANDIDATE_VOTE_COUNT];
        for i in 0..self.beat_count {
            let lag = self.beat_lag[i];
            let score = self.pulse_score[i];
            if !lag.is_finite() || score <= 0.0 {
                continue;
            }
            for j in 0..TEA_CANDIDATE_VOTE_COUNT {
                if score > top_scores[j] {
                    for k in (j + 1..TEA_CANDIDATE_VOTE_COUNT).rev() {
                        top_scores[k] = top_scores[k - 1];
                        top_indices[k] = top_indices[k - 1];
                    }
                    top_scores[j] = score;
                    top_indices[j] = i;
                    break;
                }
            }
        }

        let score_sum: f32 = top_scores.iter().sum();
        if score_sum > 0.0 {
            let inv_score_sum = 1.0 / score_sum;
            let inv_2sig2 = 1.0 / (2.0 * self.tea_sigma * self.tea_sigma);
            for tau in 0..tea.len() {
                let mut vote = 0.0f32;
                for j in 0..TEA_CANDIDATE_VOTE_COUNT {
                    let i = top_indices[j];
                    if i == usize::MAX {
                        break;
                    }
                    let lag = self.beat_lag[i];
                    let score = top_scores[j];
                    let delta = tau as f32 - lag;
                    let g = (-delta * delta * inv_2sig2).exp();
                    vote += score * inv_score_sum * g;
                }
                tea[tau] = (1.0 - alpha) * tea[tau] + alpha * vote;
            }
        } else {
            for v in tea.iter_mut() {
                *v *= 1.0 - alpha;
            }
        }

        let upper = self.beat_tau_max.min(tea.len() - 1);
        let mut best_i = self.beat_tau_min;
        let mut best_v = -1.0f32;
        for i in self.beat_tau_min..=upper {
            if tea[i] > best_v {
                best_v = tea[i];
                best_i = i;
            }
        }
        if best_v <= 0.0 {
            self.tau_smoothed = f32::NAN;
            self.phase_smoothed = f32::NAN;
            self.tea_conf_inst = 0.0;
            self.update_public_confidence(onset);
            return;
        }
        self.tea_conf_inst = self.tea_lock_confidence(tea, best_i, best_v, upper);
        let mut tau = best_i as f32;
        if best_i > self.beat_tau_min && best_i < upper {
            let y0 = tea[best_i - 1];
            let y1 = tea[best_i];
            let y2 = tea[best_i + 1];
            let denom = y0 - 2.0 * y1 + y2;
            if denom.abs() > 1e-12 {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                tau = best_i as f32 + delta;
            }
        }
        self.tau_smoothed = tau;
        // Apply a linear taper to onset before phase extraction: weight
        // `1 - d/(PULSE_N·τ)` clamped to [0,1] where `d = last - i`. Newest
        // sample at full weight, oldest pulse-reach (k = PULSE_N - 1, i.e.
        // d = (PULSE_N - 1)·τ) at weight 1/PULSE_N, anything beyond at zero.
        // Without this, when a real beat ages past the deepest pulse `corr[phi]`
        // steps and (with high PLL gain) the phase argmax flips visibly.
        let n_onset = onset.len();
        let reach = (PULSE_N as f32 * tau).max(1.0);
        let inv_reach = 1.0 / reach;
        for i in 0..n_onset {
            let d = (n_onset - 1 - i) as f32;
            let w = (1.0 - d * inv_reach).max(0.0);
            self.onset_windowed[i] = w * onset[i];
        }
        let (phi_cands, _, _, _) = score_phase_for_tau(&self.onset_windowed, tau);
        // On non-drum content the per-φ correlation curve has multiple
        // near-equal local maxima; the global argmax flips frame-to-frame
        // and yanks the grid. Snap to the candidate nearest the PLL
        // prediction (above PHI_CANDIDATE_MIN_RATIO × top floor) to
        // disambiguate. Cold start has no prediction, so take the global max.
        self.phase_smoothed = if self.phase_smoothed.is_finite() && self.phase_updated_at != 0 {
            let elapsed_hops = self
                .frame_index
                .saturating_sub(self.phase_updated_at)
                .max(1) as f32;
            let expected_phase = wrap_phase(self.phase_smoothed + elapsed_hops, tau);
            let observed_phase = pick_snapped_phi(&phi_cands, expected_phase, tau);
            if observed_phase.is_finite() {
                let correction = self.phase_correction_alpha
                    * signed_phase_delta(expected_phase, observed_phase, tau);
                wrap_phase(expected_phase + correction, tau)
            } else {
                // Observation lost (silence, transient): coast on prediction.
                expected_phase
            }
        } else {
            phi_cands[0].0 as f32
        };
        self.phase_updated_at = self.frame_index;
        self.update_public_confidence(onset);
    }

    fn write_beat_outputs(&self, beat_grid: &mut [f32], beat_state: &mut [f32], dt: f32) {
        let p = self.tau_smoothed;
        let phi = self.phase_smoothed;
        let confidence = self.confidence_smoothed.clamp(0.0, 1.0);
        if p.is_nan() || phi.is_nan() || self.score_inst <= 0.0 {
            beat_grid[0] = f32::NAN;
            beat_grid[1] = f32::NAN;
            beat_grid[2] = 0.0;
            beat_state[0] = f32::NAN;
            beat_state[1] = 0.0;
            beat_state[2] = f32::NAN;
            beat_state[3] = f32::NAN;
        } else {
            beat_grid[0] = p;
            beat_grid[1] = phi;
            beat_grid[2] = confidence;
            beat_state[0] = if p > 0.0 { 60.0 / (p * dt) } else { f32::NAN };
            beat_state[1] = confidence;
            beat_state[2] = f32::NAN;
            beat_state[3] = f32::NAN;
        }
    }

    fn update_beat_pulses(&mut self, beat_pulses: &mut [f32]) {
        let period = self.tau_smoothed;
        let phase = self.phase_smoothed;
        let score = self.score_inst;
        if period.is_nan() || period <= 0.0 || score <= 0.0 || phase.is_nan() {
            for slot in beat_pulses.iter_mut() {
                *slot = f32::NAN;
            }
            return;
        }
        let phase_frac = (phase / period).clamp(0.0, 0.999_999);
        let prev = self.beat_position;
        let mut bp = prev.floor() + phase_frac;
        if bp < prev - 0.5 {
            bp += 1.0;
        }
        self.beat_position = bp.rem_euclid(16.0);
        for (i, &m) in BEAT_PULSE_CYCLES.iter().enumerate() {
            let frac = (self.beat_position / m).fract();
            beat_pulses[i] = 1.0 - frac;
        }
    }
}

#[cfg(test)]
impl BeatState {
    pub fn test_per_frame_estimate(&self) -> (f32, f32, f32) {
        (self.period_inst, self.phase_inst, self.score_inst)
    }
    pub fn tau_smoothed(&self) -> f32 {
        self.tau_smoothed
    }
    pub fn tea_alpha(&self) -> f32 {
        self.tea_alpha
    }
}
