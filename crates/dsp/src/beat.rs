//! Candidate picking → phase scoring → TEA → beat outputs.

/// Number of pulses per train in `score_phase_for_tau`.
pub const PULSE_N: usize = 4;

/// Score one tempo lag `tau` against the OSS by sweeping integer phases
/// `phi ∈ [0, ceil(tau))`. Returns `(best_phi, best_corr, sum_corr,
/// sum_corr_sq, n_phases)`. Pulse-train is the paper's combined
/// `Φ₁ (w=1.0) + Φ₂ (w=0.5) + Φ₁.₅ (w=0.5)` with N=4 pulses each.
pub fn score_phase_for_tau(onset: &[f32], tau: f32) -> (usize, f32, f32, f32, usize) {
    let n = onset.len();
    if n == 0 || tau < 1.0 {
        return (0, 0.0, 0.0, 0.0, 0);
    }
    let last = (n - 1) as i32;
    let phi_max = (tau.ceil() as usize).max(1);

    let mut best_phi = 0usize;
    let mut best_corr = -1.0f32;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;

    for phi in 0..phi_max {
        let mut corr = 0.0f32;
        for k in 0..PULSE_N {
            let off = (k as f32 * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                corr += onset[pos as usize];
            }
        }
        for k in 0..PULSE_N {
            let off = (k as f32 * 2.0 * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                corr += 0.5 * onset[pos as usize];
            }
        }
        for k in 0..PULSE_N {
            let off = ((k as f32 + 0.5) * tau).round() as i32;
            let pos = last - phi as i32 - off;
            if pos >= 0 && (pos as usize) < n {
                corr += 0.5 * onset[pos as usize];
            }
        }

        sum += corr;
        sum_sq += corr * corr;
        if corr > best_corr {
            best_corr = corr;
            best_phi = phi;
        }
    }

    (best_phi, best_corr.max(0.0), sum, sum_sq, phi_max)
}

/// Maximum number of tempo candidates tracked per hop. Shared with `Buffers`,
/// which sizes `candidates` as `3 * MAX_PEAKS` triples (lag, mag, sharpness).
pub(crate) const MAX_PEAKS: usize = 50;
const MIN_PEAK_SPACING: usize = 3;
const MIN_BEAT_HYPOTHESIS_SPACING: f32 = 1.0;
const PEAK_SCAN_MIN_BPM: f32 = 1.0;
const BEAT_HYPOTHESIS_MIN_BPM: f32 = 40.0;
const BEAT_TRACKER_MAX_BPM: f32 = 220.0;
const BEAT_PULSE_CYCLES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const TEA_GAUSSIAN_SIGMA_DEFAULT: f32 = 5.0;
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;
const TEA_CANDIDATE_VOTE_COUNT: usize = 5;
const COMB_MAX_DIVISOR: usize = 8;
const COMB_MAX_MULTIPLE: usize = 16;
const COMB_ALIGNMENT_TOLERANCE: f32 = 2.0;
const PHASE_CORRECTION_ALPHA: f32 = 0.01;
const PHASE_CORRECTION_MAX_HOPS: f32 = 20.0;

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

pub struct BeatState {
    tau_min: usize,
    tau_max: usize,
    beat_tau_min: usize,
    beat_tau_max: usize,
    cand_scratch: Vec<(usize, f32)>,
    beat_scratch: Vec<(f32, f32)>,
    beat_lag: [f32; MAX_PEAKS],
    beat_comb_score: [f32; MAX_PEAKS],
    beat_count: usize,
    pulse_x: [f32; MAX_PEAKS],
    pulse_v: [f32; MAX_PEAKS],
    pulse_phi: [f32; MAX_PEAKS],
    pulse_score: [f32; MAX_PEAKS],
    period_inst: f32,
    phase_inst: f32,
    score_inst: f32,
    tea_alpha: f32,
    tea_sigma: f32,
    tau_smoothed: f32,
    phase_smoothed: f32,
    frame_index: u64,
    phase_updated_at: u64,
    beat_position: f32,
}

impl BeatState {
    pub fn new(rms_history_len: usize, dt: f32) -> Self {
        let onset_acf_len = rms_history_len / 2;
        let tau_min = ((60.0 / BEAT_TRACKER_MAX_BPM) / dt).floor().max(1.0) as usize;
        let tau_max_unbounded = ((60.0 / PEAK_SCAN_MIN_BPM) / dt).ceil() as usize;
        let tau_max = tau_max_unbounded.min(onset_acf_len.saturating_sub(2));
        let beat_tau_min = tau_min;
        let beat_tau_max_unbounded = ((60.0 / BEAT_HYPOTHESIS_MIN_BPM) / dt).ceil() as usize;
        let beat_tau_max = beat_tau_max_unbounded.min(onset_acf_len.saturating_sub(2));
        let tea_alpha = 1.0 - (-dt / TEA_TAU_DEFAULT_SECS).exp();
        Self {
            tau_min,
            tau_max,
            beat_tau_min,
            beat_tau_max,
            cand_scratch: Vec::with_capacity(onset_acf_len / 2 + 1),
            beat_scratch: Vec::with_capacity(MAX_PEAKS * COMB_MAX_DIVISOR),
            beat_lag: [f32::NAN; MAX_PEAKS],
            beat_comb_score: [0.0; MAX_PEAKS],
            beat_count: 0,
            pulse_x: [0.0; MAX_PEAKS],
            pulse_v: [0.0; MAX_PEAKS],
            pulse_phi: [0.0; MAX_PEAKS],
            pulse_score: [0.0; MAX_PEAKS],
            period_inst: f32::NAN,
            phase_inst: f32::NAN,
            score_inst: 0.0,
            tea_alpha,
            tea_sigma: TEA_GAUSSIAN_SIGMA_DEFAULT,
            tau_smoothed: f32::NAN,
            phase_smoothed: f32::NAN,
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
        self.pick_candidates(onset_acf_enhanced, candidates);
        // Public candidates remain raw long-range ACF peaks for debugging;
        // TEA uses derived beat-grid hypotheses so bar-level peaks can anchor
        // their faster beat-level divisors instead of dragging tempo down.
        self.score_beat_hypotheses(onset, candidates);
        self.update_tea(onset, tea);
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

    fn comb_score_for_tau(&self, tau: f32, candidates: &[f32], raw_count: usize) -> f32 {
        let mut score = 0.0f32;
        let mut total_weight = 0.0f32;
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
                score += weight * best_hit;
            }
        }

        if hit_count == 0 || total_weight <= 0.0 {
            0.0
        } else {
            score / total_weight.sqrt()
        }
    }

    fn build_beat_hypotheses(&mut self, candidates: &[f32]) {
        self.beat_scratch.clear();
        self.beat_lag.fill(f32::NAN);
        self.beat_comb_score.fill(0.0);
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

                let score = self.comb_score_for_tau(tau, candidates, raw_count);
                if score <= 0.0 {
                    continue;
                }

                let mut merged = false;
                for existing in &mut self.beat_scratch {
                    if (existing.0 - tau).abs() < MIN_BEAT_HYPOTHESIS_SPACING {
                        if score > existing.1 {
                            *existing = (tau, score);
                        }
                        merged = true;
                        break;
                    }
                }
                if !merged {
                    self.beat_scratch.push((tau, score));
                }
            }
        }

        self.beat_scratch
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let count = self.beat_scratch.len().min(MAX_PEAKS);
        for i in 0..count {
            self.beat_lag[i] = self.beat_scratch[i].0;
            self.beat_comb_score[i] = self.beat_scratch[i].1;
        }
        self.beat_count = count;
    }

    fn score_beat_hypotheses(&mut self, onset: &[f32], candidates: &[f32]) {
        self.build_beat_hypotheses(candidates);
        let count = self.beat_count;
        for i in 0..count {
            let lag = self.beat_lag[i];
            let (phi, x, sum, sum_sq, n_phi) = score_phase_for_tau(onset, lag);
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
        }
        for i in count..MAX_PEAKS {
            self.pulse_x[i] = 0.0;
            self.pulse_v[i] = 0.0;
            self.pulse_phi[i] = 0.0;
            self.pulse_score[i] = 0.0;
        }
        if count == 0 {
            self.period_inst = f32::NAN;
            self.phase_inst = f32::NAN;
            self.score_inst = 0.0;
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
        } else {
            self.period_inst = self.beat_lag[best_i];
            self.phase_inst = self.pulse_phi[best_i];
            self.score_inst = best_score;
        }
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
            return;
        }
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
        let (phi, _, _, _, _) = score_phase_for_tau(onset, tau);
        let observed_phase = phi as f32;
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
        self.phase_updated_at = self.frame_index;
    }

    fn write_beat_outputs(&self, beat_grid: &mut [f32], beat_state: &mut [f32], dt: f32) {
        let p = self.tau_smoothed;
        let phi = self.phase_smoothed;
        let s = self.score_inst;
        if p.is_nan() || s <= 0.0 {
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
            beat_grid[2] = s;
            beat_state[0] = if p > 0.0 { 60.0 / (p * dt) } else { f32::NAN };
            beat_state[1] = s;
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
