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

const MAX_PEAKS: usize = 10;
const MIN_PEAK_SPACING: usize = 3;
const BEAT_TRACKER_MIN_BPM: f32 = 40.0;
const BEAT_TRACKER_MAX_BPM: f32 = 220.0;
const BEAT_PULSE_CYCLES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const TEA_GAUSSIAN_SIGMA: f32 = 5.0;
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;

pub struct BeatState {
    tau_min: usize,
    tau_max: usize,
    cand_scratch: Vec<(usize, f32)>,
    pulse_x: [f32; MAX_PEAKS],
    pulse_v: [f32; MAX_PEAKS],
    pulse_phi: [f32; MAX_PEAKS],
    pulse_score: [f32; MAX_PEAKS],
    period_inst: f32,
    phase_inst: f32,
    score_inst: f32,
    tea_alpha: f32,
    tau_smoothed: f32,
    phase_smoothed: f32,
    beat_position: f32,
}

impl BeatState {
    pub fn new(rms_history_len: usize, dt: f32) -> Self {
        let onset_acf_len = rms_history_len / 2;
        let tau_min = ((60.0 / BEAT_TRACKER_MAX_BPM) / dt).floor().max(1.0) as usize;
        let tau_max_unbounded = ((60.0 / BEAT_TRACKER_MIN_BPM) / dt).ceil() as usize;
        let tau_max = tau_max_unbounded.min(onset_acf_len.saturating_sub(2));
        let tea_alpha = 1.0 - (-dt / TEA_TAU_DEFAULT_SECS).exp();
        Self {
            tau_min,
            tau_max,
            cand_scratch: Vec::with_capacity(onset_acf_len / 2 + 1),
            pulse_x: [0.0; MAX_PEAKS],
            pulse_v: [0.0; MAX_PEAKS],
            pulse_phi: [0.0; MAX_PEAKS],
            pulse_score: [0.0; MAX_PEAKS],
            period_inst: f32::NAN,
            phase_inst: f32::NAN,
            score_inst: 0.0,
            tea_alpha,
            tau_smoothed: f32::NAN,
            phase_smoothed: f32::NAN,
            beat_position: 0.0,
        }
    }

    pub fn set_tea_tau(&mut self, tau_secs: f32, dt: f32) {
        let tau = tau_secs.clamp(0.05, 60.0);
        self.tea_alpha = 1.0 - (-dt / tau).exp();
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
        self.pick_candidates(onset_acf_enhanced, candidates);
        self.score_candidates(onset, candidates);
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

    fn score_candidates(&mut self, onset: &[f32], candidates: &[f32]) {
        let mut count = 0usize;
        for i in 0..MAX_PEAKS {
            let lag = candidates[3 * i];
            if lag.is_nan() {
                break;
            }
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
            count += 1;
        }
        if count == 0 {
            self.period_inst = f32::NAN;
            self.phase_inst = f32::NAN;
            self.score_inst = 0.0;
            return;
        }
        let sum_x: f32 = self.pulse_x[..count].iter().sum();
        let sum_v: f32 = self.pulse_v[..count].iter().sum();
        let mut best_i = 0usize;
        let mut best_score = -1.0f32;
        for i in 0..count {
            let xn = if sum_x > 0.0 { self.pulse_x[i] / sum_x } else { 0.0 };
            let vn = if sum_v > 0.0 { self.pulse_v[i] / sum_v } else { 0.0 };
            let s = xn + vn;
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
            self.period_inst = candidates[3 * best_i];
            self.phase_inst = self.pulse_phi[best_i];
            self.score_inst = best_score;
        }
    }

    fn update_tea(&mut self, onset: &[f32], tea: &mut [f32]) {
        let alpha = self.tea_alpha;
        if self.score_inst > 0.0 && self.period_inst.is_finite() {
            let inv_2sig2 = 1.0 / (2.0 * TEA_GAUSSIAN_SIGMA * TEA_GAUSSIAN_SIGMA);
            for tau in 0..tea.len() {
                let delta = tau as f32 - self.period_inst;
                let g = (-delta * delta * inv_2sig2).exp();
                tea[tau] = (1.0 - alpha) * tea[tau] + alpha * g;
            }
        } else {
            for v in tea.iter_mut() {
                *v *= 1.0 - alpha;
            }
        }

        let upper = self.tau_max.min(tea.len() - 1);
        let mut best_i = self.tau_min;
        let mut best_v = -1.0f32;
        for i in self.tau_min..=upper {
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
        if best_i > self.tau_min && best_i < upper {
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
        self.phase_smoothed = phi as f32;
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
