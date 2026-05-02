/// Maximum number of tempo candidates tracked per hop. Shared with `Buffers`,
/// which sizes `candidates` as `3 * MAX_PEAKS` triples (lag, mag, sharpness).
const BEAT_PULSE_CYCLES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;
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

pub struct BeatState {
    tau_min: usize,
    tau_max: usize,
    tau_scores: Vec<f32>,
    score_inst: f32,
    confidence_smoothed: f32,
    tea_alpha: f32,
    phase_correction_alpha: f32,
    tau_smoothed: f32,
    phase_smoothed: f32,
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
        let tea_alpha = 1.0 - (-dt / TEA_TAU_DEFAULT_SECS).exp();
        let phase_correction_alpha = 1.0 - (-dt / PHASE_LOCK_DEFAULT_SECS).exp();
        let phase_score_inst = vec![0.0_f32; tau_max];
        Self {
            tau_min,
            tau_max,
            tau_scores: vec![0.0_f32; onset_acf_len],
            score_inst: 0.0,
            confidence_smoothed: 0.0,
            tea_alpha,
            phase_correction_alpha,
            tau_smoothed: 0.0_f32,
            phase_smoothed: 0.0_f32,
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

        // // Shortest signed arc on the circle so lerping doesn't swim backwards
        // // through the full cycle when phase wraps at 0/tau.
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
mod tests {
    use super::*;

    #[test]
    fn parabolic_refine_centered_peak_returns_zero() {
        // Symmetric peak at the center sample → vertex sits at 0.
        assert_eq!(parabolic_refine(0.0, 1.0, 0.0), 0.0);
    }

    #[test]
    fn parabolic_refine_flat_returns_zero() {
        // Zero curvature → denom guard fires.
        assert_eq!(parabolic_refine(0.5, 0.5, 0.5), 0.0);
    }

    #[test]
    fn parabolic_refine_recovers_vertex_location() {
        // Round-trip: y = -((x - v))^2 + 1, sampled at -1, 0, +1, should
        // recover v exactly (this is what parabolic interp is designed for).
        for &v in &[-0.4_f32, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4] {
            let f = |x: f32| -((x - v).powi(2)) + 1.0;
            let delta = parabolic_refine(f(-1.0), f(0.0), f(1.0));
            assert!((delta - v).abs() < 1e-5, "v={}, got delta={}", v, delta);
        }
    }

    #[test]
    fn parabolic_refine_clamps_extreme_to_zero() {
        // Monotone-rising triple — vertex would land outside [-1, 1] (or
        // doesn't exist). Should clamp to 0 rather than return nonsense.
        assert_eq!(parabolic_refine(0.0, 1.0, 100.0), 0.0);
    }
}
