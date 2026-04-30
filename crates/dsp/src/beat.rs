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
