//! Per-channel running-peak autogain. Amplitude-normalizes a stream so beat
//! detection and visualization see a roughly [0, 1] signal regardless of
//! input level. The peak grows instantly with the latest sample and decays
//! as exp(-dt/τ) per call.

const PEAK_FLOOR: f32 = 1e-3;

pub struct AutoGain {
    peak: f32,
    /// exp(-dt/τ). Recomputed when τ or dt changes.
    retention: f32,
}

impl AutoGain {
    pub fn new(dt: f32, tau_secs: f32) -> Self {
        let mut g = Self { peak: 0.0, retention: 0.0 };
        g.set_tau(tau_secs, dt);
        g
    }

    pub fn set_tau(&mut self, tau_secs: f32, dt: f32) {
        let tau = tau_secs.max(0.001);
        self.retention = (-dt / tau).exp();
    }

    pub fn apply(&mut self, value: f32) -> f32 {
        self.peak = value.max(self.retention * self.peak);
        value / self.peak.max(PEAK_FLOOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_constant_normalizes_to_one() {
        let mut g = AutoGain::new(0.02, 1.0);
        for _ in 0..5 {
            assert!((g.apply(1.0) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn silence_stays_zero() {
        let mut g = AutoGain::new(0.02, 1.0);
        for _ in 0..5 {
            assert_eq!(g.apply(0.0), 0.0);
        }
    }

    #[test]
    fn rising_signal_normalizes_to_one_at_each_new_peak() {
        let mut g = AutoGain::new(0.02, 1.0);
        for v in [0.1, 0.5, 1.0, 2.0, 5.0] {
            let n = g.apply(v);
            assert!((n - 1.0).abs() < 1e-6, "input {} got {}", v, n);
        }
    }

    #[test]
    fn loud_then_silence_decays_peak() {
        let mut g = AutoGain::new(0.02, 0.1);
        g.apply(1.0);
        let p0 = g.peak;
        for _ in 0..50 {
            g.apply(0.0);
        }
        assert!(g.peak < p0 * 0.1, "peak {} did not decay below 0.1*p0={}", g.peak, p0 * 0.1);
    }

    #[test]
    fn pulse_then_quiet_normalizes_below_one() {
        // After a peak of 1.0, a quieter value 0.3 in the same frame epoch
        // (high retention) should still be far below 1.0.
        let mut g = AutoGain::new(0.02, 5.0);
        g.apply(1.0);
        let n = g.apply(0.3);
        assert!(n < 0.5 && n > 0.0, "expected ~0.3, got {}", n);
    }
}
