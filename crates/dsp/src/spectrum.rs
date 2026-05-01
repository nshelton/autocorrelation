//! FFT + windowing + spectrum smoothing + per-band RMS + spectral flux.

use crate::acf::bin_for_hz;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Spectrum smoothing time constant in seconds (default).
const SMOOTHING_TAU_SECS_DEFAULT: f32 = 0.0956;
/// Scalar onset envelope release time constant in seconds (default).
const ONSET_SMOOTHING_TAU_SECS_DEFAULT: f32 = 0.05;
const LOW_BAND_HZ_MAX: f32 = 150.0;
const MID_BAND_HZ_MAX: f32 = 1500.0;

pub struct SpectrumState {
    fft: Arc<dyn RealToComplex<f32>>,
    fft_buffer: Vec<f32>,
    freq_buffer: Vec<Complex<f32>>,
    hann: Vec<f32>,
    /// 2/sum(hann). FFT bin magnitude → amplitude-equivalent units.
    mag_scale: f32,
    /// Previous frame's per-bin |X|, scaled by `mag_scale`. Used for spectral flux.
    prev_mag: Vec<f32>,
    /// EMA coefficient: `1 - exp(-dt / tau)`. Recomputed by `set_smoothing_tau`.
    smoothing_alpha: f32,
    /// Per-hop retention for the scalar onset envelope's falling edge.
    onset_release_retention: f32,
    onset_envelope: f32,
    low_band_bin_end: usize,
    mid_band_bin_end: usize,
    /// Parseval scale: 2 / (N · Σ hann²). Maps Σ|X[k]|² over a band → band RMS².
    parseval_band_scale: f32,
}

impl SpectrumState {
    pub fn new(window_size: usize, sample_rate: f32, dt: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let freq_buffer = fft.make_output_vec();
        let spectrum_len = freq_buffer.len() - 1;
        let hann: Vec<f32> = (0..window_size)
            .map(|i| {
                0.5 - 0.5
                    * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos()
            })
            .collect();
        let mag_scale = 2.0 / hann.iter().sum::<f32>();
        let smoothing_alpha = 1.0 - (-dt / SMOOTHING_TAU_SECS_DEFAULT).exp();
        let onset_release_retention = (-dt / ONSET_SMOOTHING_TAU_SECS_DEFAULT).exp();
        let low_band_bin_end = bin_for_hz(LOW_BAND_HZ_MAX, sample_rate, window_size);
        let mid_band_bin_end = bin_for_hz(MID_BAND_HZ_MAX, sample_rate, window_size);
        let hann_energy: f32 = hann.iter().map(|h| h * h).sum();
        let parseval_band_scale = 2.0 / (window_size as f32 * hann_energy);
        Self {
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            hann,
            mag_scale,
            prev_mag: vec![0.0; spectrum_len],
            smoothing_alpha,
            onset_release_retention,
            onset_envelope: 0.0,
            low_band_bin_end,
            mid_band_bin_end,
            parseval_band_scale,
        }
    }

    pub fn set_smoothing_tau(&mut self, tau_secs: f32, dt: f32) {
        let tau = tau_secs.clamp(0.001, 10.0);
        self.smoothing_alpha = 1.0 - (-dt / tau).exp();
    }

    pub fn set_onset_release_tau(&mut self, tau_secs: f32, dt: f32) {
        if tau_secs <= 0.0 {
            self.onset_release_retention = 0.0;
            return;
        }
        let tau = tau_secs.clamp(0.001, 10.0);
        self.onset_release_retention = (-dt / tau).exp();
    }

    /// Run one FFT hop. Writes the smoothed normalized [0,1] `spectrum`.
    /// Returns `(low_rms, mid_rms, high_rms, onset)` — three Parseval-correct
    /// band-RMS scalars (caller pushes into history buffers via push_history)
    /// and the instant-attack / exponential-release spectral-flux onset envelope.
    pub fn process(
        &mut self,
        input: &[f32],
        spectrum: &mut [f32],
        db_floor: f32,
    ) -> (f32, f32, f32, f32) {
        let window_size = self.fft_buffer.len();
        let n = input.len().min(window_size);

        for i in 0..n {
            self.fft_buffer[i] = input[i] * self.hann[i];
        }
        for i in n..window_size {
            self.fft_buffer[i] = 0.0;
        }

        let _ = self
            .fft
            .process(&mut self.fft_buffer, &mut self.freq_buffer);

        // Spectral flux + spectrum smoothing in one pass over bins 1..=N/2.
        let mut flux = 0.0f32;
        for (out_i, bin) in self.freq_buffer[1..=spectrum.len()].iter().enumerate() {
            let mag = (bin.re * bin.re + bin.im * bin.im).sqrt() * self.mag_scale;
            flux += (mag - self.prev_mag[out_i]).max(0.0);
            self.prev_mag[out_i] = mag;

            let db = if mag > 0.0 {
                20.0 * mag.log10()
            } else {
                db_floor
            };
            let clipped = db.clamp(db_floor, 0.0);
            let normalized = (clipped - db_floor) / (-db_floor);
            spectrum[out_i] =
                self.smoothing_alpha * normalized + (1.0 - self.smoothing_alpha) * spectrum[out_i];
        }

        // Per-band RMS via Parseval-correct FFT-bin energy summation.
        let nyquist_bin = self.freq_buffer.len() - 1;
        let mut low_e = 0.0f32;
        for k in 1..=self.low_band_bin_end {
            let c = self.freq_buffer[k];
            low_e += c.re * c.re + c.im * c.im;
        }
        let mut mid_e = 0.0f32;
        for k in (self.low_band_bin_end + 1)..=self.mid_band_bin_end {
            let c = self.freq_buffer[k];
            mid_e += c.re * c.re + c.im * c.im;
        }
        let mut high_e = 0.0f32;
        for k in (self.mid_band_bin_end + 1)..nyquist_bin {
            let c = self.freq_buffer[k];
            high_e += c.re * c.re + c.im * c.im;
        }
        let low_rms = (low_e * self.parseval_band_scale).sqrt();
        let mid_rms = (mid_e * self.parseval_band_scale).sqrt();
        let high_rms = (high_e * self.parseval_band_scale).sqrt();

        self.onset_envelope =
            follow_onset_envelope(self.onset_envelope, flux, self.onset_release_retention);

        (low_rms, mid_rms, high_rms, self.onset_envelope)
    }
}

fn follow_onset_envelope(previous: f32, input: f32, release_retention: f32) -> f32 {
    if input >= previous {
        input
    } else {
        (previous * release_retention).max(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothing_alpha_matches_time_constant_formula() {
        // alpha = 1 - exp(-dt/tau) where dt = hop_size / sample_rate
        let dt = 1024.0_f32 / 48000.0;
        let state = SpectrumState::new(2048, 48000.0, dt);
        let expected = 1.0 - (-dt / SMOOTHING_TAU_SECS_DEFAULT).exp();
        assert!(
            (state.smoothing_alpha - expected).abs() < 1e-6,
            "alpha {} != expected {}",
            state.smoothing_alpha,
            expected
        );
    }

    #[test]
    fn onset_release_retention_matches_time_constant_formula() {
        let dt = 1024.0_f32 / 48000.0;
        let state = SpectrumState::new(2048, 48000.0, dt);
        let expected = (-dt / ONSET_SMOOTHING_TAU_SECS_DEFAULT).exp();
        assert!(
            (state.onset_release_retention - expected).abs() < 1e-6,
            "retention {} != expected {}",
            state.onset_release_retention,
            expected
        );
    }

    #[test]
    fn onset_smoothing_can_be_disabled() {
        let dt = 1024.0_f32 / 48000.0;
        let mut state = SpectrumState::new(2048, 48000.0, dt);
        state.set_onset_release_tau(0.0, dt);
        assert_eq!(state.onset_release_retention, 0.0);
    }

    #[test]
    fn onset_envelope_has_instant_attack_and_exponential_release() {
        let retention = 0.8;
        assert_eq!(follow_onset_envelope(0.2, 1.0, retention), 1.0);
        assert_eq!(follow_onset_envelope(1.0, 0.0, retention), 0.8);
        assert_eq!(follow_onset_envelope(1.0, 0.9, retention), 0.9);
    }

    #[test]
    fn smoothing_alpha_at_legacy_settings_is_approximately_0_2() {
        // SMOOTHING_TAU_SECS_DEFAULT is chosen so that at sr=48000, hop=1024
        // alpha ≈ 0.2 — i.e., the legacy hard-coded value is preserved.
        let dt = 1024.0_f32 / 48000.0;
        let state = SpectrumState::new(2048, 48000.0, dt);
        assert!(
            (state.smoothing_alpha - 0.2).abs() < 0.005,
            "expected alpha ≈ 0.2 at legacy settings, got {}",
            state.smoothing_alpha
        );
    }

    #[test]
    fn smoothing_alpha_shrinks_at_smaller_hop() {
        // Halving hop ≈ halves alpha (small-dt regime: 1 - exp(-x) ≈ x).
        // Wall-clock dynamics stay the same; per-call coefficient changes.
        let dt_large = 1024.0_f32 / 48000.0;
        let dt_small = 512.0_f32 / 48000.0;
        let large = SpectrumState::new(2048, 48000.0, dt_large);
        let small = SpectrumState::new(2048, 48000.0, dt_small);
        assert!(
            small.smoothing_alpha < large.smoothing_alpha,
            "small {} should be < large {}",
            small.smoothing_alpha,
            large.smoothing_alpha
        );
        let ratio = small.smoothing_alpha / large.smoothing_alpha;
        assert!(
            (0.45..=0.55).contains(&ratio),
            "expected ratio ≈ 0.5, got {}",
            ratio
        );
    }

    #[test]
    fn band_bin_ends_at_default_settings() {
        let dt = 1024.0_f32 / 48000.0;
        let state = SpectrumState::new(2048, 48000.0, dt);
        assert_eq!(state.low_band_bin_end, 6);
        assert_eq!(state.mid_band_bin_end, 64);
    }

    #[test]
    fn parseval_band_scale_matches_formula() {
        // parseval_band_scale = 2 / (N · Σ hann²)
        let dt = 1024.0_f32 / 48000.0;
        let state = SpectrumState::new(2048, 48000.0, dt);
        let n = 2048usize;
        let hann_energy: f32 = (0..n)
            .map(|i| {
                let h =
                    0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();
                h * h
            })
            .sum();
        let expected = 2.0 / (n as f32 * hann_energy);
        assert!(
            (state.parseval_band_scale - expected).abs() < 1e-10,
            "got {}, expected {}",
            state.parseval_band_scale,
            expected
        );
    }
}
