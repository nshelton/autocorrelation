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

/// Asymmetric Hann: a rising half-Hann over the first `n - fall` samples, then
/// a falling half-Hann over the last `fall`. `fall = n/2` is the ordinary
/// symmetric Hann.
///
/// This exists for latency, not for spectral shape. The analysis window is
/// right-aligned — the newest sample sits at index `n-1`, where a symmetric
/// Hann weighs ~0 — so a fresh transient contributes nothing to the spectrum
/// until it has slid `n/2` samples to the center. That taper IS the onset
/// detection latency (21 ms at n=2048, 48 kHz). Shortening `fall` moves the
/// peak next to the newest sample instead, so a transient is at near-full
/// weight the moment it arrives. The cost is a wider mainlobe and higher near
/// sidelobes — the window stays C¹ at both ends and at the peak, so the
/// asymptotic rolloff is unchanged.
///
/// Both halves have mean 0.5 and mean-square 3/8 whatever their length, so
/// `sum(w)` and `sum(w²)` — and therefore `mag_scale` and
/// `parseval_band_scale` — come out the same for every `fall`. Amplitude
/// calibration is unaffected.
///
/// NOTE: at `fall = n/2` this is the *periodic* (DFT-even) Hann, `2πi/n`. The
/// window this replaced used the symmetric `2πi/(n-1)` form. The periodic form
/// is the correct one for DFT analysis; the difference is under 1e-5 per sample.
fn build_window(n: usize, fall: usize) -> Vec<f32> {
    let fall = fall.clamp(2, n / 2);
    let rise = n - fall;
    (0..n)
        .map(|i| {
            if i < rise {
                0.5 - 0.5 * (std::f32::consts::PI * i as f32 / rise as f32).cos()
            } else {
                0.5 + 0.5 * (std::f32::consts::PI * (i - rise) as f32 / fall as f32).cos()
            }
        })
        .collect()
}

/// `(mag_scale, parseval_band_scale)` for a window. Recomputed from the actual
/// coefficients rather than assumed, so a window shape change can never
/// silently desync the amplitude calibration.
fn window_scales(window: &[f32], n: usize) -> (f32, f32) {
    let energy: f32 = window.iter().map(|w| w * w).sum();
    (2.0 / window.iter().sum::<f32>(), 2.0 / (n as f32 * energy))
}

pub struct SpectrumState {
    fft: Arc<dyn RealToComplex<f32>>,
    fft_buffer: Vec<f32>,
    freq_buffer: Vec<Complex<f32>>,
    window: Vec<f32>,
    /// 2/sum(window). FFT bin magnitude → amplitude-equivalent units.
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
    /// Parseval scale: 2 / (N · Σ w²). Maps Σ|X[k]|² over a band → band RMS².
    parseval_band_scale: f32,
}

impl SpectrumState {
    pub fn new(window_size: usize, sample_rate: f32, dt: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let freq_buffer = fft.make_output_vec();
        let spectrum_len = freq_buffer.len() - 1;
        // Symmetric by default — same latency as before until `windowFall` moves.
        let window = build_window(window_size, window_size / 2);
        let (mag_scale, parseval_band_scale) = window_scales(&window, window_size);
        let smoothing_alpha = 1.0 - (-dt / SMOOTHING_TAU_SECS_DEFAULT).exp();
        let onset_release_retention = (-dt / ONSET_SMOOTHING_TAU_SECS_DEFAULT).exp();
        let low_band_bin_end = bin_for_hz(LOW_BAND_HZ_MAX, sample_rate, window_size);
        let mid_band_bin_end = bin_for_hz(MID_BAND_HZ_MAX, sample_rate, window_size);
        Self {
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            window,
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

    /// Length of the window's falling edge, in samples. Clamped to [2, n/2];
    /// n/2 is the symmetric Hann. Shorter = lower onset latency, more leakage.
    pub fn set_window_fall(&mut self, fall: usize) {
        let n = self.fft_buffer.len();
        self.window = build_window(n, fall);
        let (mag, parseval) = window_scales(&self.window, n);
        self.mag_scale = mag;
        self.parseval_band_scale = parseval;
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
            self.fft_buffer[i] = input[i] * self.window[i];
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
        // parseval_band_scale = 2 / (N · Σ w²). The default window is the
        // PERIODIC Hann (2πi/n) — see build_window's note.
        let dt = 1024.0_f32 / 48000.0;
        let state = SpectrumState::new(2048, 48000.0, dt);
        let n = 2048usize;
        let energy: f32 = (0..n)
            .map(|i| {
                let h = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
                h * h
            })
            .sum();
        let expected = 2.0 / (n as f32 * energy);
        assert!(
            (state.parseval_band_scale - expected).abs() < 1e-10,
            "got {}, expected {}",
            state.parseval_band_scale,
            expected
        );
    }

    #[test]
    fn default_window_is_the_symmetric_hann() {
        let w = build_window(2048, 1024);
        assert!((w[0] - 0.0).abs() < 1e-6);
        assert!((w[1024] - 1.0).abs() < 1e-6, "peak at center, got {}", w[1024]);
        // Symmetric about the peak.
        for k in 1..1024 {
            assert!((w[1024 - k] - w[1024 + k]).abs() < 1e-5, "asymmetric at k={}", k);
        }
    }

    #[test]
    fn short_fall_puts_full_weight_next_to_the_newest_sample() {
        let n = 2048;
        let fall = 64;
        let w = build_window(n, fall);
        // The peak — and therefore the detection latency — moves from n/2 to
        // `fall` samples from the right edge. That IS the latency win.
        assert!((w[n - fall] - 1.0).abs() < 1e-6, "peak value {}", w[n - fall]);
        // A transient one hop old (1024 samples in from the right) used to sit
        // at full weight; now it's already on the way out, and a BRAND NEW
        // transient is the one being weighted heavily.
        assert!(w[n - 1] < 0.01, "newest sample still tapered: {}", w[n - 1]);
        assert!(w[n - fall / 2] > 0.4, "half a fall in should be substantial");
        // Ends still touch zero, so the window stays C¹ and the asymptotic
        // sidelobe rolloff is preserved.
        assert!(w[0] < 1e-6);
    }

    #[test]
    fn amplitude_calibration_is_independent_of_fall() {
        // Both half-Hanns have mean 0.5 and mean-square 3/8 whatever their
        // length, so sum(w) and sum(w²) — and every scale derived from them —
        // must not move when the window shape does.
        let dt = 1024.0_f32 / 48000.0;
        let mut state = SpectrumState::new(2048, 48000.0, dt);
        let (mag0, parseval0) = (state.mag_scale, state.parseval_band_scale);
        for fall in [512, 256, 128, 64, 32] {
            state.set_window_fall(fall);
            assert!(
                (state.mag_scale - mag0).abs() / mag0 < 1e-4,
                "mag_scale moved at fall={}: {} vs {}",
                fall,
                state.mag_scale,
                mag0
            );
            assert!(
                (state.parseval_band_scale - parseval0).abs() / parseval0 < 1e-4,
                "parseval scale moved at fall={}",
                fall
            );
        }
    }

    #[test]
    fn window_fall_is_clamped_to_half_the_window() {
        // Over-large values (the schema default is 2048 at every windowSize)
        // must land on the symmetric window, not panic on `n - fall`.
        let w = build_window(512, 99_999);
        assert_eq!(w.len(), 512);
        assert!((w[256] - 1.0).abs() < 1e-6);
        // Degenerate small values are floored, not zero-length.
        assert_eq!(build_window(512, 0).len(), 512);
    }

    #[test]
    fn short_fall_detects_a_transient_a_full_hop_earlier() {
        // The whole point, end to end: a burst at the very end of the window
        // (i.e. one that JUST arrived) must produce real flux with a short
        // fall, and near-nothing with the symmetric window.
        let dt = 1024.0_f32 / 48000.0;
        let n = 2048;
        let mut input = vec![0.0f32; n];
        for (i, s) in input.iter_mut().enumerate().skip(n - 256) {
            // Alternating full-scale — broadband, like the LatencyProbe burst.
            *s = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut spectrum = vec![0.0f32; n / 2 - 1];

        let mut sym = SpectrumState::new(n, 48000.0, dt);
        sym.process(&vec![0.0; n], &mut spectrum, -100.0);
        let (_, _, _, flux_sym) = sym.process(&input, &mut spectrum, -100.0);

        let mut asym = SpectrumState::new(n, 48000.0, dt);
        asym.set_window_fall(64);
        asym.process(&vec![0.0; n], &mut spectrum, -100.0);
        let (_, _, _, flux_asym) = asym.process(&input, &mut spectrum, -100.0);

        assert!(
            flux_asym > flux_sym * 4.0,
            "short fall should see a just-arrived transient far more strongly: \
             asym {} vs sym {}",
            flux_asym,
            flux_sym
        );
    }
}
