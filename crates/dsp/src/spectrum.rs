//! FFT + windowing + spectrum smoothing + per-band RMS + spectral flux.

use crate::acf::bin_for_hz;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Spectrum smoothing time constant in seconds (default).
const SMOOTHING_TAU_SECS_DEFAULT: f32 = 0.0956;
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
    pub(crate) prev_mag: Vec<f32>,
    /// EMA coefficient: `1 - exp(-dt / tau)`. Recomputed by `set_smoothing_tau`.
    pub(crate) smoothing_alpha: f32,
    pub(crate) low_band_bin_end: usize,
    pub(crate) mid_band_bin_end: usize,
    /// Parseval scale: 2 / (N · Σ hann²). Maps Σ|X[k]|² over a band → band RMS².
    pub(crate) parseval_band_scale: f32,
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
            low_band_bin_end,
            mid_band_bin_end,
            parseval_band_scale,
        }
    }

    pub fn set_smoothing_tau(&mut self, tau_secs: f32, dt: f32) {
        let tau = tau_secs.clamp(0.001, 10.0);
        self.smoothing_alpha = 1.0 - (-dt / tau).exp();
    }

    /// Run one FFT hop. Writes the smoothed normalized [0,1] `spectrum`.
    /// Returns `(low_rms, mid_rms, high_rms, flux)` — three Parseval-correct
    /// band-RMS scalars (caller pushes into history buffers via push_history)
    /// and the spectral-flux onset value (Σ max(0, |X[k]| - prev_mag[k])).
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

        let _ = self.fft.process(&mut self.fft_buffer, &mut self.freq_buffer);

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

        (low_rms, mid_rms, high_rms, flux)
    }
}
