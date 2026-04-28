use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// Spectrum smoothing time constant in seconds. Chosen to preserve the
/// legacy alpha ≈ 0.2 behavior at sr=48000, hop=1024:
///   alpha = 1 - exp(-dt/tau), dt = 1024/48000 = 21.33 ms
///   0.2 ≈ 1 - exp(-21.33ms / 95.6ms)
const SMOOTHING_TAU_SECS: f32 = 0.0956;
const DB_FLOOR: f32 = -100.0;
const RMS_HISTORY_LEN: usize = 512;
const RMS_ACF_LEN: usize = RMS_HISTORY_LEN / 2;
/// Crossover from low band to mid band, in Hz. Drum-friendly default:
/// fits the kick fundamental (typically 50–90 Hz) cleanly inside "low"
/// without bleeding much into snare body.
const LOW_BAND_HZ_MAX: f32 = 150.0;
/// Crossover from mid band to high band, in Hz.
const MID_BAND_HZ_MAX: f32 = 1500.0;

#[wasm_bindgen]
pub struct Dsp {
    waveform: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_buffer: Vec<f32>,
    freq_buffer: Vec<Complex<f32>>,
    spectrum: Vec<f32>,
    hann: Vec<f32>,
    /// Magnitude scale factor that converts raw FFT bin magnitude to
    /// amplitude-equivalent units (so a unit-amplitude sine peaks at ~1.0).
    /// Equals 2/sum(hann) — the 2 accounts for the one-sided real spectrum,
    /// and sum(hann) ≈ N/2 corrects for window attenuation.
    mag_scale: f32,
    rms_history: Vec<f32>,
    buffer_acf: Vec<f32>,
    rms_acf: Vec<f32>,
    rms_detrended: Vec<f32>,
    /// Per-process-call EMA coefficient for the spectrum. Computed from
    /// `SMOOTHING_TAU_SECS` and the wall-clock dt between hops
    /// (`hop_size / sample_rate`), so changing `hop_size` does NOT change
    /// perceived smoothing dynamics.
    smoothing_alpha: f32,
    /// Inclusive last bin index of the low band. Low band = bins 1..=low_band_bin_end.
    /// Bin 0 (DC) is always skipped.
    low_band_bin_end: usize,
    /// Inclusive last bin index of the mid band. Mid = (low_end+1)..=mid_band_bin_end.
    /// High = (mid_end+1)..=N/2-1 (Nyquist excluded).
    mid_band_bin_end: usize,
    /// Parseval scale: converts Σ|X[k]|² over a band → band RMS² in time-domain
    /// units. Equals 2 / (N · Σ hann²). The 2 accounts for the one-sided real
    /// spectrum; Σ hann² is the window's energy correction.
    parseval_band_scale: f32,
}

#[wasm_bindgen]
impl Dsp {
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize, sample_rate: f32, hop_size: usize) -> Dsp {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let freq_buffer = fft.make_output_vec();
        let spectrum = vec![0.0; freq_buffer.len() - 1]; // drop DC
        let hann: Vec<f32> = (0..window_size)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos()
            })
            .collect();
        let mag_scale = 2.0 / hann.iter().sum::<f32>();
        let dt = hop_size as f32 / sample_rate;
        let smoothing_alpha = 1.0 - (-dt / SMOOTHING_TAU_SECS).exp();
        let low_band_bin_end = bin_for_hz(LOW_BAND_HZ_MAX, sample_rate, window_size);
        let mid_band_bin_end = bin_for_hz(MID_BAND_HZ_MAX, sample_rate, window_size);
        let hann_energy: f32 = hann.iter().map(|h| h * h).sum();
        let parseval_band_scale = 2.0 / (window_size as f32 * hann_energy);
        Dsp {
            waveform: vec![0.0; window_size],
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            spectrum,
            hann,
            mag_scale,
            rms_history: vec![0.0; RMS_HISTORY_LEN],
            buffer_acf: vec![0.0; window_size / 2],
            rms_acf: vec![0.0; RMS_ACF_LEN],
            rms_detrended: vec![0.0; RMS_HISTORY_LEN],
            smoothing_alpha,
            low_band_bin_end,
            mid_band_bin_end,
            parseval_band_scale,
        }
    }

    pub fn process(&mut self, input: &[f32]) {
        let n = input.len().min(self.waveform.len());
        self.waveform[..n].copy_from_slice(&input[..n]);

        // RMS over the input window
        let mean_sq: f32 = input.iter().take(n).map(|&x| x * x).sum::<f32>() / n.max(1) as f32;
        let rms = mean_sq.sqrt();

        // Shift left and append newest at the end (oldest at index 0)
        self.rms_history.copy_within(1.., 0);
        let last = self.rms_history.len() - 1;
        self.rms_history[last] = rms;

        autocorrelate(&self.waveform, &mut self.buffer_acf);

        // RMS-envelope ACF: detrend (subtract mean) then autocorrelate.
        // Without detrending, average loudness creates a DC bias that
        // drowns out tempo peaks.
        let mean = self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32;
        for (dst, src) in self.rms_detrended.iter_mut().zip(self.rms_history.iter()) {
            *dst = src - mean;
        }
        autocorrelate(&self.rms_detrended, &mut self.rms_acf);

        // Apply Hann window
        for i in 0..n {
            self.fft_buffer[i] = input[i] * self.hann[i];
        }
        // Zero-pad if input shorter than window
        for i in n..self.fft_buffer.len() {
            self.fft_buffer[i] = 0.0;
        }

        // Forward real FFT
        let _ = self
            .fft
            .process(&mut self.fft_buffer, &mut self.freq_buffer);

        // Magnitude → dB → normalized [0, 1] → smoothed
        // Skip bin 0 (DC); use bins 1..=spectrum.len()
        for (out_i, bin) in self.freq_buffer[1..=self.spectrum.len()].iter().enumerate() {
            let mag = (bin.re * bin.re + bin.im * bin.im).sqrt() * self.mag_scale;
            let db = if mag > 0.0 {
                20.0 * mag.log10()
            } else {
                DB_FLOOR
            };
            let clipped = db.clamp(DB_FLOOR, 0.0);
            let normalized = (clipped - DB_FLOOR) / (-DB_FLOOR); // [0, 1]
            self.spectrum[out_i] = self.smoothing_alpha * normalized
                + (1.0 - self.smoothing_alpha) * self.spectrum[out_i];
        }
    }

    pub fn waveform(&self) -> Vec<f32> {
        self.waveform.clone()
    }

    pub fn spectrum(&self) -> Vec<f32> {
        self.spectrum.clone()
    }

    pub fn buffer_acf(&self) -> Vec<f32> {
        self.buffer_acf.clone()
    }

    pub fn rms_acf(&self) -> Vec<f32> {
        self.rms_acf.clone()
    }

    pub fn rms_history(&self) -> Vec<f32> {
        self.rms_history.clone()
    }
}

/// Direct time-domain autocorrelation, normalized so output[0] == 1.0
/// for any nonzero input. For all-zero input the output is filled with
/// zeros (no NaN from division by zero). The caller chooses how many
/// lags to compute via the length of `output`.
fn autocorrelate(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    for k in 0..output.len() {
        let mut sum = 0.0f32;
        if k < n {
            for i in 0..(n - k) {
                sum += input[i] * input[i + k];
            }
        }
        output[k] = sum;
    }
    let zero = output[0];
    if zero > 0.0 {
        for v in output.iter_mut() {
            *v /= zero;
        }
    } else {
        output.fill(0.0);
    }
}

/// Snap a frequency in Hz to the nearest one-sided real-FFT bin index,
/// clamped to [1, N/2 - 1] (DC and Nyquist are excluded by design).
fn bin_for_hz(hz: f32, sample_rate: f32, n: usize) -> usize {
    let bin = (hz * n as f32 / sample_rate).round() as usize;
    bin.clamp(1, n / 2 - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_then_waveform_returns_input() {
        let mut dsp = Dsp::new(8, 48000.0, 4);
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        dsp.process(&input);
        assert_eq!(dsp.waveform(), input);
    }

    #[test]
    fn spectrum_has_window_size_div_2_bins() {
        let dsp = Dsp::new(2048, 48000.0, 1024);
        assert_eq!(dsp.spectrum().len(), 1024);
    }

    #[test]
    fn silent_input_yields_low_spectrum() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let silent = vec![0.0; 2048];
        for _ in 0..30 {
            dsp.process(&silent);
        }
        let spec = dsp.spectrum();
        for &v in &spec {
            assert!(v < 0.1, "expected silent → near-zero bin, got {}", v);
        }
    }

    #[test]
    fn rms_of_unit_amplitude_constant_is_one() {
        let mut dsp = Dsp::new(8, 48000.0, 4);
        let constant = vec![1.0_f32; 8];
        dsp.process(&constant);
        let h = dsp.rms_history();
        assert_eq!(h.len(), RMS_HISTORY_LEN);
        // Newest sample at the end
        let last = h[h.len() - 1];
        assert!((last - 1.0).abs() < 1e-6, "got {}", last);
    }

    #[test]
    fn rms_of_silence_is_zero() {
        let mut dsp = Dsp::new(8, 48000.0, 4);
        dsp.process(&vec![0.0_f32; 8]);
        let h = dsp.rms_history();
        assert_eq!(h[h.len() - 1], 0.0);
    }

    #[test]
    fn rms_history_shifts_oldest_out() {
        let mut dsp = Dsp::new(4, 48000.0, 4);
        // Push three distinct values
        dsp.process(&[1.0, 1.0, 1.0, 1.0]); // rms = 1
        dsp.process(&[2.0, 2.0, 2.0, 2.0]); // rms = 2
        dsp.process(&[0.0, 0.0, 0.0, 0.0]); // rms = 0
        let h = dsp.rms_history();
        let n = h.len();
        // Newest three values are at the end in the order pushed
        assert_eq!(h[n - 3], 1.0);
        assert_eq!(h[n - 2], 2.0);
        assert_eq!(h[n - 1], 0.0);
    }

    #[test]
    fn loud_sine_produces_a_peak() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let sr = 48000.0_f32;
        let freq = 1000.0_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();
        for _ in 0..30 {
            dsp.process(&signal);
        }
        let spec = dsp.spectrum();
        let (argmax, &peak) = spec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            (40..=50).contains(&(argmax + 1)),
            "expected peak near bin 43, got {}",
            argmax + 1
        );
        assert!(peak > 0.5, "expected loud peak, got {}", peak);
    }

    #[test]
    fn autocorrelate_helper_correctness() {
        // Hand-computed for input [1, 2, 3, 4] with output length 3:
        //   raw[0] = 1*1 + 2*2 + 3*3 + 4*4 = 30
        //   raw[1] = 1*2 + 2*3 + 3*4       = 20
        //   raw[2] = 1*3 + 2*4             = 11
        // Normalized by raw[0]=30: [1.0, 20/30, 11/30].
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 3];
        autocorrelate(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6, "got {}", output[0]);
        assert!((output[1] - 20.0 / 30.0).abs() < 1e-6, "got {}", output[1]);
        assert!((output[2] - 11.0 / 30.0).abs() < 1e-6, "got {}", output[2]);
    }

    #[test]
    fn buffer_acf_has_correct_length() {
        let dsp = Dsp::new(2048, 48000.0, 1024);
        assert_eq!(dsp.buffer_acf().len(), 1024);
    }

    #[test]
    fn buffer_acf_zero_lag_is_one_for_nonzero_signal() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let signal: Vec<f32> = (0..2048)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        dsp.process(&signal);
        let acf = dsp.buffer_acf();
        assert!((acf[0] - 1.0).abs() < 1e-6, "got {}", acf[0]);
    }

    #[test]
    fn buffer_acf_of_sine_peaks_at_period() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        let sr = 48000.0_f32;
        let freq = 1000.0_f32;
        // Period at this sr/freq is exactly 48 samples.
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let acf = dsp.buffer_acf();
        // ACF of a sine has local maxima at integer multiples of the period.
        // Verify lag 48 is a local maximum (greater than its neighbors) AND
        // the correlation is strong (close to 1.0 after normalization).
        assert!(acf[48] > acf[47], "expected acf[48]={} > acf[47]={}", acf[48], acf[47]);
        assert!(acf[48] > acf[49], "expected acf[48]={} > acf[49]={}", acf[48], acf[49]);
        assert!(acf[48] > 0.9, "expected strong peak at period, got acf[48]={}", acf[48]);
    }

    #[test]
    fn rms_acf_has_correct_length() {
        let dsp = Dsp::new(2048, 48000.0, 1024);
        assert_eq!(dsp.rms_acf().len(), 256);
    }

    #[test]
    fn acf_of_silence_is_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024);
        dsp.process(&vec![0.0_f32; 2048]);
        for &v in dsp.buffer_acf().iter() {
            assert_eq!(v, 0.0);
        }
        for &v in dsp.rms_acf().iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn rms_acf_constant_input_is_zero() {
        // Fill rms_history with a constant rms value, then verify the
        // detrended (mean-subtracted) ACF is zero everywhere.
        let mut dsp = Dsp::new(8, 48000.0, 4);
        let constant = vec![0.5_f32; 8];
        // RMS of [0.5; 8] is 0.5; need >= RMS_HISTORY_LEN (512) calls to fully fill.
        for _ in 0..512 {
            dsp.process(&constant);
        }
        let acf = dsp.rms_acf();
        assert_eq!(acf.len(), 256);
        for &v in &acf {
            assert!(v.abs() < 1e-5, "expected near-zero ACF for constant rms, got {}", v);
        }
    }

    #[test]
    fn smoothing_alpha_matches_time_constant_formula() {
        // alpha = 1 - exp(-dt/tau) where dt = hop_size / sample_rate
        let dsp = Dsp::new(2048, 48000.0, 1024);
        let dt = 1024.0_f32 / 48000.0;
        let expected = 1.0 - (-dt / SMOOTHING_TAU_SECS).exp();
        assert!(
            (dsp.smoothing_alpha - expected).abs() < 1e-6,
            "alpha {} != expected {}",
            dsp.smoothing_alpha,
            expected
        );
    }

    #[test]
    fn smoothing_alpha_at_legacy_settings_is_approximately_0_2() {
        // SMOOTHING_TAU_SECS is chosen so that at sr=48000, hop=1024
        // alpha ≈ 0.2 — i.e., the legacy hard-coded value is preserved.
        let dsp = Dsp::new(2048, 48000.0, 1024);
        assert!(
            (dsp.smoothing_alpha - 0.2).abs() < 0.005,
            "expected alpha ≈ 0.2 at legacy settings, got {}",
            dsp.smoothing_alpha
        );
    }

    #[test]
    fn smoothing_alpha_shrinks_at_smaller_hop() {
        // Halving hop ≈ halves alpha (small-dt regime: 1 - exp(-x) ≈ x).
        // Wall-clock dynamics stay the same; per-call coefficient changes.
        let large = Dsp::new(2048, 48000.0, 1024);
        let small = Dsp::new(2048, 48000.0, 512);
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
    fn bin_for_hz_snaps_at_default_settings() {
        // 150 Hz at sr=48000, N=2048: 150 * 2048 / 48000 = 6.4 → 6.
        // 1500 Hz: 1500 * 2048 / 48000 = 64.0 → 64.
        assert_eq!(bin_for_hz(150.0, 48000.0, 2048), 6);
        assert_eq!(bin_for_hz(1500.0, 48000.0, 2048), 64);
    }

    #[test]
    fn band_bin_ends_at_default_settings() {
        let dsp = Dsp::new(2048, 48000.0, 1024);
        assert_eq!(dsp.low_band_bin_end, 6);
        assert_eq!(dsp.mid_band_bin_end, 64);
    }

    #[test]
    fn parseval_band_scale_matches_formula() {
        // parseval_band_scale = 2 / (N · Σ hann²)
        let dsp = Dsp::new(2048, 48000.0, 1024);
        let n = 2048usize;
        let hann_energy: f32 = (0..n)
            .map(|i| {
                let h = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();
                h * h
            })
            .sum();
        let expected = 2.0 / (n as f32 * hann_energy);
        assert!(
            (dsp.parseval_band_scale - expected).abs() < 1e-10,
            "got {}, expected {}",
            dsp.parseval_band_scale,
            expected
        );
    }
}
