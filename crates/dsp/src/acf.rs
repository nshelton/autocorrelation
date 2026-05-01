//! Generalized autocorrelation (Percival & Tzanetakis) + harmonic
//! enhancement + time-domain autocorrelate.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};
use std::sync::Arc;

/// Generalized autocorrelation per Percival & Tzanetakis 2014 §II-B.2:
/// zero-pad input to 2N, forward FFT, magnitude compression `|X|^0.5`,
/// inverse FFT, take the first N/2 lags, normalize so output[0] == 1.0.
/// Allocation-free: caller passes scratch buffers (`time_buf` length 2N,
/// `freq_buf` length N+1) and pre-built `realfft` planners.
///
/// `c = 0.5` (the paper's empirically best choice) gives narrower ACF peaks
/// than `c = 2.0` (regular ACF) — this is what makes downstream peak picking
/// and pulse-train scoring more discriminative.
pub fn compute_gen_acf(
    input: &[f32],
    output: &mut [f32],
    fft_forward: &Arc<dyn RealToComplex<f32>>,
    fft_inverse: &Arc<dyn ComplexToReal<f32>>,
    time_buf: &mut [f32],
    freq_buf: &mut [Complex<f32>],
) {
    let n = input.len();
    debug_assert_eq!(time_buf.len(), 2 * n);
    debug_assert_eq!(freq_buf.len(), n + 1);

    time_buf[..n].copy_from_slice(input);
    time_buf[n..].fill(0.0);

    let _ = fft_forward.process(time_buf, freq_buf);

    for x in freq_buf.iter_mut() {
        let compressed = (x.re * x.re + x.im * x.im).powf(0.25);
        *x = Complex::new(compressed, 0.0);
    }

    let _ = fft_inverse.process(freq_buf, time_buf);

    let zero = time_buf[0].max(1e-12);
    for i in 0..output.len() {
        output[i] = time_buf[i] / zero;
    }
}

/// Per Percival & Tzanetakis 2014 §II-B.3: boost peaks corresponding to
/// integer multiples of the underlying tempo by adding time-stretched
/// versions of the ACF.
pub fn compute_harmonic_enhanced(acf: &[f32], enhanced: &mut [f32]) {
    const HARMONIC_MULTIPLES: [usize; 2] = [2, 4];
    let n = acf.len();
    for tau in 0..n {
        let mut sum = acf[tau];
        for &mult in &HARMONIC_MULTIPLES {
            let idx = tau * mult;
            if idx < n {
                sum += acf[idx];
            }
        }
        enhanced[tau] = sum;
    }
}

/// Direct time-domain autocorrelation, normalized so output[0] == 1.0
/// for any nonzero input. For all-zero input the output is filled with
/// zeros. Caller chooses how many lags to compute via `output.len()`.
pub fn autocorrelate(input: &[f32], output: &mut [f32]) {
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
/// clamped to [1, N/2 - 1] (DC and Nyquist excluded by design).
pub fn bin_for_hz(hz: f32, sample_rate: f32, n: usize) -> usize {
    let bin = (hz * n as f32 / sample_rate).round() as usize;
    bin.clamp(1, n / 2 - 1)
}

/// Default Gaussian σ (in lag bins) for lag-axis smoothing of the
/// harmonic-enhanced ACF. Broadens narrow peaks within each frame so the
/// peak picker locks on the broad average rather than a single spike.
/// Tunable live via `dsp.set_param("acfSmoothingSigma", ...)`. σ = 0
/// disables smoothing.
const ACF_SMOOTHING_SIGMA_DEFAULT: f32 = 2.0;

pub struct AcfState {
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    time_buf: Vec<f32>,
    freq_buf: Vec<Complex<f32>>,
    /// Holds the unsmoothed gen-ACF output. Convolved into `onset_acf` to
    /// produce the smoothed raw ACF, which is then harmonic-enhanced.
    raw_scratch: Vec<f32>,
    /// Pre-normalized half Gaussian kernel (kernel[0] = center weight,
    /// kernel[k] = symmetric weight at offset ±k). Empty when σ ≤ 0.
    kernel: Vec<f32>,
}

impl AcfState {
    pub fn new(rms_history_len: usize) -> Self {
        let n = rms_history_len;
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(2 * n);
        let fft_inverse = planner.plan_fft_inverse(2 * n);
        let onset_acf_len = n / 2;
        let mut state = Self {
            fft_forward,
            fft_inverse,
            time_buf: vec![0.0; 2 * n],
            freq_buf: vec![Complex::new(0.0, 0.0); n + 1],
            raw_scratch: vec![0.0; onset_acf_len],
            kernel: Vec::new(),
        };
        state.set_smoothing_sigma(ACF_SMOOTHING_SIGMA_DEFAULT);
        state
    }

    pub fn set_smoothing_sigma(&mut self, sigma: f32) {
        let s = sigma.clamp(0.0, 100.0);
        self.kernel.clear();
        if s <= 0.0 {
            return;
        }
        let half = (3.0 * s).ceil() as usize;
        let inv_2sig2 = 1.0 / (2.0 * s * s);
        let mut wsum = 0.0f32;
        for k in 0..=half {
            let kf = k as f32;
            let w = (-kf * kf * inv_2sig2).exp();
            self.kernel.push(w);
            wsum += if k == 0 { w } else { 2.0 * w };
        }
        let inv_wsum = 1.0 / wsum;
        for w in self.kernel.iter_mut() {
            *w *= inv_wsum;
        }
    }

    /// Run gen-ACF on `onset` → unsmoothed scratch. Convolve along the lag
    /// axis with a Gaussian kernel → `onset_acf` (the smoothed raw ACF).
    /// Harmonic-enhance `onset_acf` → `onset_acf_enhanced`, which inherits
    /// the smoothing because harmonic enhancement is a sum over input
    /// samples. Edge handling: clamp (replicate edge values).
    pub fn process(
        &mut self,
        onset: &[f32],
        onset_acf: &mut [f32],
        onset_acf_enhanced: &mut [f32],
    ) {
        compute_gen_acf(
            onset,
            &mut self.raw_scratch,
            &self.fft_forward,
            &self.fft_inverse,
            &mut self.time_buf,
            &mut self.freq_buf,
        );
        if self.kernel.is_empty() {
            onset_acf.copy_from_slice(&self.raw_scratch);
        } else {
            let n = onset_acf.len();
            let half = self.kernel.len() - 1;
            for tau in 0..n {
                let mut acc = self.raw_scratch[tau] * self.kernel[0];
                for k in 1..=half {
                    let lo = if tau >= k { tau - k } else { 0 };
                    let hi = if tau + k < n { tau + k } else { n - 1 };
                    acc += (self.raw_scratch[lo] + self.raw_scratch[hi]) * self.kernel[k];
                }
                onset_acf[tau] = acc;
            }
        }
        compute_harmonic_enhanced(onset_acf, onset_acf_enhanced);
    }
}
