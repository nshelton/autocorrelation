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
