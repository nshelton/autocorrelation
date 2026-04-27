use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

const SMOOTHING_ALPHA: f32 = 0.2;
const DB_FLOOR: f32 = -100.0;
const RMS_HISTORY_LEN: usize = 256;

#[wasm_bindgen]
pub struct Dsp {
    waveform: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_buffer: Vec<f32>,
    freq_buffer: Vec<Complex<f32>>,
    spectrum: Vec<f32>,
    hann: Vec<f32>,
    rms_history: Vec<f32>,
}

#[wasm_bindgen]
impl Dsp {
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize) -> Dsp {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let freq_buffer = fft.make_output_vec();
        let spectrum = vec![0.0; freq_buffer.len() - 1]; // drop DC
        let hann = (0..window_size)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos()
            })
            .collect();
        Dsp {
            waveform: vec![0.0; window_size],
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            spectrum,
            hann,
            rms_history: vec![0.0; RMS_HISTORY_LEN],
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
            let mag = (bin.re * bin.re + bin.im * bin.im).sqrt();
            let db = if mag > 0.0 {
                20.0 * mag.log10()
            } else {
                DB_FLOOR
            };
            let clipped = db.clamp(DB_FLOOR, 0.0);
            let normalized = (clipped - DB_FLOOR) / (-DB_FLOOR); // [0, 1]
            self.spectrum[out_i] =
                SMOOTHING_ALPHA * normalized + (1.0 - SMOOTHING_ALPHA) * self.spectrum[out_i];
        }
    }

    pub fn waveform(&self) -> Vec<f32> {
        self.waveform.clone()
    }

    pub fn spectrum(&self) -> Vec<f32> {
        self.spectrum.clone()
    }

    pub fn rms_history(&self) -> Vec<f32> {
        self.rms_history.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_then_waveform_returns_input() {
        let mut dsp = Dsp::new(8);
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        dsp.process(&input);
        assert_eq!(dsp.waveform(), input);
    }

    #[test]
    fn spectrum_has_window_size_div_2_bins() {
        let dsp = Dsp::new(2048);
        assert_eq!(dsp.spectrum().len(), 1024);
    }

    #[test]
    fn silent_input_yields_low_spectrum() {
        let mut dsp = Dsp::new(2048);
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
        let mut dsp = Dsp::new(8);
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
        let mut dsp = Dsp::new(8);
        dsp.process(&vec![0.0_f32; 8]);
        let h = dsp.rms_history();
        assert_eq!(h[h.len() - 1], 0.0);
    }

    #[test]
    fn rms_history_shifts_oldest_out() {
        let mut dsp = Dsp::new(4);
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
        let mut dsp = Dsp::new(2048);
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
}
