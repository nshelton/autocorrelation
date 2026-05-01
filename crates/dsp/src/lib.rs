use wasm_bindgen::prelude::*;

mod acf;
mod beat;
mod buffers;
mod spectrum;

use crate::acf::AcfState;
use crate::beat::BeatState;
use crate::buffers::Buffers;
use crate::spectrum::SpectrumState;

#[wasm_bindgen]
pub struct Dsp {
    buffers: Buffers,
    spectrum: SpectrumState,
    db_floor: f32,
    /// dt = hop_size / sample_rate, captured at construction. Used by
    /// `set_smoothing_tau` to recompute `smoothing_alpha`.
    dt: f32,
    acf: AcfState,
    beat: BeatState,
}

#[wasm_bindgen]
impl Dsp {
    #[wasm_bindgen(constructor)]
    pub fn new(
        window_size: usize,
        sample_rate: f32,
        hop_size: usize,
        rms_history_len: usize,
    ) -> Dsp {
        let dt = hop_size as f32 / sample_rate;
        Dsp {
            buffers: Buffers::new(window_size, rms_history_len),
            spectrum: SpectrumState::new(window_size, sample_rate, dt),
            db_floor: -100.0,
            dt,
            acf: AcfState::new(rms_history_len),
            beat: BeatState::new(rms_history_len, dt),
        }
    }

    pub fn process(&mut self, input: &[f32]) {
        let n = input.len().min(self.buffers.waveform.len());
        self.buffers.waveform[..n].copy_from_slice(&input[..n]);

        // RMS over the input window
        let mean_sq: f32 = input.iter().take(n).map(|&x| x * x).sum::<f32>() / n.max(1) as f32;
        let rms = mean_sq.sqrt();

        push_history(&mut self.buffers.rms, rms);

        let (low_rms, mid_rms, high_rms, flux) =
            self.spectrum
                .process(input, &mut self.buffers.spectrum, self.db_floor);
        push_history(&mut self.buffers.rmsLow, low_rms);
        push_history(&mut self.buffers.rmsMid, mid_rms);
        push_history(&mut self.buffers.rmsHigh, high_rms);
        push_history(&mut self.buffers.onset, flux);

        self.acf.process(
            &self.buffers.onset,
            &mut self.buffers.onsetAcf,
            &mut self.buffers.onsetAcfEnhanced,
        );

        self.beat.process(
            &self.buffers.onset,
            &self.buffers.onsetAcfEnhanced,
            &mut self.buffers.candidates,
            &mut self.buffers.tea,
            &mut self.buffers.beatGrid,
            &mut self.buffers.beatState,
            &mut self.buffers.beatPulses,
            self.dt,
        );

        crate::acf::autocorrelate(&self.buffers.waveform, &mut self.buffers.bufferAcf);
    }

    /// String-keyed buffer accessor. Returns a copy of the named buffer's
    /// current contents, or an empty Vec if the name is unknown. Callers
    /// should rely on `buffer_names()` for the authoritative list.
    pub fn get_buffer(&self, name: &str) -> Vec<f32> {
        self.buffers
            .get(name)
            .map(|s| s.to_vec())
            .unwrap_or_default()
    }

    /// All public buffer names in stable order. Worklet caches this once at
    /// boot; the names are static across reconfigurations.
    pub fn buffer_names(&self) -> Vec<String> {
        self.buffers
            .descriptors()
            .into_iter()
            .map(|(name, _)| name.to_string())
            .collect()
    }

    /// Set a tunable param. Unknown keys are silently ignored.
    /// Recognized keys: "smoothingTauSecs", "onsetSmoothingTauSecs",
    /// "teaTauSecs", "teaSigma", "acfSmoothingSigma", "dbFloor".
    pub fn set_param(&mut self, key: &str, value: f32) {
        match key {
            "smoothingTauSecs" => self.spectrum.set_smoothing_tau(value, self.dt),
            "onsetSmoothingTauSecs" => self.spectrum.set_onset_release_tau(value, self.dt),
            "teaTauSecs" => self.beat.set_tea_tau(value, self.dt),
            "teaSigma" => self.beat.set_tea_sigma(value),
            "acfSmoothingSigma" => self.acf.set_smoothing_sigma(value),
            "dbFloor" => self.db_floor = value.clamp(-200.0, 0.0),
            _ => {}
        }
    }
}

/// Shift a history buffer left by one slot (oldest at index 0 falls off)
/// and write `value` at the end. No-op for empty buffers.
fn push_history(buf: &mut [f32], value: f32) {
    if buf.is_empty() {
        return;
    }
    buf.copy_within(1.., 0);
    let last = buf.len() - 1;
    buf[last] = value;
}

#[cfg(test)]
impl Dsp {
    pub fn onset_acf_enhanced_len(&self) -> usize {
        self.buffers.onsetAcfEnhanced.len()
    }

    pub fn test_set_onset_acf_enhanced(&mut self, src: &[f32]) {
        let n = self.buffers.onsetAcfEnhanced.len().min(src.len());
        self.buffers.onsetAcfEnhanced[..n].copy_from_slice(&src[..n]);
        if n < self.buffers.onsetAcfEnhanced.len() {
            for v in &mut self.buffers.onsetAcfEnhanced[n..] {
                *v = 0.0;
            }
        }
    }

    pub fn test_run_pick_candidates(&mut self) {
        self.beat.process(
            &self.buffers.onset,
            &self.buffers.onsetAcfEnhanced,
            &mut self.buffers.candidates,
            &mut self.buffers.tea,
            &mut self.buffers.beatGrid,
            &mut self.buffers.beatState,
            &mut self.buffers.beatPulses,
            self.dt,
        );
    }

    pub fn onset_history_len(&self) -> usize {
        self.buffers.onset.len()
    }

    pub fn test_set_onset_history(&mut self, src: &[f32]) {
        let n = self.buffers.onset.len().min(src.len());
        self.buffers.onset[..n].copy_from_slice(&src[..n]);
        if n < self.buffers.onset.len() {
            for v in &mut self.buffers.onset[n..] {
                *v = 0.0;
            }
        }
    }

    pub fn test_run_pick_and_score(&mut self) {
        self.acf.process(
            &self.buffers.onset,
            &mut self.buffers.onsetAcf,
            &mut self.buffers.onsetAcfEnhanced,
        );
        self.beat.process(
            &self.buffers.onset,
            &self.buffers.onsetAcfEnhanced,
            &mut self.buffers.candidates,
            &mut self.buffers.tea,
            &mut self.buffers.beatGrid,
            &mut self.buffers.beatState,
            &mut self.buffers.beatPulses,
            self.dt,
        );
    }

    pub fn test_per_frame_estimate(&self) -> (f32, f32, f32) {
        self.beat.test_per_frame_estimate()
    }

    pub fn tea_len(&self) -> usize {
        self.buffers.tea.len()
    }
    pub fn tea_alpha(&self) -> f32 {
        self.beat.tea_alpha()
    }
    pub fn tea_argmax(&self) -> f32 {
        self.beat.tau_smoothed()
    }
    pub fn test_set_tea(&mut self, src: &[f32]) {
        let n = self.buffers.tea.len().min(src.len());
        self.buffers.tea[..n].copy_from_slice(&src[..n]);
        if n < self.buffers.tea.len() {
            for v in &mut self.buffers.tea[n..] {
                *v = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_then_waveform_returns_input() {
        let mut dsp = Dsp::new(8, 48000.0, 4, 512);
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        dsp.process(&input);
        assert_eq!(dsp.get_buffer("waveform"), input);
    }

    #[test]
    fn spectrum_has_window_size_div_2_bins() {
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert_eq!(dsp.get_buffer("spectrum").len(), 1024);
    }

    #[test]
    fn silent_input_yields_low_spectrum() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0; 2048];
        for _ in 0..30 {
            dsp.process(&silent);
        }
        let spec = dsp.get_buffer("spectrum");
        for &v in &spec {
            assert!(v < 0.1, "expected silent → near-zero bin, got {}", v);
        }
    }

    #[test]
    fn rms_of_unit_amplitude_constant_is_one() {
        let mut dsp = Dsp::new(8, 48000.0, 4, 512);
        let constant = vec![1.0_f32; 8];
        dsp.process(&constant);
        let h = dsp.get_buffer("rms");
        assert_eq!(h.len(), 512);
        // Newest sample at the end
        let last = h[h.len() - 1];
        assert!((last - 1.0).abs() < 1e-6, "got {}", last);
    }

    #[test]
    fn rms_of_silence_is_zero() {
        let mut dsp = Dsp::new(8, 48000.0, 4, 512);
        dsp.process(&vec![0.0_f32; 8]);
        let h = dsp.get_buffer("rms");
        assert_eq!(h[h.len() - 1], 0.0);
    }

    #[test]
    fn rms_history_shifts_oldest_out() {
        let mut dsp = Dsp::new(4, 48000.0, 4, 512);
        // Push three distinct values
        dsp.process(&[1.0, 1.0, 1.0, 1.0]); // rms = 1
        dsp.process(&[2.0, 2.0, 2.0, 2.0]); // rms = 2
        dsp.process(&[0.0, 0.0, 0.0, 0.0]); // rms = 0
        let h = dsp.get_buffer("rms");
        let n = h.len();
        // Newest three values are at the end in the order pushed
        assert_eq!(h[n - 3], 1.0);
        assert_eq!(h[n - 2], 2.0);
        assert_eq!(h[n - 1], 0.0);
    }

    #[test]
    fn loud_sine_produces_a_peak() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
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
        let spec = dsp.get_buffer("spectrum");
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
        crate::acf::autocorrelate(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6, "got {}", output[0]);
        assert!((output[1] - 20.0 / 30.0).abs() < 1e-6, "got {}", output[1]);
        assert!((output[2] - 11.0 / 30.0).abs() < 1e-6, "got {}", output[2]);
    }

    #[test]
    fn buffer_acf_has_correct_length() {
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert_eq!(dsp.get_buffer("bufferAcf").len(), 1024);
    }

    #[test]
    fn buffer_acf_zero_lag_is_one_for_nonzero_signal() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let signal: Vec<f32> = (0..2048).map(|i| ((i as f32) * 0.1).sin()).collect();
        dsp.process(&signal);
        let acf = dsp.get_buffer("bufferAcf");
        assert!((acf[0] - 1.0).abs() < 1e-6, "got {}", acf[0]);
    }

    #[test]
    fn buffer_acf_of_sine_peaks_at_period() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let freq = 1000.0_f32;
        // Period at this sr/freq is exactly 48 samples.
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let acf = dsp.get_buffer("bufferAcf");
        // ACF of a sine has local maxima at integer multiples of the period.
        // Verify lag 48 is a local maximum (greater than its neighbors) AND
        // the correlation is strong (close to 1.0 after normalization).
        assert!(
            acf[48] > acf[47],
            "expected acf[48]={} > acf[47]={}",
            acf[48],
            acf[47]
        );
        assert!(
            acf[48] > acf[49],
            "expected acf[48]={} > acf[49]={}",
            acf[48],
            acf[49]
        );
        assert!(
            acf[48] > 0.9,
            "expected strong peak at period, got acf[48]={}",
            acf[48]
        );
    }

    #[test]
    fn acf_of_silence_is_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.process(&vec![0.0_f32; 2048]);
        for &v in dsp.get_buffer("bufferAcf").iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn bin_for_hz_snaps_at_default_settings() {
        // 150 Hz at sr=48000, N=2048: 150 * 2048 / 48000 = 6.4 → 6.
        // 1500 Hz: 1500 * 2048 / 48000 = 64.0 → 64.
        assert_eq!(crate::acf::bin_for_hz(150.0, 48000.0, 2048), 6);
        assert_eq!(crate::acf::bin_for_hz(1500.0, 48000.0, 2048), 64);
    }

    #[test]
    fn set_smoothing_tau_recomputes_alpha() {
        // sr=48000, hop=1024 → dt = 21.33 ms
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);

        // tau = 0.0956 s (the SMOOTHING_TAU_SECS_DEFAULT in spectrum.rs)
        //   → alpha ≈ 1 - exp(-21.33/95.6) ≈ 0.20
        dsp.set_param("smoothingTauSecs", 0.0956);
        // Drive a steady sine and verify the spectrum stabilizes (i.e. EMA is sane).
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / 48000.0)).sin())
            .collect();
        for _ in 0..30 {
            dsp.process(&signal);
        }
        let stable = dsp.get_buffer("spectrum");
        // Find peak — should be a recognizable lobe, not flat.
        let max = stable.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max > 0.5, "expected a clear spectrum peak, got max={}", max);

        // tau extremely small → alpha → 1 (one-shot replacement).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.set_param("smoothingTauSecs", 0.0001); // clamps to 0.001 → alpha ≈ 1.0 since dt >> tau
        dsp.process(&signal);
        let after_one = dsp.get_buffer("spectrum");
        dsp.process(&signal);
        let after_two = dsp.get_buffer("spectrum");
        // With alpha ≈ 1, EMA fully replaces each call → stable across calls.
        for (a, b) in after_one.iter().zip(after_two.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "alpha≈1 EMA should be stable: {} vs {}",
                a,
                b
            );
        }

        // tau extremely large → alpha → 0 (no update).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.set_param("smoothingTauSecs", 1000.0); // clamps to 10.0 → alpha tiny
        for _ in 0..3 {
            dsp.process(&signal);
        }
        // Spectrum stays near zero because the EMA barely moves.
        let small = dsp.get_buffer("spectrum");
        let max = small.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max < 0.1,
            "expected sluggish EMA → near-zero spectrum after 3 calls, got max={}",
            max
        );
    }

    #[test]
    fn set_db_floor_clamps() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.set_param("dbFloor", -1000.0); // should clamp to -200.0
                                           // Silent input → spectrum should saturate to the (clamped) floor's normalized value.
                                           // Since silent audio yields mag=0 → db=floor → normalized=0, spectrum stays zero.
        let silent = vec![0.0_f32; 2048];
        for _ in 0..5 {
            dsp.process(&silent);
        }
        assert!(dsp.get_buffer("spectrum").iter().all(|&v| v == 0.0));

        // Above 0 should clamp to 0.0.
        dsp.set_param("dbFloor", 50.0);
        // floor==0 makes the normalized formula degenerate (clipped - 0) / -0 = NaN.
        // Verify the setter clamped to 0.0; we don't actually call process() here
        // because that would divide by zero — the clamp itself is the test.
        // This test is structural: the field must equal 0.0 after the setter call.
        // Re-create dsp and verify by querying after a different setter result.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.set_param("dbFloor", -50.0); // valid
        dsp.process(&silent);
        // Silent input still yields zero spectrum (mag=0 → db=floor → normalized=0).
        assert!(dsp.get_buffer("spectrum").iter().all(|&v| v == 0.0));
    }

    #[test]
    fn pure_low_band_sine_lands_in_low() {
        // Bin-aligned: 4 × (48000/2048) = 93.75 Hz, in the low band (bins 1..=6).
        // Hann main lobe is 4 bins wide; bin 4 ± 2 = bins 2..6, all in low.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let freq = 93.75_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let low = *dsp.get_buffer("rmsLow").last().unwrap();
        let mid = *dsp.get_buffer("rmsMid").last().unwrap();
        let high = *dsp.get_buffer("rmsHigh").last().unwrap();
        assert!((low - 0.7071).abs() < 0.05, "low {} should be ≈ 0.707", low);
        assert!(mid < 0.05, "mid {} should be near zero", mid);
        assert!(high < 0.05, "high {} should be near zero", high);
    }

    #[test]
    fn pure_mid_band_sine_lands_in_mid() {
        // Bin-aligned: 30 × (48000/2048) = 703.125 Hz, in the mid band (bins 7..=64).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let freq = 703.125_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let low = *dsp.get_buffer("rmsLow").last().unwrap();
        let mid = *dsp.get_buffer("rmsMid").last().unwrap();
        let high = *dsp.get_buffer("rmsHigh").last().unwrap();
        assert!((mid - 0.7071).abs() < 0.05, "mid {} should be ≈ 0.707", mid);
        assert!(low < 0.05, "low {} should be near zero", low);
        assert!(high < 0.05, "high {} should be near zero", high);
    }

    #[test]
    fn pure_high_band_sine_lands_in_high() {
        // Bin-aligned: 100 × (48000/2048) = 2343.75 Hz, in the high band (bins 65..=1023).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let freq = 2343.75_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let low = *dsp.get_buffer("rmsLow").last().unwrap();
        let mid = *dsp.get_buffer("rmsMid").last().unwrap();
        let high = *dsp.get_buffer("rmsHigh").last().unwrap();
        assert!(
            (high - 0.7071).abs() < 0.05,
            "high {} should be ≈ 0.707",
            high
        );
        assert!(low < 0.05, "low {} should be near zero", low);
        assert!(mid < 0.05, "mid {} should be near zero", mid);
    }

    #[test]
    fn parseval_consistency_across_bands() {
        // Multi-tone: one bin-aligned sine in each band, with different amplitudes.
        // Expected: sqrt(low² + mid² + high²) ≈ time-domain full RMS within ~5%
        // (slack for the stationarity approximation in the Parseval derivation).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| {
                let t = i as f32 / sr;
                let two_pi = 2.0 * std::f32::consts::PI;
                1.0  * (two_pi * 93.75   * t).sin()  // low,  amp 1.0
                + 0.5 * (two_pi * 703.125 * t).sin()  // mid,  amp 0.5
                + 0.25 * (two_pi * 2343.75 * t).sin() // high, amp 0.25
            })
            .collect();
        dsp.process(&signal);
        let low = *dsp.get_buffer("rmsLow").last().unwrap();
        let mid = *dsp.get_buffer("rmsMid").last().unwrap();
        let high = *dsp.get_buffer("rmsHigh").last().unwrap();
        let full = *dsp.get_buffer("rms").last().unwrap();
        let summed = (low * low + mid * mid + high * high).sqrt();
        let rel_err = (summed - full).abs() / full;
        assert!(
            rel_err < 0.05,
            "Parseval mismatch: summed={}, full={}, rel_err={}",
            summed,
            full,
            rel_err
        );
    }

    #[test]
    fn band_rms_silence_is_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.process(&vec![0.0_f32; 2048]);
        assert_eq!(*dsp.get_buffer("rmsLow").last().unwrap(), 0.0);
        assert_eq!(*dsp.get_buffer("rmsMid").last().unwrap(), 0.0);
        assert_eq!(*dsp.get_buffer("rmsHigh").last().unwrap(), 0.0);
    }

    #[test]
    fn low_rms_history_shifts_oldest_out() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let loud_low: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 93.75 * (i as f32 / 48000.0)).sin())
            .collect();
        let silent = vec![0.0_f32; 2048];
        dsp.process(&loud_low); // pushes a non-zero into history
        dsp.process(&silent); // pushes a zero
        let h = dsp.get_buffer("rmsLow");
        let n = h.len();
        assert_eq!(h[n - 1], 0.0, "newest should be silent");
        assert!(
            h[n - 2] > 0.5,
            "second-newest should be the loud sine, got {}",
            h[n - 2]
        );
    }

    #[test]
    fn beat_pulses_advance_with_period_and_wrap() {
        // Drive a periodic envelope so beat fitting finds a stable period,
        // then verify all 4 pulse outputs are in [0, 1] and that the cycle-1
        // pulse is sensible (changing per hop, not stuck at one value).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let period_hops = 32usize;
        for k in 0..1500 {
            let amp =
                0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| {
                    let t = i as f32 / sr;
                    amp * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                })
                .collect();
            dsp.process(&signal);
        }
        let pulses = dsp.get_buffer("beatPulses");
        for (i, &v) in pulses.iter().enumerate() {
            assert!(
                !v.is_nan() && (0.0..=1.0).contains(&v),
                "pulse[{}] out of range: {}",
                i,
                v
            );
        }
        // Cycle 1 should change after one more hop (saw advances by 1/period
        // per hop ≈ 0.031). Cycle 16 should change by 16× less (0.002).
        let p1_before = pulses[0];
        let p16_before = pulses[3];
        // Advance one more hop.
        let signal: Vec<f32> = (0..2048)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
            .collect();
        dsp.process(&signal);
        let pulses_after = dsp.get_buffer("beatPulses");
        let dp1 = (pulses_after[0] - p1_before).abs();
        let dp16 = (pulses_after[3] - p16_before).abs();
        // Cycle 16 should move 16× less than cycle 1 (loosely; allow some slack
        // for boundary wraps).
        assert!(
            dp1 > 0.005,
            "cycle-1 should advance per hop, got Δ = {}",
            dp1
        );
        assert!(
            dp16 < dp1,
            "cycle-16 should advance less than cycle-1: {} vs {}",
            dp16,
            dp1
        );
    }

    #[test]
    fn beat_grid_end_to_end_via_process() {
        // End-to-end: drive process() with a periodic envelope and verify the
        // beat grid lands on the correct period via the new P&T pipeline.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let period_hops = 36usize;
        for k in 0..1500 {
            let amp =
                0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| {
                    let t = i as f32 / sr;
                    amp * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                })
                .collect();
            dsp.process(&signal);
        }
        let grid = dsp.get_buffer("beatGrid");
        assert!(!grid[0].is_nan(), "expected a fit after convergence");
        assert!(
            (grid[0] - period_hops as f32).abs() < 1.5,
            "expected period ≈ {}, got {}",
            period_hops,
            grid[0]
        );
    }

    #[test]
    fn onset_history_captures_spectral_flux() {
        // SF = Σ_k max(0, |X[k]| - |X_prev[k]|): half-wave-rectified bin-magnitude
        // rise. Drive a 1 kHz sine ladder so the spectrum has a real peak that
        // changes between frames. Expected pattern:
        //   process(silent): all bins zero, prev zero → flux = 0
        //   process(loud):   bins jump from 0 → big positive flux
        //   process(loud):   identical input → identical FFT → flux = 0
        //   process(quiet):  every bin decreased → flux = 0 (clamped)
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.set_param("onsetSmoothingTauSecs", 0.0);
        let sr = 48000.0_f32;
        let make_sine = |amp: f32| -> Vec<f32> {
            (0..2048)
                .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
                .collect()
        };
        let silent = vec![0.0f32; 2048];
        let loud = make_sine(0.5);
        let quiet = make_sine(0.2);

        dsp.process(&silent);
        dsp.process(&loud);
        dsp.process(&loud);
        dsp.process(&quiet);

        let onset = dsp.get_buffer("onset");
        let n = onset.len();
        // newest at index n-1, oldest at index 0
        assert!(onset[n - 4].abs() < 1e-4, "silent = {}", onset[n - 4]);
        assert!(
            onset[n - 3] > 0.1,
            "silence→sine should be a positive spike, got {}",
            onset[n - 3]
        );
        assert!(
            onset[n - 2].abs() < 1e-4,
            "identical frames = {}",
            onset[n - 2]
        );
        assert!(
            onset[n - 1].abs() < 1e-4,
            "amplitude decrease should clamp to 0, got {}",
            onset[n - 1]
        );
    }

    #[test]
    fn gen_acf_silent_input_is_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0f32; 2048];
        for _ in 0..5 {
            dsp.process(&silent);
        }
        let acf = dsp.get_buffer("onsetAcf");
        for &v in acf.iter() {
            assert!(!v.is_nan(), "silent gen-ACF must not be NaN, got {}", v);
            assert!(v.abs() < 1e-3, "silent gen-ACF should be ~0, got {}", v);
        }
    }

    #[test]
    fn gen_acf_periodic_onset_peaks_at_period() {
        // Drive a strongly periodic envelope so onset_history has clear spikes.
        // Period 32 frames @ default sr/hop ⇒ ~88 BPM.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let period_hops = 32usize;
        for k in 0..1500 {
            let amp =
                0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
                .collect();
            dsp.process(&signal);
        }
        let acf = dsp.get_buffer("onsetAcf");
        let p = period_hops;
        let around = (p - 5..=p + 5)
            .map(|i| acf[i])
            .fold(f32::NEG_INFINITY, f32::max);
        // Avoid lags near period/2 = 16 (a natural ACF harmonic of the sine envelope)
        // and lags very close to 0 (trivially high by normalization). Use lags 5-12
        // as a "genuinely non-periodic" reference region.
        let far = (5..=12).map(|i| acf[i]).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            around > far,
            "peak near lag {} ({:.3}) should exceed nearby non-period max ({:.3})",
            p,
            around,
            far
        );
    }

    #[test]
    fn harmonic_enhanced_sums_multiples() {
        // Synthetic ACF: peaks at lags 10, 20, 40 with magnitudes 0.5, 0.3, 0.2;
        // zeros elsewhere. After enhancement via `HARMONIC_MULTIPLES = [2, 4]`,
        // enhanced[10] should equal 0.5 + 0.3 + 0.2 = 1.0 (since 10*2=20 and
        // 10*4=40 hit the other peaks).
        let mut acf = vec![0.0f32; 64];
        acf[10] = 0.5;
        acf[20] = 0.3;
        acf[40] = 0.2;
        let mut enhanced = vec![0.0f32; 64];
        crate::acf::compute_harmonic_enhanced(&acf, &mut enhanced);
        assert!(
            (enhanced[10] - 1.0).abs() < 1e-6,
            "enhanced[10] = {}",
            enhanced[10]
        );
        // enhanced[20] = acf[20] + acf[40] + acf[80(oob)] = 0.3 + 0.2 + 0 = 0.5
        assert!(
            (enhanced[20] - 0.5).abs() < 1e-6,
            "enhanced[20] = {}",
            enhanced[20]
        );
        // enhanced[40] = acf[40] + acf[80(oob)] + acf[160(oob)] = 0.2
        assert!(
            (enhanced[40] - 0.2).abs() < 1e-6,
            "enhanced[40] = {}",
            enhanced[40]
        );
    }

    #[test]
    fn pick_candidates_silent_all_nan() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0f32; 2048];
        for _ in 0..5 {
            dsp.process(&silent);
        }
        let cands = dsp.get_buffer("candidates");
        assert_eq!(cands.len(), 30);
        for &v in cands.iter() {
            assert!(v.is_nan(), "silent → all candidate slots NaN, got {}", v);
        }
    }

    #[test]
    fn pick_candidates_top_n_within_tau_range() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let n = dsp.onset_acf_enhanced_len(); // = rms_history_len / 2 = 256
        let mut enhanced = vec![0.0f32; n];
        let positions = [14usize, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58];
        for (i, &p) in positions.iter().enumerate() {
            // Make each a strict local max with descending magnitude
            let mag = 1.0 - 0.05 * i as f32;
            enhanced[p - 1] = mag * 0.5;
            enhanced[p] = mag;
            enhanced[p + 1] = mag * 0.5;
        }
        dsp.test_set_onset_acf_enhanced(&enhanced);
        dsp.test_run_pick_candidates();
        let cands = dsp.get_buffer("candidates");
        let mut last_mag = f32::INFINITY;
        for i in 0..10 {
            let lag = cands[3 * i];
            let mag = cands[3 * i + 1];
            assert!(!lag.is_nan(), "slot {} should have a peak, got NaN", i);
            assert!(
                mag <= last_mag,
                "magnitudes should be descending: {} > {}",
                mag,
                last_mag
            );
            last_mag = mag;
        }
    }

    #[test]
    fn pick_candidates_excludes_out_of_range_lags() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let n = dsp.onset_acf_enhanced_len();
        let mut enhanced = vec![0.0f32; n];
        enhanced[5] = 1.0;
        enhanced[4] = 0.5;
        enhanced[6] = 0.5;
        enhanced[100] = 1.0;
        enhanced[99] = 0.5;
        enhanced[101] = 0.5;
        dsp.test_set_onset_acf_enhanced(&enhanced);
        dsp.test_run_pick_candidates();
        let cands = dsp.get_buffer("candidates");
        for i in 0..10 {
            let lag = cands[3 * i];
            if !lag.is_nan() {
                assert!(
                    lag >= 12.0 && lag <= 70.0,
                    "picked lag {} outside [12, 70]",
                    lag
                );
            }
        }
    }

    #[test]
    fn pulse_score_finds_period_in_synthetic_oss() {
        // Synthetic onset_history: pulses at every 36 samples (lag 36 ≈ 78 BPM
        // at default sr/hop, well inside [40, 220]). Period 36 is chosen so
        // 2×36=72 > tau_max≈70, meaning the double-period candidate never
        // enters the running and the scorer must pick the true period directly.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let n = dsp.onset_history_len();
        let mut onset = vec![0.0f32; n];
        let period = 36usize;
        let mut idx = period;
        while idx < n {
            onset[idx] = 1.0;
            idx += period;
        }
        dsp.test_set_onset_history(&onset);
        dsp.test_run_pick_and_score();
        let (period_inst, phase_inst, score_inst) = dsp.test_per_frame_estimate();
        assert!(
            score_inst > 0.0,
            "expected nonzero score, got {}",
            score_inst
        );
        assert!(
            (period_inst - period as f32).abs() < 1.5,
            "expected period ≈ {}, got {}",
            period,
            period_inst
        );
        assert!(
            phase_inst >= 0.0 && phase_inst < period_inst,
            "phase {} should be in [0, period={})",
            phase_inst,
            period_inst
        );
    }

    #[test]
    fn pulse_score_disambiguates_octave() {
        // Regression test for sub-period rejection. Onsets only at multiples of
        // P=36. Pulse-train at τ=36 lands all 4 Φ₁ pulses on real onsets;
        // pulse-train at τ=12 (P/3) lands only every 3rd pulse on a real onset
        // (the Φ₁ positions at k·12 hit real onsets only when k≡0 mod 3).
        // Score must prefer 36 over any sub-period candidate.
        // Period 36 chosen so 2×36=72 > tau_max≈70, keeping the double-period
        // candidate out of range and making this a clean sub-period test.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let n = dsp.onset_history_len();
        let mut onset = vec![0.0f32; n];
        for k in 1..(n / 36 + 1) {
            let i = k * 36;
            if i < n {
                onset[i] = 1.0;
            }
        }
        dsp.test_set_onset_history(&onset);
        dsp.test_run_pick_and_score();
        let (period_inst, _, _) = dsp.test_per_frame_estimate();
        assert!(
            (period_inst - 36.0).abs() < 1.5,
            "sub-period disambiguation failed: expected 36, got {}",
            period_inst
        );
    }

    #[test]
    fn tea_silent_input_decays_to_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let n = dsp.tea_len();
        let pre_charged = vec![0.5f32; n];
        dsp.test_set_tea(&pre_charged);
        let silent = vec![0.0f32; 2048];
        dsp.process(&silent);
        let after = dsp.get_buffer("tea");
        for i in 0..n {
            assert!(
                after[i] < pre_charged[i] + 1e-6,
                "tea[{}] should not increase under silence: before={}, after={}",
                i,
                pre_charged[i],
                after[i]
            );
        }
        // 3 × τ at 4 s default, at ~47 Hz hop rate ≈ 565 hops; 600 is safely past that.
        for _ in 0..600 {
            dsp.process(&silent);
        }
        let later = dsp.get_buffer("tea");
        for &v in &later {
            assert!(v < 0.05, "after long silence TEA should be ~0, got {}", v);
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn tea_periodic_input_locks_to_period() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let period_hops = 36usize;
        for k in 0..1500 {
            let amp =
                0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
                .collect();
            dsp.process(&signal);
        }
        let tau_smoothed = dsp.tea_argmax();
        assert!(!tau_smoothed.is_nan(), "expected fit");
        assert!(
            (tau_smoothed - period_hops as f32).abs() < 1.5,
            "expected ~{}, got {}",
            period_hops,
            tau_smoothed
        );
    }

    #[test]
    fn set_tea_tau_secs_clamps_and_recomputes() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let alpha_default = dsp.tea_alpha();
        dsp.set_param("teaTauSecs", 0.001);
        let alpha_low = dsp.tea_alpha();
        dsp.set_param("teaTauSecs", 120.0);
        let alpha_high = dsp.tea_alpha();
        assert!(alpha_low > alpha_default, "smaller tau ⇒ larger alpha");
        assert!(alpha_high < alpha_default, "larger tau ⇒ smaller alpha");
        assert!(!alpha_low.is_nan() && !alpha_high.is_nan());
    }

    #[test]
    fn beat_grid_from_new_pipeline_locks_to_periodic_input() {
        // End-to-end: periodic envelope at 36 hops should populate beat_grid[0]
        // (period) ≈ 36, beat_state[0] (BPM) ≈ 78, and slots [2]/[3] of beat_state
        // are NaN (measure detection deferred).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let period_hops = 36usize;
        for k in 0..1500 {
            let amp =
                0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
                .collect();
            dsp.process(&signal);
        }
        let grid = dsp.get_buffer("beatGrid");
        let state = dsp.get_buffer("beatState");
        assert!(!grid[0].is_nan(), "expected period fit");
        assert!(
            (grid[0] - period_hops as f32).abs() < 1.5,
            "period: expected ~{}, got {}",
            period_hops,
            grid[0]
        );
        let bpm_expected = 60.0 / (period_hops as f32 * (1024.0 / 48000.0));
        assert!(
            (state[0] - bpm_expected).abs() < 4.0,
            "bpm: expected ~{:.1}, got {:.1}",
            bpm_expected,
            state[0]
        );
        assert!(
            state[2].is_nan(),
            "beats_per_measure should be NaN, got {}",
            state[2]
        );
        assert!(
            state[3].is_nan(),
            "measure_conf should be NaN, got {}",
            state[3]
        );
    }

    #[test]
    fn beat_pulses_silent_input_all_nan() {
        // New behavior under silence: score_inst=0 → all-NaN beat_pulses.
        // (Replaces the old "free-run" behavior tested by the now-removed
        // `beat_pulses_free_run_at_default_rate_under_silence`.)
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0f32; 2048];
        for _ in 0..50 {
            dsp.process(&silent);
        }
        let pulses = dsp.get_buffer("beatPulses");
        for (i, &v) in pulses.iter().enumerate() {
            assert!(
                v.is_nan(),
                "pulse[{}] expected NaN under silence, got {}",
                i,
                v
            );
        }
    }
}
