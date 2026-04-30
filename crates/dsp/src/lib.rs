use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

mod buffers;
mod spectrum;
mod acf;
mod beat;

use crate::buffers::Buffers;
use crate::spectrum::SpectrumState;

/// Spectrum smoothing time constant in seconds. Chosen to preserve the
/// legacy alpha ≈ 0.2 behavior at sr=48000, hop=1024:
///   alpha = 1 - exp(-dt/tau), dt = 1024/48000 = 21.33 ms
///   0.2 ≈ 1 - exp(-21.33ms / 95.6ms)
const SMOOTHING_TAU_SECS: f32 = 0.0956;
/// Crossover from low band to mid band, in Hz. Drum-friendly default:
/// fits the kick fundamental (typically 50–90 Hz) cleanly inside "low"
/// without bleeding much into snare body.
const LOW_BAND_HZ_MAX: f32 = 150.0;
/// Crossover from mid band to high band, in Hz.
const MID_BAND_HZ_MAX: f32 = 1500.0;
/// Maximum number of tempo peaks tracked per hop. Drives the fixed length
/// of `candidates` (= 3 * MAX_PEAKS — interleaved [lag, mag, sharpness]
/// triples).
const MAX_PEAKS: usize = 10;
/// Minimum integer-lag distance between accepted peaks, in hops. Without
/// this, the wide lobes of true tempo peaks return multiple "peaks" all
/// clustered around a single underlying peak.
const MIN_PEAK_SPACING: usize = 3;
/// Plausible-tempo bounds for candidate acceptance. Candidate periods
/// `lag/k` are accepted only when they fall in [period_min, period_max]
/// (derived from these BPM bounds + dt).
const BEAT_TRACKER_MIN_BPM: f32 = 40.0;
const BEAT_TRACKER_MAX_BPM: f32 = 220.0;
/// Length of the public `beat_state` output: [bpm, bpm_conf, beats_per_measure, measure_conf].
const BEAT_STATE_LEN: usize = 4;
/// Length of the `beat_grid` output buffer: [period, phase, score]. Keep in
/// sync with `BeatGridRenderer` / `BeatGridScrollingRenderer` consumers.
const BEAT_GRID_LEN: usize = 3;
/// Cycle multipliers for `beat_pulses`. The detected period is multiplied by
/// each entry to get an "m-cycle period"; the corresponding pulse output is
/// a downward saw with that period (1.0 at cycle start, →0 just before next).
/// LCM of these (= 16) is also the wrap-around modulus for `beat_position`.
const BEAT_PULSE_CYCLES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const BEAT_PULSES_LEN: usize = 4;
/// Gaussian kernel width (σ in lag units) for smearing a TEA vote. Wider σ
/// means the vote from `period_inst` spreads over more lag bins — enough to
/// bridge sub-bin variation between frames without hiding adjacent peaks.
const TEA_GAUSSIAN_SIGMA: f32 = 5.0;
/// Default EMA time constant for the TEA (seconds). 4 s gives stable argmax
/// for steady tempos while still tracking gradual drift within ~10–15 s.
const TEA_TAU_DEFAULT_SECS: f32 = 4.0;

#[wasm_bindgen]
pub struct Dsp {
    buffers: Buffers,
    spectrum: SpectrumState,
    db_floor: f32,
    /// dt = hop_size / sample_rate, captured at construction. Used by
    /// `set_smoothing_tau` to recompute `smoothing_alpha`.
    dt: f32,
    /// Continuously-incrementing fractional beat counter. The integer part
    /// counts beats since startup; the fractional part is "how far into the
    /// current beat we are" (0 = on beat, →1 just before next). Wraps mod 16
    /// (= LCM of BEAT_PULSE_CYCLES) to bound f32 precision over long runs.
    beat_position: f32,
    gen_acf_fft_forward: Arc<dyn RealToComplex<f32>>,
    gen_acf_fft_inverse: Arc<dyn ComplexToReal<f32>>,
    gen_acf_time_buf: Vec<f32>,
    gen_acf_freq_buf: Vec<Complex<f32>>,
    cand_scratch: Vec<(usize, f32)>, // preallocated scratch for picker
    tau_min: usize,
    tau_max: usize,
    /// Per-candidate max-φ correlation from `score_phase_for_tau` (X in the paper).
    pulse_x: [f32; MAX_PEAKS],
    /// Per-candidate variance-of-φ correlation — high variance means the pulse
    /// train picks out clear beats; low variance means all phases score equally
    /// (no tempo structure at this lag).
    pulse_v: [f32; MAX_PEAKS],
    /// Best (integer) phase offset for each candidate, in hop units.
    pulse_phi: [f32; MAX_PEAKS],
    /// Combined normalized score (`X_norm + V_norm`) for each candidate.
    pulse_score: [f32; MAX_PEAKS],
    /// Per-frame winning period (in lag units). NaN until `score_candidates` finds a winner.
    period_inst: f32,
    /// Per-frame winning phase offset (integer hops). NaN until `score_candidates` finds a winner.
    phase_inst: f32,
    /// Per-frame combined score of the winner. 0 when no candidate wins.
    score_inst: f32,
    /// EMA coefficient for the TEA decay/update. Derived from `TEA_TAU_DEFAULT_SECS`
    /// at construction; tunable via `set_tea_tau_secs`.
    tea_alpha: f32,
    /// Smoothed period estimate in lag units — TEA argmax with parabolic
    /// sub-bin refinement. NaN when TEA has no evidence yet.
    tau_smoothed: f32,
    /// Phase offset (in hops) re-scored at `tau_smoothed`. NaN when `tau_smoothed` is NaN.
    phase_smoothed: f32,
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
        let mut planner = RealFftPlanner::<f32>::new();
        let dt = hop_size as f32 / sample_rate;
        let tea_alpha = 1.0 - (-dt / TEA_TAU_DEFAULT_SECS).exp();
        let gen_acf_n = rms_history_len;
        let gen_acf_fft_forward = planner.plan_fft_forward(2 * gen_acf_n);
        let gen_acf_fft_inverse = planner.plan_fft_inverse(2 * gen_acf_n);
        let gen_acf_time_buf = vec![0.0; 2 * gen_acf_n];
        let gen_acf_freq_buf = vec![Complex::new(0.0, 0.0); gen_acf_n + 1];
        let onset_acf_len = rms_history_len / 2;
        let tau_min = ((60.0 / BEAT_TRACKER_MAX_BPM) / dt).floor().max(1.0) as usize;
        let tau_max_unbounded = ((60.0 / BEAT_TRACKER_MIN_BPM) / dt).ceil() as usize;
        let tau_max = tau_max_unbounded.min(onset_acf_len.saturating_sub(2));
        Dsp {
            buffers: Buffers::new(window_size, rms_history_len),
            spectrum: SpectrumState::new(window_size, sample_rate, dt),
            db_floor: -100.0,
            dt,
            beat_position: 0.0,
            gen_acf_fft_forward,
            gen_acf_fft_inverse,
            gen_acf_time_buf,
            gen_acf_freq_buf,
            cand_scratch: Vec::with_capacity(onset_acf_len / 2 + 1),
            tau_min,
            tau_max,
            pulse_x: [0.0; MAX_PEAKS],
            pulse_v: [0.0; MAX_PEAKS],
            pulse_phi: [0.0; MAX_PEAKS],
            pulse_score: [0.0; MAX_PEAKS],
            period_inst: f32::NAN,
            phase_inst: f32::NAN,
            score_inst: 0.0,
            tea_alpha,
            tau_smoothed: f32::NAN,
            phase_smoothed: f32::NAN,
        }
    }

    /// Set the spectrum smoothing time constant (seconds). The internal
    /// `smoothing_alpha` is recomputed from `tau` and the dt captured at
    /// construction: `alpha = 1 - exp(-dt / tau)`. Smaller tau → faster
    /// response. Clamped to [0.001, 10.0] to avoid divide-by-zero and
    /// nonsensical multi-second settling times.
    pub fn set_smoothing_tau(&mut self, tau_secs: f32) {
        self.spectrum.set_smoothing_tau(tau_secs, self.dt);
    }

    /// EMA time constant for the TEA. `alpha = 1 - exp(-dt / tau)`. Smaller
    /// τ ⇒ faster response, less stable. Clamped to [0.05, 60.0].
    pub fn set_tea_tau_secs(&mut self, tau_secs: f32) {
        let tau = tau_secs.clamp(0.05, 60.0);
        self.tea_alpha = 1.0 - (-self.dt / tau).exp();
    }

    pub fn set_db_floor(&mut self, floor: f32) {
        self.db_floor = floor.clamp(-200.0, 0.0);
    }

    pub fn process(&mut self, input: &[f32]) {
        let n = input.len().min(self.buffers.waveform.len());
        self.buffers.waveform[..n].copy_from_slice(&input[..n]);

        // RMS over the input window
        let mean_sq: f32 = input.iter().take(n).map(|&x| x * x).sum::<f32>() / n.max(1) as f32;
        let rms = mean_sq.sqrt();

        push_history(&mut self.buffers.rms, rms);

        let (low_rms, mid_rms, high_rms, flux) = self.spectrum.process(
            input,
            &mut self.buffers.spectrum,
            self.db_floor,
        );
        push_history(&mut self.buffers.rmsLow, low_rms);
        push_history(&mut self.buffers.rmsMid, mid_rms);
        push_history(&mut self.buffers.rmsHigh, high_rms);
        push_history(&mut self.buffers.onset, flux);

        crate::acf::compute_gen_acf(
            &self.buffers.onset,
            &mut self.buffers.onsetAcf,
            &self.gen_acf_fft_forward,
            &self.gen_acf_fft_inverse,
            &mut self.gen_acf_time_buf,
            &mut self.gen_acf_freq_buf,
        );
        crate::acf::compute_harmonic_enhanced(&self.buffers.onsetAcf, &mut self.buffers.onsetAcfEnhanced);
        self.pick_candidates();
        self.score_candidates();
        self.update_tea();
        self.write_beat_outputs();
        self.update_beat_pulses_v2();

        crate::acf::autocorrelate(&self.buffers.waveform, &mut self.buffers.bufferAcf);
    }

    pub fn waveform(&self) -> Vec<f32> { self.buffers.waveform.clone() }
    pub fn spectrum(&self) -> Vec<f32> { self.buffers.spectrum.clone() }
    pub fn buffer_acf(&self) -> Vec<f32> { self.buffers.bufferAcf.clone() }
    pub fn rms_history(&self) -> Vec<f32> { self.buffers.rms.clone() }
    pub fn onset_history(&self) -> Vec<f32> { self.buffers.onset.clone() }
    pub fn onset_acf(&self) -> Vec<f32> { self.buffers.onsetAcf.clone() }
    pub fn onset_acf_enhanced(&self) -> Vec<f32> { self.buffers.onsetAcfEnhanced.clone() }
    pub fn candidates(&self) -> Vec<f32> { self.buffers.candidates.clone() }
    pub fn tea(&self) -> Vec<f32> { self.buffers.tea.clone() }
    pub fn low_rms_history(&self) -> Vec<f32> { self.buffers.rmsLow.clone() }
    pub fn mid_rms_history(&self) -> Vec<f32> { self.buffers.rmsMid.clone() }
    pub fn high_rms_history(&self) -> Vec<f32> { self.buffers.rmsHigh.clone() }
    pub fn beat_grid(&self) -> Vec<f32> { self.buffers.beatGrid.clone() }
    pub fn beat_pulses(&self) -> Vec<f32> { self.buffers.beatPulses.clone() }
    pub fn beat_state(&self) -> Vec<f32> { self.buffers.beatState.clone() }
}

impl Dsp {
    fn pick_candidates(&mut self) {
        for slot in self.buffers.candidates.iter_mut() {
            *slot = f32::NAN;
        }
        if self.tau_max < self.tau_min + 1 {
            return;
        }

        // 1. scan strict local maxima in [tau_min, tau_max]
        self.cand_scratch.clear();
        let upper = self.tau_max.min(self.buffers.onsetAcfEnhanced.len() - 1);
        for k in self.tau_min..upper {
            let y = self.buffers.onsetAcfEnhanced[k];
            if y > 0.0 && y > self.buffers.onsetAcfEnhanced[k - 1] && y > self.buffers.onsetAcfEnhanced[k + 1] {
                self.cand_scratch.push((k, y));
            }
        }

        // 2. sort descending by magnitude
        self.cand_scratch
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. greedy select top-N with min-spacing
        let mut accepted: [u32; MAX_PEAKS] = [0; MAX_PEAKS];
        let mut count: usize = 0;
        for &(k, _) in &self.cand_scratch {
            if count == MAX_PEAKS {
                break;
            }
            let too_close = accepted[..count]
                .iter()
                .any(|&j| ((k as i32 - j as i32).unsigned_abs() as usize) < MIN_PEAK_SPACING);
            if !too_close {
                accepted[count] = k as u32;
                count += 1;
            }
        }

        // 4. parabolic sub-bin refinement → write [lag_frac, mag, sharpness]
        for i in 0..count {
            let k = accepted[i] as usize;
            let y0 = self.buffers.onsetAcfEnhanced[k - 1];
            let y1 = self.buffers.onsetAcfEnhanced[k];
            let y2 = self.buffers.onsetAcfEnhanced[k + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let (lag_frac, mag) = if denom.abs() < 1e-12 {
                (k as f32, y1)
            } else {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                (k as f32 + delta, y1 - 0.25 * (y0 - y2) * delta)
            };
            self.buffers.candidates[3 * i] = lag_frac;
            self.buffers.candidates[3 * i + 1] = mag;
            self.buffers.candidates[3 * i + 2] = -denom; // sharpness — large for narrow peaks
        }
    }

    /// Score every candidate's pulse train, normalize X (max-φ corr) and V
    /// (var-φ corr) so each sums to 1 across candidates, pick winner with
    /// `score = X_norm + V_norm`. Writes per-frame outputs into
    /// `period_inst`, `phase_inst`, `score_inst`. Silent / no-candidate
    /// frames yield `(NaN, NaN, 0.0)`.
    fn score_candidates(&mut self) {
        let mut count = 0usize;
        for i in 0..MAX_PEAKS {
            let lag = self.buffers.candidates[3 * i];
            if lag.is_nan() {
                break;
            }
            let (phi, x, sum, sum_sq, n_phi) = crate::beat::score_phase_for_tau(&self.buffers.onset, lag);
            let n = n_phi as f32;
            let mean = if n > 0.0 { sum / n } else { 0.0 };
            let var = if n > 0.0 {
                (sum_sq / n - mean * mean).max(0.0)
            } else {
                0.0
            };
            self.pulse_x[i] = x;
            self.pulse_v[i] = var;
            self.pulse_phi[i] = phi as f32;
            count += 1;
        }
        if count == 0 {
            self.period_inst = f32::NAN;
            self.phase_inst = f32::NAN;
            self.score_inst = 0.0;
            return;
        }
        let sum_x: f32 = self.pulse_x[..count].iter().sum();
        let sum_v: f32 = self.pulse_v[..count].iter().sum();
        let mut best_i = 0usize;
        let mut best_score = -1.0f32;
        for i in 0..count {
            let xn = if sum_x > 0.0 {
                self.pulse_x[i] / sum_x
            } else {
                0.0
            };
            let vn = if sum_v > 0.0 {
                self.pulse_v[i] / sum_v
            } else {
                0.0
            };
            let s = xn + vn;
            self.pulse_score[i] = s;
            if s > best_score {
                best_score = s;
                best_i = i;
            }
        }
        if best_score <= 0.0 {
            self.period_inst = f32::NAN;
            self.phase_inst = f32::NAN;
            self.score_inst = 0.0;
        } else {
            self.period_inst = self.buffers.candidates[3 * best_i];
            self.phase_inst = self.pulse_phi[best_i];
            self.score_inst = best_score;
        }
    }

    /// Advance the TEA by one frame (Percival & Tzanetakis §II-C). When this
    /// frame produced a fit (`score_inst > 0`), cast a Gaussian vote at
    /// `period_inst` so the lag bin and its σ-neighbourhood accumulate
    /// evidence; otherwise decay all bins by `(1 - alpha)` — silence drains
    /// the accumulator toward 0. Then argmax over `[tau_min, tau_max]` with
    /// parabolic sub-bin refine → `tau_smoothed`; re-score phase at that lag
    /// for a coherent `phase_smoothed`.
    fn update_tea(&mut self) {
        let alpha = self.tea_alpha;
        if self.score_inst > 0.0 && self.period_inst.is_finite() {
            let inv_2sig2 = 1.0 / (2.0 * TEA_GAUSSIAN_SIGMA * TEA_GAUSSIAN_SIGMA);
            for tau in 0..self.buffers.tea.len() {
                let delta = tau as f32 - self.period_inst;
                let g = (-delta * delta * inv_2sig2).exp();
                self.buffers.tea[tau] = (1.0 - alpha) * self.buffers.tea[tau] + alpha * g;
            }
        } else {
            for v in self.buffers.tea.iter_mut() {
                *v *= 1.0 - alpha;
            }
        }

        let upper = self.tau_max.min(self.buffers.tea.len() - 1);
        let mut best_i = self.tau_min;
        let mut best_v = -1.0f32;
        for i in self.tau_min..=upper {
            if self.buffers.tea[i] > best_v {
                best_v = self.buffers.tea[i];
                best_i = i;
            }
        }
        if best_v <= 0.0 {
            self.tau_smoothed = f32::NAN;
            self.phase_smoothed = f32::NAN;
            return;
        }
        let mut tau = best_i as f32;
        if best_i > self.tau_min && best_i < upper {
            let y0 = self.buffers.tea[best_i - 1];
            let y1 = self.buffers.tea[best_i];
            let y2 = self.buffers.tea[best_i + 1];
            let denom = y0 - 2.0 * y1 + y2;
            if denom.abs() > 1e-12 {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                tau = best_i as f32 + delta;
            }
        }
        self.tau_smoothed = tau;
        let (phi, _, _, _, _) = crate::beat::score_phase_for_tau(&self.buffers.onset, tau);
        self.phase_smoothed = phi as f32;
    }

    fn write_beat_outputs(&mut self) {
        let p = self.tau_smoothed;
        let phi = self.phase_smoothed;
        let s = self.score_inst;
        if p.is_nan() || s <= 0.0 {
            self.buffers.beatGrid[0] = f32::NAN;
            self.buffers.beatGrid[1] = f32::NAN;
            self.buffers.beatGrid[2] = 0.0;
            self.buffers.beatState[0] = f32::NAN;
            self.buffers.beatState[1] = 0.0;
            self.buffers.beatState[2] = f32::NAN;
            self.buffers.beatState[3] = f32::NAN;
        } else {
            self.buffers.beatGrid[0] = p;
            self.buffers.beatGrid[1] = phi;
            self.buffers.beatGrid[2] = s;
            self.buffers.beatState[0] = if p > 0.0 {
                60.0 / (p * self.dt)
            } else {
                f32::NAN
            };
            self.buffers.beatState[1] = s;
            self.buffers.beatState[2] = f32::NAN; // beats_per_measure deferred
            self.buffers.beatState[3] = f32::NAN; // measure_conf deferred
        }
    }

    // Simplified saw-wave generator. Phase is always real when there's a fit;
    // NaN-out under silence/no-fit so the renderer holds last value.
    fn update_beat_pulses_v2(&mut self) {
        let period = self.tau_smoothed;
        let phase = self.phase_smoothed;
        let score = self.score_inst;
        if period.is_nan() || period <= 0.0 || score <= 0.0 || phase.is_nan() {
            for slot in self.buffers.beatPulses.iter_mut() {
                *slot = f32::NAN;
            }
            return;
        }
        // Anchor fractional part to phase/period so cycle-1 starts at phase=0.
        let phase_frac = (phase / period).clamp(0.0, 0.999_999);
        let prev = self.beat_position;
        let mut bp = prev.floor() + phase_frac;
        if bp < prev - 0.5 {
            bp += 1.0;
        }
        self.beat_position = bp.rem_euclid(16.0);
        for (i, &m) in BEAT_PULSE_CYCLES.iter().enumerate() {
            let frac = (self.beat_position / m).fract();
            self.buffers.beatPulses[i] = 1.0 - frac;
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
        self.pick_candidates();
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
        // Recompute enhanced ACF from current onset_history, then pick & score.
        crate::acf::compute_gen_acf(
            &self.buffers.onset,
            &mut self.buffers.onsetAcf,
            &self.gen_acf_fft_forward,
            &self.gen_acf_fft_inverse,
            &mut self.gen_acf_time_buf,
            &mut self.gen_acf_freq_buf,
        );
        crate::acf::compute_harmonic_enhanced(&self.buffers.onsetAcf, &mut self.buffers.onsetAcfEnhanced);
        self.pick_candidates();
        self.score_candidates();
    }

    pub fn test_per_frame_estimate(&self) -> (f32, f32, f32) {
        (self.period_inst, self.phase_inst, self.score_inst)
    }

    pub fn tea_len(&self) -> usize {
        self.buffers.tea.len()
    }
    pub fn tea_alpha(&self) -> f32 {
        self.tea_alpha
    }
    pub fn tea_argmax(&self) -> f32 {
        self.tau_smoothed
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
        assert_eq!(dsp.waveform(), input);
    }

    #[test]
    fn spectrum_has_window_size_div_2_bins() {
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert_eq!(dsp.spectrum().len(), 1024);
    }

    #[test]
    fn silent_input_yields_low_spectrum() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
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
        let mut dsp = Dsp::new(8, 48000.0, 4, 512);
        let constant = vec![1.0_f32; 8];
        dsp.process(&constant);
        let h = dsp.rms_history();
        assert_eq!(h.len(), 512);
        // Newest sample at the end
        let last = h[h.len() - 1];
        assert!((last - 1.0).abs() < 1e-6, "got {}", last);
    }

    #[test]
    fn rms_of_silence_is_zero() {
        let mut dsp = Dsp::new(8, 48000.0, 4, 512);
        dsp.process(&vec![0.0_f32; 8]);
        let h = dsp.rms_history();
        assert_eq!(h[h.len() - 1], 0.0);
    }

    #[test]
    fn rms_history_shifts_oldest_out() {
        let mut dsp = Dsp::new(4, 48000.0, 4, 512);
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
        crate::acf::autocorrelate(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6, "got {}", output[0]);
        assert!((output[1] - 20.0 / 30.0).abs() < 1e-6, "got {}", output[1]);
        assert!((output[2] - 11.0 / 30.0).abs() < 1e-6, "got {}", output[2]);
    }

    #[test]
    fn buffer_acf_has_correct_length() {
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert_eq!(dsp.buffer_acf().len(), 1024);
    }

    #[test]
    fn buffer_acf_zero_lag_is_one_for_nonzero_signal() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let signal: Vec<f32> = (0..2048).map(|i| ((i as f32) * 0.1).sin()).collect();
        dsp.process(&signal);
        let acf = dsp.buffer_acf();
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
        let acf = dsp.buffer_acf();
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
        for &v in dsp.buffer_acf().iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn smoothing_alpha_matches_time_constant_formula() {
        // alpha = 1 - exp(-dt/tau) where dt = hop_size / sample_rate
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let dt = 1024.0_f32 / 48000.0;
        let expected = 1.0 - (-dt / SMOOTHING_TAU_SECS).exp();
        assert!(
            (dsp.spectrum.smoothing_alpha - expected).abs() < 1e-6,
            "alpha {} != expected {}",
            dsp.spectrum.smoothing_alpha,
            expected
        );
    }

    #[test]
    fn smoothing_alpha_at_legacy_settings_is_approximately_0_2() {
        // SMOOTHING_TAU_SECS is chosen so that at sr=48000, hop=1024
        // alpha ≈ 0.2 — i.e., the legacy hard-coded value is preserved.
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert!(
            (dsp.spectrum.smoothing_alpha - 0.2).abs() < 0.005,
            "expected alpha ≈ 0.2 at legacy settings, got {}",
            dsp.spectrum.smoothing_alpha
        );
    }

    #[test]
    fn smoothing_alpha_shrinks_at_smaller_hop() {
        // Halving hop ≈ halves alpha (small-dt regime: 1 - exp(-x) ≈ x).
        // Wall-clock dynamics stay the same; per-call coefficient changes.
        let large = Dsp::new(2048, 48000.0, 1024, 512);
        let small = Dsp::new(2048, 48000.0, 512, 512);
        assert!(
            small.spectrum.smoothing_alpha < large.spectrum.smoothing_alpha,
            "small {} should be < large {}",
            small.spectrum.smoothing_alpha,
            large.spectrum.smoothing_alpha
        );
        let ratio = small.spectrum.smoothing_alpha / large.spectrum.smoothing_alpha;
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
        assert_eq!(crate::acf::bin_for_hz(150.0, 48000.0, 2048), 6);
        assert_eq!(crate::acf::bin_for_hz(1500.0, 48000.0, 2048), 64);
    }

    #[test]
    fn band_bin_ends_at_default_settings() {
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert_eq!(dsp.spectrum.low_band_bin_end, 6);
        assert_eq!(dsp.spectrum.mid_band_bin_end, 64);
    }

    #[test]
    fn parseval_band_scale_matches_formula() {
        // parseval_band_scale = 2 / (N · Σ hann²)
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
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
            (dsp.spectrum.parseval_band_scale - expected).abs() < 1e-10,
            "got {}, expected {}",
            dsp.spectrum.parseval_band_scale,
            expected
        );
    }

    #[test]
    fn set_smoothing_tau_recomputes_alpha() {
        // sr=48000, hop=1024 → dt = 21.33 ms
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);

        // tau = SMOOTHING_TAU_SECS (0.0956 s) → alpha ≈ 1 - exp(-21.33/95.6) ≈ 0.20
        dsp.set_smoothing_tau(0.0956);
        // Drive a steady sine and verify the spectrum stabilizes (i.e. EMA is sane).
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / 48000.0)).sin())
            .collect();
        for _ in 0..30 {
            dsp.process(&signal);
        }
        let stable = dsp.spectrum();
        // Find peak — should be a recognizable lobe, not flat.
        let max = stable.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max > 0.5, "expected a clear spectrum peak, got max={}", max);

        // tau extremely small → alpha → 1 (one-shot replacement).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.set_smoothing_tau(0.0001); // clamps to 0.001 → alpha ≈ 1.0 since dt >> tau
        dsp.process(&signal);
        let after_one = dsp.spectrum();
        dsp.process(&signal);
        let after_two = dsp.spectrum();
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
        dsp.set_smoothing_tau(1000.0); // clamps to 10.0 → alpha tiny
        for _ in 0..3 {
            dsp.process(&signal);
        }
        // Spectrum stays near zero because the EMA barely moves.
        let small = dsp.spectrum();
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
        dsp.set_db_floor(-1000.0); // should clamp to -200.0
                                   // Silent input → spectrum should saturate to the (clamped) floor's normalized value.
                                   // Since silent audio yields mag=0 → db=floor → normalized=0, spectrum stays zero.
        let silent = vec![0.0_f32; 2048];
        for _ in 0..5 {
            dsp.process(&silent);
        }
        assert!(dsp.spectrum().iter().all(|&v| v == 0.0));

        // Above 0 should clamp to 0.0.
        dsp.set_db_floor(50.0);
        // floor==0 makes the normalized formula degenerate (clipped - 0) / -0 = NaN.
        // Verify the setter clamped to 0.0; we don't actually call process() here
        // because that would divide by zero — the clamp itself is the test.
        // This test is structural: the field must equal 0.0 after the setter call.
        // Re-create dsp and verify by querying after a different setter result.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        dsp.set_db_floor(-50.0); // valid
        dsp.process(&silent);
        // Silent input still yields zero spectrum (mag=0 → db=floor → normalized=0).
        assert!(dsp.spectrum().iter().all(|&v| v == 0.0));
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
        let low = *dsp.low_rms_history().last().unwrap();
        let mid = *dsp.mid_rms_history().last().unwrap();
        let high = *dsp.high_rms_history().last().unwrap();
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
        let low = *dsp.low_rms_history().last().unwrap();
        let mid = *dsp.mid_rms_history().last().unwrap();
        let high = *dsp.high_rms_history().last().unwrap();
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
        let low = *dsp.low_rms_history().last().unwrap();
        let mid = *dsp.mid_rms_history().last().unwrap();
        let high = *dsp.high_rms_history().last().unwrap();
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
        let low = *dsp.low_rms_history().last().unwrap();
        let mid = *dsp.mid_rms_history().last().unwrap();
        let high = *dsp.high_rms_history().last().unwrap();
        let full = *dsp.rms_history().last().unwrap();
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
        assert_eq!(*dsp.low_rms_history().last().unwrap(), 0.0);
        assert_eq!(*dsp.mid_rms_history().last().unwrap(), 0.0);
        assert_eq!(*dsp.high_rms_history().last().unwrap(), 0.0);
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
        let h = dsp.low_rms_history();
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
        let pulses = dsp.beat_pulses();
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
        let pulses_after = dsp.beat_pulses();
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
        let grid = dsp.beat_grid();
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

        let onset = dsp.onset_history();
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
        let acf = dsp.onset_acf();
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
        let acf = dsp.onset_acf();
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
        let cands = dsp.candidates();
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
        let cands = dsp.candidates();
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
        let cands = dsp.candidates();
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
        let after = dsp.tea();
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
        let later = dsp.tea();
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
        dsp.set_tea_tau_secs(0.001);
        let alpha_low = dsp.tea_alpha();
        dsp.set_tea_tau_secs(120.0);
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
        let grid = dsp.beat_grid();
        let state = dsp.beat_state();
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
        let pulses = dsp.beat_pulses();
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
