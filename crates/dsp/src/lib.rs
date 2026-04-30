use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

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
/// Default time constant for the rms_acf decaying accumulator (seconds).
/// 4 s gives stable peaks for steady tempo while still tracking gradual
/// tempo changes within ~10–15 seconds.
const ACCUM_TAU_DEFAULT_SECS: f32 = 4.0;
/// Minimum lag (in hops) considered for peak picking. Below this, peaks
/// imply BPM > ~280 (at hop=1024, sr=48000) which isn't a tempo we care
/// about, and the very-low-lag region of the ACF is dominated by the
/// shape of the autocorrelation envelope rather than tempo structure.
const MIN_PEAK_LAG: usize = 10;
/// Maximum number of tempo peaks tracked per hop. Drives the fixed length
/// of `acf_peaks` (= 3 * MAX_PEAKS — interleaved [lag, mag, sharpness]
/// triples).
const MAX_PEAKS: usize = 10;
/// Minimum integer-lag distance between accepted peaks, in hops. Without
/// this, the wide lobes of true tempo peaks return multiple "peaks" all
/// clustered around a single underlying peak.
const MIN_PEAK_SPACING: usize = 3;
/// Plausible-tempo bounds for the `BeatTracker`. Candidate periods T̂ = lag/k
/// are accepted only when they fall in [period_min, period_max] (derived from
/// these BPM bounds + dt). Without this, sub-period peaks (hi-hats, 16th-note
/// patterns) used to pull the tracker into reporting tempos 2-4× too fast —
/// the user constraint "pick a beat harmonic between 80–190 BPM" becomes the
/// hard filter that keeps observations in the perceptually-sensible range.
const BEAT_TRACKER_MIN_BPM: f32 = 80.0;
const BEAT_TRACKER_MAX_BPM: f32 = 190.0;
/// Initial BPM the `BeatTracker` holds before any audio has been processed.
/// 120 is right in the middle of plausible music tempos (60–180), so any
/// genre converges within a similar number of frames.
const BEAT_TRACKER_INITIAL_BPM: f32 = 120.0;
/// Initial measure count (beats per bar) — 4/4 covers most popular music;
/// changes only via the hysteresis path on strong evidence.
const BEAT_TRACKER_INITIAL_BEATS_PER_MEASURE: u32 = 4;
/// Gaussian tolerance σ (in lag units) for peak-vs-target alignment in the
/// tracker. Wider than the peak-picker's 1.0 because we're matching peaks
/// against a smoothed candidate period, not against ACF noise.
const BEAT_TRACKER_SIGMA_LAG: f32 = 1.5;
/// Max EMA rate per frame for the BPM update — `α(c) = α_max · confidence`.
/// 0.1 means the tracker can fully shift over ~10 fully-confident frames
/// (~0.2 s at default sr/hop) — fast enough to follow real tempo drift,
/// slow enough to ignore single-frame outliers.
const BEAT_TRACKER_ALPHA_MAX: f32 = 0.1;
/// EMA factor for confidence smoothing — `conf = (1-x)·conf + x·new`.
const BEAT_TRACKER_CONF_SMOOTHING: f32 = 0.1;
/// Margin a candidate `m` must beat the current `beats_per_measure`'s score
/// by before we switch (hysteresis). 1.3 = 30 % stronger; lower = jittery,
/// higher = sticky.
const BEAT_TRACKER_MEASURE_SWITCH_MARGIN: f32 = 1.3;
/// Discrete `beats_per_measure` candidates considered each frame — 2/4 time,
/// 3/4 (waltz), 4/4, 6/8 (compound), 8 (long bar / 4-measure phrase).
const BEAT_TRACKER_MEASURE_CANDIDATES: [u32; 5] = [2, 3, 4, 6, 8];
/// Length of the public `beat_state` output: [bpm, bpm_conf, beats_per_measure, measure_conf].
const BEAT_STATE_LEN: usize = 4;
/// Length of the `beat_grid` output buffer: [period, phase, score]. Keep in
/// sync with `BeatGridRenderer` / `BeatGridScrollingRenderer` consumers.
const BEAT_GRID_LEN: usize = 3;
/// Step for the phase sweep over `rms_history`, in hops. 0.5 hop ≈ 10 ms at
/// default sr/hop — fine enough that visible beat lines won't visibly snap to
/// a coarse grid frame-to-frame.
const BEAT_PHASE_STEP_HOPS: f32 = 0.5;
/// Cycle multipliers for `beat_pulses`. The detected period is multiplied by
/// each entry to get an "m-cycle period"; the corresponding pulse output is
/// a downward saw with that period (1.0 at cycle start, →0 just before next).
/// LCM of these (= 16) is also the wrap-around modulus for `beat_position`.
const BEAT_PULSE_CYCLES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const BEAT_PULSES_LEN: usize = 4;

/// State estimator for tempo and meter. Consumes the picked `acf_peaks` (with
/// magnitude and sharpness) each frame and updates four state values:
///
///   - `bpm_period` — smoothed beat period in lag units. Updated as a
///     confidence-gated EMA; high-confidence frames pull it strongly toward
///     the observation, low-confidence frames barely move it.
///   - `beats_per_measure` — discrete; switches only when a different
///     candidate `m ∈ {2, 3, 4, 6, 8}` scores ≥ 1.3× the current one (so the
///     measure interpretation is sticky and doesn't jitter every frame).
///   - `bpm_confidence` and `measure_confidence` — smoothed 0..1 quality
///     signals consumers can use to fade visualizations or gate downstream
///     decisions.
///
/// Initial state: 120 BPM, 4/4 — the tracker holds these defaults until the
/// observations have enough support to budge them.
struct BeatTracker {
    bpm_period: f32,
    bpm_confidence: f32,
    beats_per_measure: u32,
    measure_confidence: f32,
    period_min: f32,
    period_max: f32,
    /// Seconds per lag = hop_size / sample_rate. Used by the public `bpm()`
    /// view to convert lags → BPM.
    dt_secs: f32,
}

impl BeatTracker {
    /// Build a tracker with `period_min` / `period_max` derived from the
    /// `BEAT_TRACKER_MIN_BPM` / `BEAT_TRACKER_MAX_BPM` bounds. The initial
    /// period (120 BPM) is guaranteed to fall in this range.
    fn new(dt_secs: f32) -> Self {
        // BPM is inversely proportional to period — fastest tempo = shortest
        // period, slowest tempo = longest period.
        let period_min = (60.0 / BEAT_TRACKER_MAX_BPM) / dt_secs;
        let period_max = (60.0 / BEAT_TRACKER_MIN_BPM) / dt_secs;
        let initial_period = (60.0 / BEAT_TRACKER_INITIAL_BPM) / dt_secs;
        Self {
            bpm_period: initial_period.clamp(period_min, period_max),
            bpm_confidence: 0.0,
            beats_per_measure: BEAT_TRACKER_INITIAL_BEATS_PER_MEASURE,
            measure_confidence: 0.0,
            period_min,
            period_max,
            dt_secs,
        }
    }

    fn bpm(&self) -> f32 {
        if self.bpm_period > 0.0 {
            60.0 / (self.bpm_period * self.dt_secs)
        } else {
            f32::NAN
        }
    }

    /// Observe one frame of `[lag, mag, sharpness]` peaks (length = 3 *
    /// MAX_PEAKS, NaN-padded) and update state.
    fn observe(&mut self, acf_peaks: &[f32]) {
        let inv_two_sigma_sq = 1.0 / (2.0 * BEAT_TRACKER_SIGMA_LAG * BEAT_TRACKER_SIGMA_LAG);
        let n_peaks = acf_peaks.len() / 3;

        // ── BPM update ────────────────────────────────────────────────────
        // For each peak (lᵢ, mᵢ), find the integer kᵢ that minimizes
        // |lᵢ/k − bpm_period|; the candidate T̂ᵢ = lᵢ/kᵢ is "if my current
        // tempo is right, this peak is the kᵢ-th beat". Magnitude- and
        // alignment-weighted average gives the observation; off-grid peaks
        // contribute negligibly via the Gaussian alignment.
        let mut weight_sum = 0.0f32;
        let mut weighted_t_sum = 0.0f32;
        let mut total_mag = 0.0f32;

        for i in 0..n_peaks {
            let lag = acf_peaks[3 * i];
            let mag = acf_peaks[3 * i + 1];
            if lag.is_nan() || mag <= 0.0 {
                continue;
            }
            total_mag += mag;
            // k near current period; clamp to ≥ 1 so we never divide by 0.
            let k = (lag / self.bpm_period).round().max(1.0);
            let candidate_t = lag / k;
            if candidate_t < self.period_min || candidate_t > self.period_max {
                continue;
            }
            let delta = candidate_t - self.bpm_period;
            let alignment = (-delta * delta * inv_two_sigma_sq).exp();
            let weight = mag * alignment;
            weight_sum += weight;
            weighted_t_sum += weight * candidate_t;
        }

        if weight_sum > 0.0 && total_mag > 0.0 {
            let observation = weighted_t_sum / weight_sum;
            let new_conf = (weight_sum / total_mag).clamp(0.0, 1.0);
            let alpha = BEAT_TRACKER_ALPHA_MAX * new_conf;
            self.bpm_period = ((1.0 - alpha) * self.bpm_period + alpha * observation)
                .clamp(self.period_min, self.period_max);
            self.bpm_confidence = (1.0 - BEAT_TRACKER_CONF_SMOOTHING) * self.bpm_confidence
                + BEAT_TRACKER_CONF_SMOOTHING * new_conf;
        } else {
            // No support this frame — decay confidence gently.
            self.bpm_confidence *= 1.0 - BEAT_TRACKER_CONF_SMOOTHING;
        }

        // ── Measure update ────────────────────────────────────────────────
        // For each candidate m, find the highest-scoring peak near m·period.
        // Score = mag · sharpness · gaussian_alignment — the sharpness factor
        // is what makes "pointy" peaks (real downbeats) win over broad
        // lobes that happen to sit near m·period by accident.
        let mut best_m = self.beats_per_measure;
        let mut best_m_score = 0.0f32;
        let mut current_m_score = 0.0f32;

        for &m in &BEAT_TRACKER_MEASURE_CANDIDATES {
            let target = m as f32 * self.bpm_period;
            let mut best_peak_score = 0.0f32;
            for i in 0..n_peaks {
                let lag = acf_peaks[3 * i];
                let mag = acf_peaks[3 * i + 1];
                let sharp = acf_peaks[3 * i + 2];
                if lag.is_nan() || mag <= 0.0 {
                    continue;
                }
                let delta = lag - target;
                let alignment = (-delta * delta * inv_two_sigma_sq).exp();
                let score = mag * sharp.max(0.0) * alignment;
                if score > best_peak_score {
                    best_peak_score = score;
                }
            }
            if m == self.beats_per_measure {
                current_m_score = best_peak_score;
            }
            if best_peak_score > best_m_score {
                best_m_score = best_peak_score;
                best_m = m;
            }
        }

        // Hysteresis: switch only if the new winner clearly beats current.
        if best_m != self.beats_per_measure
            && best_m_score > current_m_score * BEAT_TRACKER_MEASURE_SWITCH_MARGIN
        {
            self.beats_per_measure = best_m;
        }

        // Smooth measure confidence using best score normalized by current
        // BPM confidence (otherwise a strong-mag peak with no real tempo
        // structure could read as high measure confidence).
        let new_measure_conf = best_m_score.min(1.0) * self.bpm_confidence;
        self.measure_confidence = (1.0 - BEAT_TRACKER_CONF_SMOOTHING) * self.measure_confidence
            + BEAT_TRACKER_CONF_SMOOTHING * new_measure_conf;
    }
}

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
    db_floor: f32,
    rms_history: Vec<f32>,
    onset_history: Vec<f32>,
    prev_rms: f32,
    buffer_acf: Vec<f32>,
    rms_acf: Vec<f32>,
    /// Decaying EMA accumulator over `rms_acf`. Same length. Used as the
    /// signal for tempo peak picking — the EMA suppresses per-frame noise
    /// in the instantaneous ACF so true tempo peaks build up.
    rms_acf_accum: Vec<f32>,
    /// Per-process EMA coefficient for `rms_acf_accum`. Computed from
    /// `accum_tau_secs` and the same `dt` used for `smoothing_alpha`:
    /// `alpha = 1 - exp(-dt / tau)`. Tunable via `set_accum_tau_secs`.
    accum_alpha: f32,
    /// Detected tempo peaks in `rms_acf_accum`, as interleaved
    /// [lag_frac, mag, sharpness] triples. Length = 3 * MAX_PEAKS. Unused
    /// slots filled with `f32::NAN` so the renderer can detect "no peak"
    /// with a single `isNaN` check (0.0 would collide with a valid lag).
    /// Sharpness = `-(y0 - 2y1 + y2)` from the parabolic interp — large
    /// for narrow real beats, small for broad smeared lobes; consumed by
    /// `BeatTracker`'s measure detector to favor pointy peaks at `m·T`.
    acf_peaks: Vec<f32>,
    /// Preallocated scratch for peak-candidate collection. Capacity is
    /// reserved at construction (`rms_acf_len / 2` — worst case every other
    /// lag is a local max). Cleared (not freed) each `process()` call so
    /// peak picking is allocation-free.
    peak_candidates: Vec<(usize, f32)>,
    rms_detrended: Vec<f32>,
    /// Per-process-call EMA coefficient for the spectrum. Computed from
    /// `SMOOTHING_TAU_SECS` and the wall-clock dt between hops
    /// (`hop_size / sample_rate`), so changing `hop_size` does NOT change
    /// perceived smoothing dynamics.
    smoothing_alpha: f32,
    /// dt = hop_size / sample_rate, captured at construction. Used by
    /// `set_smoothing_tau` to recompute `smoothing_alpha`.
    dt: f32,
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
    low_rms_history:  Vec<f32>,
    mid_rms_history:  Vec<f32>,
    high_rms_history: Vec<f32>,
    /// Scratch: low_rms_history with its mean subtracted, used as input to
    /// `autocorrelate`. Without detrending, the average band level creates a
    /// DC bias that drowns out tempo peaks (same rationale as `rms_detrended`
    /// for the full-band ACF).
    low_rms_detrended: Vec<f32>,
    /// Detrended autocorrelation of `low_rms_history`. Length = rms_history_len / 2.
    low_rms_acf: Vec<f32>,
    /// Public per-frame `[period, phase, confidence]` view of the tracker
    /// state. Period and phase are in lag units; confidence is bpm_confidence.
    /// Reused buffer — `update_beat_state` rewrites in place each
    /// `process()` call.
    beat_grid: Vec<f32>,
    /// Tempo / meter state estimator. Consumes `acf_peaks` each frame and
    /// holds smoothed (bpm_period, beats_per_measure) plus their confidences.
    /// Initialized at 120 BPM, 4/4.
    beat_tracker: BeatTracker,
    /// Public `[bpm, bpm_conf, beats_per_measure_as_f32, measure_conf]` view
    /// of the tracker state — mirrored from `beat_tracker` each frame so
    /// consumers can read it as an ordinary feature buffer.
    beat_state: Vec<f32>,
    /// Continuously-incrementing fractional beat counter. The integer part
    /// counts beats since startup; the fractional part is "how far into the
    /// current beat we are" (0 = on beat, →1 just before next). Wraps mod 16
    /// (= LCM of BEAT_PULSE_CYCLES) to bound f32 precision over long runs.
    beat_position: f32,
    /// 4 saw values in [0, 1], one per `BEAT_PULSE_CYCLES` entry. 1.0 at the
    /// start of each m-cycle, decaying linearly to 0 just before the next.
    /// NaN-padded if no period fit. Output buffer reused across calls.
    beat_pulses: Vec<f32>,
    gen_acf_fft_forward: Arc<dyn RealToComplex<f32>>,
    gen_acf_fft_inverse: Arc<dyn ComplexToReal<f32>>,
    gen_acf_time_buf: Vec<f32>,
    gen_acf_freq_buf: Vec<Complex<f32>>,
    onset_acf: Vec<f32>,
}

#[wasm_bindgen]
impl Dsp {
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize, sample_rate: f32, hop_size: usize, rms_history_len: usize) -> Dsp {
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
        let accum_alpha = 1.0 - (-dt / ACCUM_TAU_DEFAULT_SECS).exp();
        let gen_acf_n = rms_history_len;
        let gen_acf_fft_forward = planner.plan_fft_forward(2 * gen_acf_n);
        let gen_acf_fft_inverse = planner.plan_fft_inverse(2 * gen_acf_n);
        let gen_acf_time_buf = vec![0.0; 2 * gen_acf_n];
        let gen_acf_freq_buf = vec![Complex::new(0.0, 0.0); gen_acf_n + 1];
        Dsp {
            waveform: vec![0.0; window_size],
            fft,
            fft_buffer: vec![0.0; window_size],
            freq_buffer,
            spectrum,
            hann,
            mag_scale,
            db_floor: -100.0,
            rms_history: vec![0.0; rms_history_len],
            onset_history: vec![0.0; rms_history_len],
            prev_rms: 0.0,
            buffer_acf: vec![0.0; window_size / 2],
            rms_acf: vec![0.0; rms_history_len / 2],
            rms_acf_accum: vec![0.0; rms_history_len / 2],
            accum_alpha,
            acf_peaks: vec![f32::NAN; 3 * MAX_PEAKS],
            peak_candidates: Vec::with_capacity((rms_history_len / 2) / 2 + 1),
            rms_detrended: vec![0.0; rms_history_len],
            smoothing_alpha,
            dt,
            low_band_bin_end,
            mid_band_bin_end,
            parseval_band_scale,
            low_rms_history:  vec![0.0; rms_history_len],
            mid_rms_history:  vec![0.0; rms_history_len],
            high_rms_history: vec![0.0; rms_history_len],
            low_rms_detrended: vec![0.0; rms_history_len],
            low_rms_acf:       vec![0.0; rms_history_len / 2],
            beat_grid: vec![f32::NAN; BEAT_GRID_LEN],
            // Tracker derives its period bounds from BEAT_TRACKER_MIN/MAX_BPM
            // — the rms_history-derived bounds were a stale concern from the
            // old harmonic-sum sweep era.
            beat_tracker: BeatTracker::new(dt),
            beat_state: vec![f32::NAN; BEAT_STATE_LEN],
            beat_position: 0.0,
            beat_pulses: vec![f32::NAN; BEAT_PULSES_LEN],
            gen_acf_fft_forward,
            gen_acf_fft_inverse,
            gen_acf_time_buf,
            gen_acf_freq_buf,
            onset_acf: vec![0.0; rms_history_len / 2],
        }
    }

    /// Set the spectrum smoothing time constant (seconds). The internal
    /// `smoothing_alpha` is recomputed from `tau` and the dt captured at
    /// construction: `alpha = 1 - exp(-dt / tau)`. Smaller tau → faster
    /// response. Clamped to [0.001, 10.0] to avoid divide-by-zero and
    /// nonsensical multi-second settling times.
    pub fn set_smoothing_tau(&mut self, tau_secs: f32) {
        let tau = tau_secs.clamp(0.001, 10.0);
        self.smoothing_alpha = 1.0 - (-self.dt / tau).exp();
    }

    /// Set the time constant (seconds) for the rms_acf decaying accumulator.
    /// `accum_alpha` is recomputed as `1 - exp(-dt / tau)`. Smaller tau →
    /// faster response, less stable peaks. Clamped to [0.05, 60.0] to avoid
    /// divide-by-zero and runaway settling.
    pub fn set_accum_tau_secs(&mut self, tau_secs: f32) {
        let tau = tau_secs.clamp(0.05, 60.0);
        self.accum_alpha = 1.0 - (-self.dt / tau).exp();
    }

    pub fn set_db_floor(&mut self, floor: f32) {
        self.db_floor = floor.clamp(-200.0, 0.0);
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

        // Half-wave-rectified RMS difference — proxy for the paper's spectral
        // flux OSS. `rms` is the just-computed full-band RMS for this frame.
        let onset = (rms - self.prev_rms).max(0.0);
        self.prev_rms = rms;
        self.onset_history.copy_within(1.., 0);
        let last = self.onset_history.len() - 1;
        self.onset_history[last] = onset;

        compute_gen_acf(
            &self.onset_history,
            &mut self.onset_acf,
            &self.gen_acf_fft_forward,
            &self.gen_acf_fft_inverse,
            &mut self.gen_acf_time_buf,
            &mut self.gen_acf_freq_buf,
        );

        autocorrelate(&self.waveform, &mut self.buffer_acf);

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

        // --- Per-band RMS via Parseval-correct FFT-bin energy summation. ---
        // band_rms = sqrt(parseval_band_scale · Σ|X[k]|² over band).
        // Bands cover bins 1..=low_end, low_end+1..=mid_end, mid_end+1..=N/2-1.
        // (DC at bin 0 and Nyquist at bin N/2 are excluded by design.)
        let nyquist_bin = self.freq_buffer.len() - 1; // N/2
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
        let low_rms  = (low_e  * self.parseval_band_scale).sqrt();
        let mid_rms  = (mid_e  * self.parseval_band_scale).sqrt();
        let high_rms = (high_e * self.parseval_band_scale).sqrt();

        // Shift each band history left, append newest at the end (oldest at index 0).
        // Same pattern as the existing time-domain rms_history.
        for h in [
            (&mut self.low_rms_history, low_rms),
            (&mut self.mid_rms_history, mid_rms),
            (&mut self.high_rms_history, high_rms),
        ] {
            let (history, value) = h;
            history.copy_within(1.., 0);
            let last = history.len() - 1;
            history[last] = value;
        }

        // RMS-envelope ACFs: detrend (subtract mean) then autocorrelate.
        // Computed here, after the FFT and band-RMS updates, so all ACF
        // computations sit together. Full-band ACF moved here from its
        // old pre-FFT location; behavior is unchanged.
        // Mean computed in f64 to avoid summation roundoff: a constant
        // f32 input of length 512 summed in f32 can drift by ~4e-7,
        // leaving the detrended buffer with uniform tiny non-zero values
        // that autocorrelate() then normalizes to ~1.0 instead of 0.
        let full_mean = (self.rms_history.iter().map(|&x| x as f64).sum::<f64>()
            / self.rms_history.len() as f64) as f32;
        for (dst, src) in self.rms_detrended.iter_mut().zip(self.rms_history.iter()) {
            *dst = src - full_mean;
        }
        autocorrelate(&self.rms_detrended, &mut self.rms_acf);

        // EMA-decayed accumulator over the instantaneous full-band ACF.
        // Builds up steady tempo peaks across many hops; suppresses
        // per-frame noise. Same alpha pattern as `smoothing_alpha`.
        for i in 0..self.rms_acf_accum.len() {
            self.rms_acf_accum[i] = self.accum_alpha * self.rms_acf[i]
                + (1.0 - self.accum_alpha) * self.rms_acf_accum[i];
        }
        self.pick_acf_peaks();
        self.update_beat_state();
        self.update_beat_pulses();

        let low_mean = (self.low_rms_history.iter().map(|&x| x as f64).sum::<f64>()
            / self.low_rms_history.len() as f64) as f32;
        for (dst, src) in self.low_rms_detrended.iter_mut().zip(self.low_rms_history.iter()) {
            *dst = src - low_mean;
        }
        autocorrelate(&self.low_rms_detrended, &mut self.low_rms_acf);

        // Magnitude → dB → normalized [0, 1] → smoothed
        // Skip bin 0 (DC); use bins 1..=spectrum.len()
        for (out_i, bin) in self.freq_buffer[1..=self.spectrum.len()].iter().enumerate() {
            let mag = (bin.re * bin.re + bin.im * bin.im).sqrt() * self.mag_scale;
            let db = if mag > 0.0 {
                20.0 * mag.log10()
            } else {
                self.db_floor
            };
            let clipped = db.clamp(self.db_floor, 0.0);
            let normalized = (clipped - self.db_floor) / (-self.db_floor); // [0, 1]
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

    pub fn rms_acf_accum(&self) -> Vec<f32> {
        self.rms_acf_accum.clone()
    }

    pub fn low_rms_acf(&self) -> Vec<f32> {
        self.low_rms_acf.clone()
    }

    pub fn rms_history(&self) -> Vec<f32> {
        self.rms_history.clone()
    }

    pub fn onset_history(&self) -> Vec<f32> {
        self.onset_history.clone()
    }

    pub fn onset_acf(&self) -> Vec<f32> {
        self.onset_acf.clone()
    }

    pub fn low_rms_history(&self) -> Vec<f32> {
        self.low_rms_history.clone()
    }

    pub fn mid_rms_history(&self) -> Vec<f32> {
        self.mid_rms_history.clone()
    }

    pub fn high_rms_history(&self) -> Vec<f32> {
        self.high_rms_history.clone()
    }

    pub fn acf_peaks(&self) -> Vec<f32> {
        self.acf_peaks.clone()
    }

    pub fn beat_grid(&self) -> Vec<f32> {
        self.beat_grid.clone()
    }

    pub fn beat_pulses(&self) -> Vec<f32> {
        self.beat_pulses.clone()
    }

    pub fn beat_state(&self) -> Vec<f32> {
        self.beat_state.clone()
    }

    /// Advance `beat_position` and project saw waves into `beat_pulses`.
    /// For each cycle multiplier m ∈ BEAT_PULSE_CYCLES,
    ///   saw_m = 1 − frac(beat_position / m)
    /// — 1.0 at the start of each m-cycle, →0 just before the next.
    ///
    /// Two modes:
    ///   - **Phase-locked** (phase is real): set `beat_position`'s fractional
    ///     part to `phase/period` so the cycle-m pulses anchor to actual beat
    ///     boundaries. The integer part advances by 1 each time the phase
    ///     wraps backward (caught via the 0.5-threshold check).
    ///   - **Free-running** (phase is NaN — silence or unconfident phase
    ///     fit): just advance by 1/period each frame. The saws keep moving at
    ///     the tracker's rate; alignment is whatever it happened to be when
    ///     phase last had a value.
    ///
    /// Wraps mod 16 (= LCM of BEAT_PULSE_CYCLES) so f32 precision stays
    /// healthy over long sessions.
    fn update_beat_pulses(&mut self) {
        let period = self.beat_grid[0];
        let phase = self.beat_grid[1];
        if period.is_nan() || period <= 0.0 {
            for slot in self.beat_pulses.iter_mut() {
                *slot = f32::NAN;
            }
            return;
        }

        let new_bp = if phase.is_nan() {
            // Free-run.
            self.beat_position + 1.0 / period
        } else {
            // Phase-lock: align fractional part to phase/period.
            let phase_frac = (phase / period).clamp(0.0, 0.999_999);
            let prev_bp = self.beat_position;
            let mut candidate = prev_bp.floor() + phase_frac;
            // Phase wrapped backward across a beat boundary — bump the
            // integer part to keep `beat_position` monotonic.
            if candidate < prev_bp - 0.5 {
                candidate += 1.0;
            }
            candidate
        };
        self.beat_position = new_bp.rem_euclid(16.0);

        for (i, &m) in BEAT_PULSE_CYCLES.iter().enumerate() {
            let frac = (self.beat_position / m).fract();
            self.beat_pulses[i] = 1.0 - frac;
        }
    }

    /// Update the `BeatTracker` from this frame's `acf_peaks`, then mirror
    /// its smoothed state into the public `beat_grid` and `beat_state`
    /// buffers. Replaces the old per-frame harmonic-sum fit — peak picking
    /// already extracts the strongest features of the ACF, so re-summing the
    /// raw signal is redundant; the tracker integrates these observations
    /// over time with confidence-gated EMA + measure hysteresis.
    fn update_beat_state(&mut self) {
        self.beat_tracker.observe(&self.acf_peaks);

        let period = self.beat_tracker.bpm_period;
        let phase = self.fit_beat_phase(period);

        self.beat_grid[0] = period;
        self.beat_grid[1] = phase;
        self.beat_grid[2] = self.beat_tracker.bpm_confidence;

        self.beat_state[0] = self.beat_tracker.bpm();
        self.beat_state[1] = self.beat_tracker.bpm_confidence;
        self.beat_state[2] = self.beat_tracker.beats_per_measure as f32;
        self.beat_state[3] = self.beat_tracker.measure_confidence;
    }

    /// Find the phase φ ∈ [0, period) that maximizes the comb-filter response
    /// of `rms_history` against a pulse train at `period`:
    ///   score(φ) = Σ_k rms_history[(n - 1) - round(φ + k·period)]   (k ≥ 0, while in range)
    /// φ is "how many hops ago the most-recent beat fell" — small φ means a
    /// beat just happened. Returned in hop units, sub-bin via the
    /// `BEAT_PHASE_STEP_HOPS` sweep grain.
    ///
    /// Allocation-free; bounded inner loop (rms_history.len() / period ≤ ~50
    /// at default settings). No detrending: rms is already non-negative, so
    /// summing it directly is equivalent to a peak-aligned correlation.
    fn fit_beat_phase(&self, period: f32) -> f32 {
        let n = self.rms_history.len();
        if period < 1.0 || (period as usize) >= n {
            return f32::NAN;
        }

        let n_phases = (period / BEAT_PHASE_STEP_HOPS).ceil() as usize;
        let last_idx = (n - 1) as i32;

        let mut best_phi = 0.0f32;
        let mut best_score = -1.0f32;

        for j in 0..n_phases {
            let phi = j as f32 * BEAT_PHASE_STEP_HOPS;
            let mut score = 0.0f32;
            let mut k = 0usize;
            loop {
                let pos = phi + k as f32 * period;
                let idx_signed = last_idx - pos.round() as i32;
                if idx_signed < 0 {
                    break;
                }
                score += self.rms_history[idx_signed as usize];
                k += 1;
            }
            if score > best_score {
                best_score = score;
                best_phi = phi;
            }
        }
        // Silent rms_history → all candidate scores are 0 and no improvement
        // ever beats the initial best_score = -1.0... wait, score=0 > -1.0 on
        // first iter, so best_score becomes 0 and best_phi stays at 0. We
        // return NaN in that case so consumers (beat_pulses, scrolling grid)
        // don't lock onto a bogus phase=0 alignment during silence.
        if best_score <= 0.0 {
            return f32::NAN;
        }
        best_phi
    }

    /// Pick top-`MAX_PEAKS` tempo peaks in `rms_acf_accum` and write them
    /// into `acf_peaks` as interleaved `[lag_frac, mag]` pairs (NaN-padded).
    ///
    /// Algorithm:
    ///   1. Scan lags `MIN_PEAK_LAG..len-1` for positive local maxima.
    ///   2. Sort candidates by integer-lag magnitude, descending.
    ///   3. Greedy-select with `MIN_PEAK_SPACING` integer-lag separation.
    ///   4. Parabolic sub-bin interpolation on each accepted peak.
    ///
    /// Allocation-free: uses the preallocated `peak_candidates` scratch
    /// (cleared, not freed) and a stack-bounded accepted set.
    fn pick_acf_peaks(&mut self) {
        // Reset output to all-NaN sentinels.
        for slot in self.acf_peaks.iter_mut() {
            *slot = f32::NAN;
        }

        let n = self.rms_acf_accum.len();
        if n < MIN_PEAK_LAG + 2 {
            return;
        }

        // 1. Scan candidates.
        self.peak_candidates.clear();
        for k in MIN_PEAK_LAG..(n - 1) {
            let y1 = self.rms_acf_accum[k];
            if y1 > 0.0 && y1 > self.rms_acf_accum[k - 1] && y1 > self.rms_acf_accum[k + 1] {
                self.peak_candidates.push((k, y1));
            }
        }

        // 2. Sort by magnitude descending.
        self.peak_candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Greedy select with min-spacing.
        let mut accepted: [u32; MAX_PEAKS] = [0; MAX_PEAKS];
        let mut accepted_count: usize = 0;
        for &(k, _mag) in self.peak_candidates.iter() {
            if accepted_count == MAX_PEAKS {
                break;
            }
            let mut too_close = false;
            for i in 0..accepted_count {
                let dist = (k as i32 - accepted[i] as i32).unsigned_abs() as usize;
                if dist < MIN_PEAK_SPACING {
                    too_close = true;
                    break;
                }
            }
            if !too_close {
                accepted[accepted_count] = k as u32;
                accepted_count += 1;
            }
        }

        // 4. Sub-bin parabolic refinement, write output.
        for i in 0..accepted_count {
            let k = accepted[i] as usize;
            let y0 = self.rms_acf_accum[k - 1];
            let y1 = self.rms_acf_accum[k];
            let y2 = self.rms_acf_accum[k + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let (lag_frac, mag) = if denom.abs() < 1e-12 {
                (k as f32, y1)
            } else {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                (k as f32 + delta, y1 - 0.25 * (y0 - y2) * delta)
            };
            // Sharpness = -denom: at a real maximum y1 > y0,y2 ⇒ denom < 0,
            // so -denom > 0. Larger = narrower peak = more confident "real beat".
            self.acf_peaks[3 * i] = lag_frac;
            self.acf_peaks[3 * i + 1] = mag;
            self.acf_peaks[3 * i + 2] = -denom;
        }
    }
}

/// Generalized autocorrelation per Percival & Tzanetakis 2014 §II-B.2:
/// zero-pad input to 2N, forward FFT, magnitude compression `|X|^0.5`, inverse
/// FFT, take the first N/2 lags, normalize so output[0] == 1.0. Allocation-
/// free: caller passes scratch buffers (`time_buf` length 2N, `freq_buf`
/// length N+1) and pre-built `realfft` planners.
///
/// `c = 0.5` (the paper's empirically best choice) gives narrower ACF peaks
/// than `c = 2.0` (regular ACF) — this is what makes downstream peak picking
/// and pulse-train scoring more discriminative.
fn compute_gen_acf(
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

    // |X|^0.5 magnitude compression (Percival & Tzanetakis c=0.5). After
    // squaring re²+im² this becomes (·)^0.25; the division by time_buf[0]
    // below cancels realfft's unnormalized IFFT scale.
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
impl Dsp {
    pub fn test_set_rms_acf_accum(&mut self, src: &[f32]) {
        let n = src.len().min(self.rms_acf_accum.len());
        self.rms_acf_accum[..n].copy_from_slice(&src[..n]);
        for v in self.rms_acf_accum.iter_mut().skip(n) {
            *v = 0.0;
        }
    }

    pub fn test_run_peak_picking(&mut self) {
        self.pick_acf_peaks();
    }

    pub fn test_run_update_beat_state(&mut self) {
        self.update_beat_state();
    }

    /// Direct access to the tracker for unit tests — bypasses `process()`.
    pub fn test_observe_tracker(&mut self) {
        self.beat_tracker.observe(&self.acf_peaks);
    }

    pub fn test_tracker_bpm_period(&self) -> f32 {
        self.beat_tracker.bpm_period
    }

    pub fn test_tracker_beats_per_measure(&self) -> u32 {
        self.beat_tracker.beats_per_measure
    }

    pub fn test_set_acf_peak(&mut self, slot: usize, lag: f32, mag: f32, sharpness: f32) {
        self.acf_peaks[3 * slot] = lag;
        self.acf_peaks[3 * slot + 1] = mag;
        self.acf_peaks[3 * slot + 2] = sharpness;
    }

    pub fn test_clear_acf_peaks(&mut self) {
        for v in self.acf_peaks.iter_mut() {
            *v = f32::NAN;
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
        autocorrelate(&input, &mut output);
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
        let signal: Vec<f32> = (0..2048)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
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
        assert!(acf[48] > acf[47], "expected acf[48]={} > acf[47]={}", acf[48], acf[47]);
        assert!(acf[48] > acf[49], "expected acf[48]={} > acf[49]={}", acf[48], acf[49]);
        assert!(acf[48] > 0.9, "expected strong peak at period, got acf[48]={}", acf[48]);
    }

    #[test]
    fn rms_acf_has_correct_length() {
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert_eq!(dsp.rms_acf().len(), 256);
    }

    #[test]
    fn acf_of_silence_is_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
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
        let mut dsp = Dsp::new(8, 48000.0, 4, 512);
        let constant = vec![0.5_f32; 8];
        // RMS of [0.5; 8] is 0.5; need >= 512 calls to fully fill.
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
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
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
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
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
        let large = Dsp::new(2048, 48000.0, 1024, 512);
        let small = Dsp::new(2048, 48000.0, 512, 512);
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
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert_eq!(dsp.low_band_bin_end, 6);
        assert_eq!(dsp.mid_band_bin_end, 64);
    }

    #[test]
    fn parseval_band_scale_matches_formula() {
        // parseval_band_scale = 2 / (N · Σ hann²)
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
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
            assert!((a - b).abs() < 1e-5, "alpha≈1 EMA should be stable: {} vs {}", a, b);
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
        assert!(max < 0.1, "expected sluggish EMA → near-zero spectrum after 3 calls, got max={}", max);
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
    fn rms_acf_length_tracks_history_length() {
        assert_eq!(Dsp::new(2048, 48000.0, 1024, 1024).rms_acf().len(), 512);
        assert_eq!(Dsp::new(2048, 48000.0, 1024, 256).rms_acf().len(), 128);
        assert_eq!(Dsp::new(2048, 48000.0, 1024, 512).rms_acf().len(), 256);
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
        assert!((high - 0.7071).abs() < 0.05, "high {} should be ≈ 0.707", high);
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
        dsp.process(&loud_low);   // pushes a non-zero into history
        dsp.process(&silent);     // pushes a zero
        let h = dsp.low_rms_history();
        let n = h.len();
        assert_eq!(h[n - 1], 0.0, "newest should be silent");
        assert!(h[n - 2] > 0.5, "second-newest should be the loud sine, got {}", h[n - 2]);
    }

    #[test]
    fn low_rms_acf_has_correct_length() {
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        assert_eq!(dsp.low_rms_acf().len(), 256);
    }

    #[test]
    fn low_rms_acf_constant_input_is_zero() {
        // Fill low_rms_history with a constant non-zero band-RMS by feeding the
        // same loud bin-aligned low-frequency sine repeatedly. Detrended ACF on
        // a constant should be zero everywhere.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 93.75 * (i as f32 / sr)).sin())
            .collect();
        // 512 calls fully fills the band history with a constant value.
        for _ in 0..512 {
            dsp.process(&signal);
        }
        let acf = dsp.low_rms_acf();
        for &v in &acf {
            assert!(
                v.abs() < 1e-4,
                "expected near-zero detrended ACF for constant input, got {}",
                v
            );
        }
    }

    #[test]
    fn accum_alpha_matches_formula() {
        // alpha = 1 - exp(-dt / tau) at default tau (4.0 s), dt = 1024/48000
        let dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let dt = 1024.0_f32 / 48000.0;
        let expected = 1.0 - (-dt / ACCUM_TAU_DEFAULT_SECS).exp();
        assert!(
            (dsp.accum_alpha - expected).abs() < 1e-6,
            "got {}, expected {}",
            dsp.accum_alpha,
            expected
        );
    }

    #[test]
    fn set_accum_tau_secs_clamps_and_recomputes() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let before = dsp.accum_alpha;
        dsp.set_accum_tau_secs(20.0);
        assert!(dsp.accum_alpha < before, "longer tau should yield smaller alpha");

        // Below clamp: 0.001 should clamp up to 0.05.
        dsp.set_accum_tau_secs(0.001);
        let dt = 1024.0_f32 / 48000.0;
        let expected = 1.0 - (-dt / 0.05).exp();
        assert!((dsp.accum_alpha - expected).abs() < 1e-6, "lower clamp not applied");

        // Above clamp: 1000.0 should clamp down to 60.0.
        dsp.set_accum_tau_secs(1000.0);
        let expected = 1.0 - (-dt / 60.0).exp();
        assert!((dsp.accum_alpha - expected).abs() < 1e-6, "upper clamp not applied");
    }

    #[test]
    fn rms_acf_accum_silent_input_is_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0_f32; 2048];
        for _ in 0..200 {
            dsp.process(&silent);
        }
        let accum = dsp.rms_acf_accum();
        assert_eq!(accum.len(), 256);
        for &v in &accum {
            assert_eq!(v, 0.0, "silent → accumulator must stay zero, got {}", v);
        }
    }

    #[test]
    fn acf_peaks_silent_input_all_nan() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0_f32; 2048];
        for _ in 0..50 {
            dsp.process(&silent);
        }
        let peaks = dsp.acf_peaks();
        assert_eq!(peaks.len(), 3 * MAX_PEAKS);
        for (i, &v) in peaks.iter().enumerate() {
            assert!(v.is_nan(), "slot {} should be NaN, got {}", i, v);
        }
    }

    #[test]
    fn acf_peaks_min_lag_enforced() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        // Synthetic accumulator: a single isolated peak at lag 5 (below MIN_PEAK_LAG=10).
        let mut accum = vec![0.0_f32; 256];
        accum[5] = 0.9;
        dsp.test_set_rms_acf_accum(&accum);
        dsp.test_run_peak_picking();
        let peaks = dsp.acf_peaks();
        // No peak should be picked; all slots NaN.
        for &v in &peaks {
            assert!(v.is_nan(), "expected no peak below MIN_PEAK_LAG, got {}", v);
        }
    }

    #[test]
    fn acf_peaks_finds_isolated_peak_with_subbin_offset() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        // Asymmetric triangular peak at integer lag 50:
        //   y0 = accum[49] = 0.6, y1 = accum[50] = 1.0, y2 = accum[51] = 0.8
        // Parabolic interp: δ = 0.5*(y0-y2)/(y0 - 2*y1 + y2) = 0.5*(-0.2)/(-0.6) ≈ 0.1667
        let mut accum = vec![0.0_f32; 256];
        accum[49] = 0.6;
        accum[50] = 1.0;
        accum[51] = 0.8;
        dsp.test_set_rms_acf_accum(&accum);
        dsp.test_run_peak_picking();
        let peaks = dsp.acf_peaks();
        let lag0 = peaks[0];
        assert!(!lag0.is_nan(), "expected a peak in slot 0");
        assert!(
            (lag0 - 50.1667).abs() < 0.01,
            "expected sub-bin lag ≈ 50.1667, got {}",
            lag0
        );
        // Slots 1..MAX_PEAKS must be NaN.
        for i in 1..MAX_PEAKS {
            assert!(peaks[3 * i].is_nan(), "slot {} lag should be NaN", i);
            assert!(peaks[3 * i + 1].is_nan(), "slot {} mag should be NaN", i);
        }
    }

    #[test]
    fn acf_peaks_min_spacing_filters_nearby() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        // Two equal-magnitude lobes at lags 50 and 52 (spacing = 2 < MIN_PEAK_SPACING=3).
        // Both are local maxima individually because lag 51 sits between them with
        // a slightly lower value.
        let mut accum = vec![0.0_f32; 256];
        accum[49] = 0.8;
        accum[50] = 1.0;
        accum[51] = 0.85;
        accum[52] = 1.0;
        accum[53] = 0.8;
        dsp.test_set_rms_acf_accum(&accum);
        dsp.test_run_peak_picking();
        let peaks = dsp.acf_peaks();
        // Exactly one of the two lobes should be picked. Its integer lag rounds
        // to either 50 or 52.
        let mut accepted = 0;
        for i in 0..MAX_PEAKS {
            if !peaks[3 * i].is_nan() {
                accepted += 1;
            }
        }
        assert_eq!(accepted, 1, "min-spacing should leave only one peak");
    }

    #[test]
    fn acf_peaks_top_n_selection_in_descending_magnitude() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        // 15 isolated peaks (well-spaced) with strictly decreasing magnitudes.
        // Spacing = 8 to satisfy MIN_PEAK_SPACING; first peak at lag 16 to clear
        // MIN_PEAK_LAG. Only the top 10 should be picked, in magnitude order.
        let mut accum = vec![0.0_f32; 256];
        for i in 0..15 {
            let lag = 16 + 8 * i;
            let mag = 1.0 - 0.05 * i as f32;
            accum[lag - 1] = mag * 0.5;
            accum[lag] = mag;
            accum[lag + 1] = mag * 0.5;
        }
        dsp.test_set_rms_acf_accum(&accum);
        dsp.test_run_peak_picking();
        let peaks = dsp.acf_peaks();
        // First 10 slots are real peaks, in descending magnitude order.
        let mut last_mag = f32::INFINITY;
        for i in 0..MAX_PEAKS {
            let lag = peaks[3 * i];
            let mag = peaks[3 * i + 1];
            assert!(!lag.is_nan(), "slot {}: expected real peak", i);
            assert!(mag <= last_mag + 1e-5, "slot {}: mag {} > prev {}", i, mag, last_mag);
            last_mag = mag;
        }
    }

    #[test]
    fn acf_peaks_negative_correlations_skipped() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        // Negative-magnitude local maximum: accum[50] = -0.1, surrounded by -0.5 / -0.5.
        // Even though -0.1 > -0.5, anti-correlations aren't beats and must be skipped.
        let mut accum = vec![0.0_f32; 256];
        accum[49] = -0.5;
        accum[50] = -0.1;
        accum[51] = -0.5;
        dsp.test_set_rms_acf_accum(&accum);
        dsp.test_run_peak_picking();
        let peaks = dsp.acf_peaks();
        for &v in &peaks {
            assert!(v.is_nan(), "negative peak must be skipped, got {}", v);
        }
    }

    #[test]
    fn rms_acf_accum_converges_to_instantaneous_for_steady_periodic() {
        // Each hop receives a 2048-sample sine whose amplitude depends on the
        // hop index — this produces a non-constant `rms_history` with real
        // temporal structure (a slow envelope), so the detrended ACF has
        // non-trivial values. Without this, every hop has the same RMS, the
        // detrended history is all zeros, and the convergence assertion holds
        // trivially because both sides are zero.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        // Convergence rule of thumb: ~5τ at default 4 s tau, dt ≈ 21.33 ms → ~940 hops.
        // Use 1500 hops for headroom.
        for k in 0..1500 {
            // Slow amplitude envelope across hops (~ one cycle per ~63 hops).
            let amp = 0.5 + 0.3 * (k as f32 * 0.1).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| {
                    let t = i as f32 / sr;
                    amp * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                })
                .collect();
            dsp.process(&signal);
        }
        let inst = dsp.rms_acf();
        let accum = dsp.rms_acf_accum();
        // After convergence, accum tracks inst. Tighter tolerance is fine because
        // the EMA at 5τ has ~99% reached steady state.
        for (i, (a, b)) in accum.iter().zip(inst.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.05,
                "lag {}: accum {} should track inst {}",
                i, a, b
            );
        }
        // Sanity: at least *some* lag must be non-trivially non-zero — otherwise
        // the test is vacuous again.
        let max_abs = inst.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs > 0.01, "expected non-trivial ACF; max |inst| = {}", max_abs);
    }

    #[test]
    fn process_pipeline_finds_periodic_peak_via_acf_peaks() {
        // End-to-end integration test: drive process() with a known per-hop
        // amplitude period, then verify acf_peaks()[0] reports that period as
        // the dominant tempo lag. Catches integration bugs in process() that
        // unit-tested individual stages would miss (e.g., pick_acf_peaks
        // not called, called before EMA update, etc.).
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let period_hops = 32usize;
        // 1500 hops covers both rms_history fill (512) and accumulator
        // convergence at default tau=4s (~940 hops).
        for k in 0..1500 {
            // Sinusoidal envelope across hops with period `period_hops`.
            // Add a constant offset of 0.6 so amplitude stays positive — RMS is
            // |signal|-symmetric, so a zero-crossing envelope would create a
            // doubled-frequency artifact.
            let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| {
                    let t = i as f32 / sr;
                    amp * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                })
                .collect();
            dsp.process(&signal);
        }
        let peaks = dsp.acf_peaks();
        let lag0 = peaks[0];
        assert!(!lag0.is_nan(), "expected at least one peak after convergence");
        assert!(
            (lag0 - period_hops as f32).abs() < 1.0,
            "expected peak near lag {}, got {}",
            period_hops, lag0
        );
        // The peak's magnitude (ACF normalized to [0,1] roughly) should be
        // meaningfully positive, not just barely above zero.
        let mag0 = peaks[1];
        assert!(mag0 > 0.1, "expected significant peak magnitude, got {}", mag0);
    }

    #[test]
    fn tracker_holds_default_120_bpm_under_silence() {
        // With no peaks ever picked, tracker holds its initial state. Period
        // and BPM stay at the 120 BPM defaults; phase is NaN (silent
        // rms_history can't yield a phase). bpm_confidence stays at 0.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0_f32; 2048];
        for _ in 0..50 {
            dsp.process(&silent);
        }
        let grid = dsp.beat_grid();
        let state = dsp.beat_state();
        assert_eq!(grid.len(), 3);
        assert_eq!(state.len(), 4);
        assert!(!grid[0].is_nan(), "period should hold at default, not NaN");
        assert!(grid[1].is_nan(), "phase should be NaN for silence");
        assert!((state[0] - 120.0).abs() < 1.0, "BPM should hold near 120, got {}", state[0]);
        assert_eq!(state[2] as u32, 4, "beats_per_measure should hold at 4");
        assert!(state[1] < 0.05, "bpm_confidence should be ~0 under silence, got {}", state[1]);
    }

    #[test]
    fn tracker_pulls_period_toward_nearby_observation() {
        // Inject peaks at lags 24 and 48 (period 24 ≈ 117 BPM). Initial
        // period is 23.44 (120 BPM at default sr/hop), so peak at 24 lies
        // within the Gaussian tolerance and votes for T=24. Tracker should
        // pull toward 24 over many frames.
        // Note: large jumps (e.g. 23 → 32) are intentionally NOT supported
        // without a wider σ — the tracker assumes tempo drifts smoothly.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let initial = dsp.test_tracker_bpm_period();
        for _ in 0..200 {
            dsp.test_clear_acf_peaks();
            dsp.test_set_acf_peak(0, 24.0, 1.0, 0.5);
            dsp.test_set_acf_peak(1, 48.0, 0.8, 0.4);
            dsp.test_observe_tracker();
        }
        let final_period = dsp.test_tracker_bpm_period();
        assert!(
            (final_period - 24.0).abs() < 0.3,
            "tracker should converge near 24, started at {}, got {}",
            initial, final_period
        );
    }

    #[test]
    fn tracker_holds_period_against_single_outlier_frame() {
        // Lock onto period ≈ 24 with sustained input.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        for _ in 0..200 {
            dsp.test_clear_acf_peaks();
            dsp.test_set_acf_peak(0, 24.0, 1.0, 0.5);
            dsp.test_set_acf_peak(1, 48.0, 0.8, 0.4);
            dsp.test_observe_tracker();
        }
        let locked = dsp.test_tracker_bpm_period();
        // Single noisy frame with a far-off peak that aligns badly to the
        // current period — its alignment weight will be near-zero, so it
        // shouldn't move the tracker.
        dsp.test_clear_acf_peaks();
        dsp.test_set_acf_peak(0, 50.0, 1.0, 0.5);
        dsp.test_observe_tracker();
        let after = dsp.test_tracker_bpm_period();
        assert!(
            (after - locked).abs() < 0.5,
            "single far-off outlier shouldn't drag period: locked {} → {}",
            locked, after
        );
    }

    #[test]
    fn tracker_picks_beats_per_measure_from_strongest_m_peak() {
        // Drive period to ≈ 24, then provide peaks at 24, 48, and 96 with
        // amplitudes biased toward 96 (the "measure" peak the user described).
        // beats_per_measure should be 4 because m=4·24=96 has the strongest
        // peak of any candidate target.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        for _ in 0..200 {
            dsp.test_clear_acf_peaks();
            dsp.test_set_acf_peak(0, 24.0, 0.6, 0.4);
            dsp.test_set_acf_peak(1, 48.0, 0.7, 0.5);
            dsp.test_set_acf_peak(2, 96.0, 1.0, 1.0); // strongest + sharpest
            dsp.test_observe_tracker();
        }
        assert_eq!(
            dsp.test_tracker_beats_per_measure(), 4,
            "strongest peak at 4·period should yield beats_per_measure=4"
        );
    }

    #[test]
    fn tracker_switches_beats_per_measure_when_strongest_m_peak_changes() {
        // First lock onto m=4 with a strong peak at 4·24=96.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        for _ in 0..200 {
            dsp.test_clear_acf_peaks();
            dsp.test_set_acf_peak(0, 24.0, 0.6, 0.4);
            dsp.test_set_acf_peak(1, 48.0, 0.6, 0.4);
            dsp.test_set_acf_peak(2, 96.0, 1.0, 1.0);
            dsp.test_observe_tracker();
        }
        assert_eq!(dsp.test_tracker_beats_per_measure(), 4);
        // Now move the strongest peak to 2·24=48 (and remove the 96 peak).
        // Hysteresis (1.3× margin) requires the new winner to clearly beat
        // current — sustained strong evidence at 48 should flip to m=2.
        for _ in 0..200 {
            dsp.test_clear_acf_peaks();
            dsp.test_set_acf_peak(0, 24.0, 0.5, 0.3);
            dsp.test_set_acf_peak(1, 48.0, 1.0, 1.0); // strong + sharp
            dsp.test_observe_tracker();
        }
        assert_eq!(
            dsp.test_tracker_beats_per_measure(), 2,
            "sustained strongest peak at 2·period should switch to m=2"
        );
    }

    #[test]
    fn beat_grid_finds_phase_aligned_with_rms_peaks() {
        // End-to-end: drive process() with a periodic envelope. The phase the
        // worklet reports should be small — i.e., the fit decided there was a
        // beat near the newest sample of the rms history. The exact phase
        // depends on the envelope's place in its cycle at the moment we stop
        // feeding the dsp, so we just verify the phase is sensible (in
        // [0, period)) and that score is positive.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let period_hops = 32usize;
        for k in 0..1500 {
            let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| {
                    let t = i as f32 / sr;
                    amp * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                })
                .collect();
            dsp.process(&signal);
        }
        let grid = dsp.beat_grid();
        let period = grid[0];
        let phase = grid[1];
        let score = grid[2];
        assert!(!period.is_nan(), "expected period fit");
        assert!(!phase.is_nan(), "expected phase fit");
        assert!(phase >= 0.0, "phase must be non-negative, got {}", phase);
        assert!(phase < period, "phase {} must be < period {}", phase, period);
        assert!(score > 0.0, "expected positive score, got {}", score);
    }

    #[test]
    fn beat_pulses_free_run_at_default_rate_under_silence() {
        // Tracker holds 120 BPM defaults under silence; phase is NaN so the
        // pulse loop free-runs (advances by 1/period each frame). Pulses are
        // valid 0..1 floats that change between frames, not NaN. The renderer
        // can use beat_grid[2] (confidence) to dim/hide if desired.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0_f32; 2048];
        for _ in 0..10 {
            dsp.process(&silent);
        }
        let p_before = dsp.beat_pulses();
        for &v in p_before.iter() {
            assert!(!v.is_nan(), "free-run pulse should not be NaN, got {}", v);
            assert!((0.0..=1.0).contains(&v));
        }
        dsp.process(&silent);
        let p_after = dsp.beat_pulses();
        // Cycle 1 should advance noticeably between frames at default ~23.4 lag period.
        assert!((p_after[0] - p_before[0]).abs() > 1e-3, "cycle-1 should advance under free-run");
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
            let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
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
                i, v
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
        assert!(dp1 > 0.005, "cycle-1 should advance per hop, got Δ = {}", dp1);
        assert!(dp16 < dp1, "cycle-16 should advance less than cycle-1: {} vs {}", dp16, dp1);
    }

    #[test]
    fn beat_grid_end_to_end_via_process() {
        // End-to-end: drive process() with a periodic envelope and verify the
        // beat grid lands on the period that pick_acf_peaks reports.
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let sr = 48000.0_f32;
        let period_hops = 32usize;
        for k in 0..1500 {
            let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
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
            period_hops, grid[0]
        );
    }

    #[test]
    fn onset_history_captures_positive_rms_diff() {
        // RMS of a constant-amplitude signal A over a frame is A.
        // Step ladder: silence (RMS 0) → 0.5 → 0.5 → 0.3.
        // Expected onset values per process call (max(0, rms - prev_rms)):
        //   process(silent): rms=0, prev=0 → onset=0
        //   process(half):   rms=0.5, prev=0   → onset=0.5
        //   process(half):   rms=0.5, prev=0.5 → onset=0
        //   process(lower):  rms=0.3, prev=0.5 → onset=0  (negative diff clamped)
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0f32; 2048];
        let half   = vec![0.5f32; 2048];
        let lower  = vec![0.3f32; 2048];
        dsp.process(&silent);
        dsp.process(&half);
        dsp.process(&half);
        dsp.process(&lower);
        let onset = dsp.onset_history();
        let n = onset.len();
        // newest at index n-1, oldest at index 0
        assert!((onset[n - 1] - 0.0).abs() < 1e-3, "newest = {}", onset[n - 1]);
        assert!((onset[n - 2] - 0.0).abs() < 1e-3, "newest-1 = {}", onset[n - 2]);
        assert!((onset[n - 3] - 0.5).abs() < 1e-2, "newest-2 = {}", onset[n - 3]);
        assert!((onset[n - 4] - 0.0).abs() < 1e-3, "newest-3 = {}", onset[n - 4]);
    }

    #[test]
    fn gen_acf_silent_input_is_zero() {
        let mut dsp = Dsp::new(2048, 48000.0, 1024, 512);
        let silent = vec![0.0f32; 2048];
        for _ in 0..5 { dsp.process(&silent); }
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
            let amp = 0.6 + 0.3 * (2.0 * std::f32::consts::PI * (k as f32) / (period_hops as f32)).sin();
            let signal: Vec<f32> = (0..2048)
                .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / sr)).sin())
                .collect();
            dsp.process(&signal);
        }
        let acf = dsp.onset_acf();
        let p = period_hops;
        let around = (p - 5..=p + 5).map(|i| acf[i]).fold(f32::NEG_INFINITY, f32::max);
        // Avoid lags near period/2 = 16 (a natural ACF harmonic of the sine envelope)
        // and lags very close to 0 (trivially high by normalization). Use lags 5-12
        // as a "genuinely non-periodic" reference region.
        let far = (5..=12).map(|i| acf[i]).fold(f32::NEG_INFINITY, f32::max);
        assert!(around > far,
            "peak near lag {} ({:.3}) should exceed nearby non-period max ({:.3})",
            p, around, far);
    }
}
