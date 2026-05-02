use wasm_bindgen::prelude::*;

mod acf;
mod autogain;
mod beat;
mod buffers;
mod perf;
mod spectrum;

use crate::acf::AcfState;
use crate::autogain::AutoGain;
use crate::beat::BeatState;
use crate::buffers::Buffers;
use crate::perf::PerfState;
use crate::spectrum::SpectrumState;

const AUTOGAIN_DEFAULT_TAU_SECS: f32 = 1.0;

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
    perf: PerfState,
    auto_rms: AutoGain,
    auto_rms_low: AutoGain,
    auto_rms_mid: AutoGain,
    auto_rms_high: AutoGain,
    auto_onset: AutoGain,
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
            perf: PerfState::new(dt),
            auto_rms: AutoGain::new(dt, AUTOGAIN_DEFAULT_TAU_SECS),
            auto_rms_low: AutoGain::new(dt, AUTOGAIN_DEFAULT_TAU_SECS),
            auto_rms_mid: AutoGain::new(dt, AUTOGAIN_DEFAULT_TAU_SECS),
            auto_rms_high: AutoGain::new(dt, AUTOGAIN_DEFAULT_TAU_SECS),
            auto_onset: AutoGain::new(dt, AUTOGAIN_DEFAULT_TAU_SECS),
        }
    }

    pub fn process(&mut self, input: &[f32]) {
        let t_start = self.perf.frame_start();

        let n = input.len().min(self.buffers.waveform.len());
        self.buffers.waveform[..n].copy_from_slice(&input[..n]);
        let mean_sq: f32 = input.iter().take(n).map(|&x| x * x).sum::<f32>() / n.max(1) as f32;
        let rms = mean_sq.sqrt();
        push_history(&mut self.buffers.rms, self.auto_rms.apply(rms));

        let (low_rms, mid_rms, high_rms, flux) =
            self.spectrum
                .process(input, &mut self.buffers.spectrum, self.db_floor);
        push_history(&mut self.buffers.rmsLow, self.auto_rms_low.apply(low_rms));
        push_history(&mut self.buffers.rmsMid, self.auto_rms_mid.apply(mid_rms));
        push_history(
            &mut self.buffers.rmsHigh,
            self.auto_rms_high.apply(high_rms),
        );
        push_history(&mut self.buffers.onset, self.auto_onset.apply(flux));

        self.acf.process(
            &self.buffers.onset,
            &mut self.buffers.onsetAcf,
            &mut self.buffers.onsetAcfEnhanced,
            self.dt,
        );

        self.beat.process(
            &self.buffers.onset,
            &self.buffers.onsetAcfEnhanced,
            &mut self.buffers.beatGrid,
            &mut self.buffers.beatState,
            &mut self.buffers.beatPulses,
            self.dt,
        );

        crate::acf::autocorrelate(&self.buffers.waveform, &mut self.buffers.bufferAcf);

        self.perf.frame_end(t_start, &mut self.buffers.dspPerf);
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
    /// "teaTauSecs", "teaSigma", "acfSmoothingSigma", "dbFloor", "phaseLock",
    /// "autoGain".
    pub fn set_param(&mut self, key: &str, value: f32) {
        match key {
            "smoothingTauSecs" => self.spectrum.set_smoothing_tau(value, self.dt),
            "onsetSmoothingTauSecs" => self.spectrum.set_onset_release_tau(value, self.dt),
            "teaTauSecs" => self.beat.set_tea_tau(value, self.dt),
            "acfSmoothingSigma" => self.acf.set_smoothing_sigma(value),
            "acfDecay" => self.acf.set_decay(value),
            "dbFloor" => self.db_floor = value.clamp(-200.0, 0.0),
            "phaseLock" => self.beat.set_phase_lock_tau(value, self.dt),
            "autoGain" => {
                self.auto_rms.set_tau(value, self.dt);
                self.auto_rms_low.set_tau(value, self.dt);
                self.auto_rms_mid.set_tau(value, self.dt);
                self.auto_rms_high.set_tau(value, self.dt);
                self.auto_onset.set_tau(value, self.dt);
            }
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
