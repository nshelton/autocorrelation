//! Output-buffer registry. Single source of truth for the JS-visible buffer
//! key vocabulary.
//!
//! Field names use camelCase to match the JS-side keys exactly — the same
//! string is used as Rust struct field, registry-lookup match arm, worklet
//! message field, and FeatureStore key. `#[allow(non_snake_case)]` accepts
//! these names.

use crate::beat::MAX_PEAKS;
use crate::perf::PERF_METRIC_COUNT;

const BEAT_GRID_LEN: usize = 3;
const BEAT_PULSES_LEN: usize = 4;
const BEAT_STATE_LEN: usize = 4;

#[allow(non_snake_case)]
pub struct Buffers {
    pub waveform: Vec<f32>,
    pub spectrum: Vec<f32>,
    pub bufferAcf: Vec<f32>,
    pub rms: Vec<f32>,
    pub rmsLow: Vec<f32>,
    pub rmsMid: Vec<f32>,
    pub rmsHigh: Vec<f32>,
    pub onset: Vec<f32>,
    pub onsetAcf: Vec<f32>,
    pub onsetAcfEnhanced: Vec<f32>,
    pub tea: Vec<f32>,
    pub candidates: Vec<f32>,
    pub beatGrid: Vec<f32>,
    pub beatPulses: Vec<f32>,
    pub beatState: Vec<f32>,
    pub dspPerfUs: Vec<f32>,
}

impl Buffers {
    pub fn new(window_size: usize, rms_history_len: usize) -> Self {
        let onset_acf_len = rms_history_len / 2;
        Self {
            waveform: vec![0.0; window_size],
            spectrum: vec![0.0; window_size / 2],
            bufferAcf: vec![0.0; window_size / 2],
            rms: vec![0.0; rms_history_len],
            rmsLow: vec![0.0; rms_history_len],
            rmsMid: vec![0.0; rms_history_len],
            rmsHigh: vec![0.0; rms_history_len],
            onset: vec![0.0; rms_history_len],
            onsetAcf: vec![0.0; onset_acf_len],
            onsetAcfEnhanced: vec![0.0; onset_acf_len],
            tea: vec![0.0; onset_acf_len],
            candidates: vec![f32::NAN; 3 * MAX_PEAKS],
            beatGrid: vec![f32::NAN; BEAT_GRID_LEN],
            beatPulses: vec![f32::NAN; BEAT_PULSES_LEN],
            beatState: vec![f32::NAN; BEAT_STATE_LEN],
            dspPerfUs: vec![f32::NAN; PERF_METRIC_COUNT],
        }
    }

    pub fn descriptors(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("waveform", self.waveform.len()),
            ("spectrum", self.spectrum.len()),
            ("bufferAcf", self.bufferAcf.len()),
            ("rms", self.rms.len()),
            ("rmsLow", self.rmsLow.len()),
            ("rmsMid", self.rmsMid.len()),
            ("rmsHigh", self.rmsHigh.len()),
            ("onset", self.onset.len()),
            ("onsetAcf", self.onsetAcf.len()),
            ("onsetAcfEnhanced", self.onsetAcfEnhanced.len()),
            ("tea", self.tea.len()),
            ("candidates", self.candidates.len()),
            ("beatGrid", self.beatGrid.len()),
            ("beatPulses", self.beatPulses.len()),
            ("beatState", self.beatState.len()),
            ("dspPerfUs", self.dspPerfUs.len()),
        ]
    }

    /// Look up a buffer's current contents by string key. Returns `None`
    /// for unknown names. The keys here ARE the JS contract — keep this
    /// match in sync with `descriptors()` and the `Buffers` field list.
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        match name {
            "waveform" => Some(&self.waveform),
            "spectrum" => Some(&self.spectrum),
            "bufferAcf" => Some(&self.bufferAcf),
            "rms" => Some(&self.rms),
            "rmsLow" => Some(&self.rmsLow),
            "rmsMid" => Some(&self.rmsMid),
            "rmsHigh" => Some(&self.rmsHigh),
            "onset" => Some(&self.onset),
            "onsetAcf" => Some(&self.onsetAcf),
            "onsetAcfEnhanced" => Some(&self.onsetAcfEnhanced),
            "tea" => Some(&self.tea),
            "candidates" => Some(&self.candidates),
            "beatGrid" => Some(&self.beatGrid),
            "beatPulses" => Some(&self.beatPulses),
            "beatState" => Some(&self.beatState),
            "dspPerfUs" => Some(&self.dspPerfUs),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_returns_some_for_known_keys_none_for_unknown() {
        let b = Buffers::new(2048, 512);
        assert!(b.get("waveform").is_some());
        assert!(b.get("rmsHigh").is_some());
        assert!(b.get("notARealKey").is_none());
    }

    #[test]
    fn descriptors_are_in_get_order() {
        let b = Buffers::new(2048, 512);
        let names: Vec<&str> = b.descriptors().into_iter().map(|(name, _)| name).collect();
        assert_eq!(
            names,
            vec![
                "waveform",
                "spectrum",
                "bufferAcf",
                "rms",
                "rmsLow",
                "rmsMid",
                "rmsHigh",
                "onset",
                "onsetAcf",
                "onsetAcfEnhanced",
                "tea",
                "candidates",
                "beatGrid",
                "beatPulses",
                "beatState",
                "dspPerfUs",
            ]
        );
    }

    #[test]
    fn dsp_perf_us_is_nan_at_init_and_correct_length() {
        let b = Buffers::new(2048, 512);
        let v = b.get("dspPerfUs").unwrap();
        assert_eq!(v.len(), PERF_METRIC_COUNT);
        for &x in v {
            assert!(x.is_nan());
        }
    }
}
