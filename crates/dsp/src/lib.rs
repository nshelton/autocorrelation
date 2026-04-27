use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Dsp {
    waveform: Vec<f32>,
}

#[wasm_bindgen]
impl Dsp {
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize) -> Dsp {
        Dsp {
            waveform: vec![0.0; window_size],
        }
    }

    /// Run analysis on one input window. After this returns, the
    /// `waveform()`, `spectrum()`, and `rms_history()` getters expose
    /// the latest results.
    pub fn process(&mut self, input: &[f32]) {
        let n = input.len().min(self.waveform.len());
        self.waveform[..n].copy_from_slice(&input[..n]);
    }

    /// The most recent input window (passthrough in v1; identical in v2).
    pub fn waveform(&self) -> Vec<f32> {
        self.waveform.clone()
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
}
