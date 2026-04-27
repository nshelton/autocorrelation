use wasm_bindgen::prelude::*;

/// v1: passthrough. Copies the input window into the output buffer.
/// Future versions will compute FFT, autocorrelation, RMS, beat phase.
#[wasm_bindgen]
pub struct Dsp {
    out: Vec<f32>,
}

#[wasm_bindgen]
impl Dsp {
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize) -> Dsp {
        Dsp { out: vec![0.0; window_size] }
    }

    /// Process one analysis window. Returns a borrowed view of the
    /// internal output buffer; the caller must copy before the next call.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let n = input.len().min(self.out.len());
        self.out[..n].copy_from_slice(&input[..n]);
        self.out[..n].to_vec()
    }
}
