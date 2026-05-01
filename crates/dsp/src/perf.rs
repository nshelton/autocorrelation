//! Live diagnostic timing for `Dsp::process`. Microsecond durations only —
//! never store absolute timestamps. Timer calls themselves perturb the
//! result slightly; values are not benchmark-grade.

pub const PERF_METRIC_COUNT: usize = 6;

pub const PERF_TOTAL: usize = 0;
pub const PERF_INPUT_RMS: usize = 1;
pub const PERF_SPECTRUM: usize = 2;
pub const PERF_ONSET_ACF: usize = 3;
pub const PERF_BEAT: usize = 4;
pub const PERF_BUFFER_ACF: usize = 5;

#[allow(dead_code)]
pub const PERF_METRIC_NAMES: [&str; PERF_METRIC_COUNT] = [
    "total",
    "inputRms",
    "spectrum",
    "onsetAcf",
    "beat",
    "bufferAcf",
];

#[cfg(target_arch = "wasm32")]
pub fn now_us() -> f64 {
    // performance.now() returns ms with sub-ms resolution where available.
    // Fall back to Date.now() (1ms resolution) if performance is missing.
    let global = js_sys::global();
    if let Ok(perf) = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("performance"))
    {
        if !perf.is_undefined() && !perf.is_null() {
            if let Ok(now_fn) =
                js_sys::Reflect::get(&perf, &wasm_bindgen::JsValue::from_str("now"))
            {
                if let Ok(func) = now_fn.dyn_into::<js_sys::Function>() {
                    if let Ok(v) = func.call0(&perf) {
                        if let Some(ms) = v.as_f64() {
                            return ms * 1000.0;
                        }
                    }
                }
            }
        }
    }
    js_sys::Date::now() * 1000.0
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(not(target_arch = "wasm32"))]
pub fn now_us() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_secs_f64() * 1_000_000.0
}

/// Elapsed `now - start` in microseconds, clamped to non-negative finite f32.
/// Defends against backwards clock movement on the JS side.
#[inline]
pub fn elapsed_us(start: f64, now: f64) -> f32 {
    let d = now - start;
    if d.is_finite() && d > 0.0 {
        d as f32
    } else {
        0.0
    }
}
