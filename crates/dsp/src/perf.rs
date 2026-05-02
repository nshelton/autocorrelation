//! Live diagnostic perf for `Dsp::process`. Publishes two EMAs:
//!   [0] totalMs — wall-clock per process() call, ms.
//!   [1] freqHz  — derived from EMA of call-to-call period.
//!
//! Inside an AudioWorklet the only clock JS hands wasm is `performance.now()`
//! (often 1 ms quantized) or `Date.now()` (1 ms). EMA smoothing recovers
//! sub-ms effective resolution from those quantized samples.

pub const PERF_METRIC_COUNT: usize = 2;
pub const PERF_TOTAL_MS: usize = 0;
pub const PERF_FREQ_HZ: usize = 1;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(target_arch = "wasm32")]
struct Clock {
    /// `(performance.now, performance)` resolved once at construction. Per-call
    /// `Reflect.get` round-trips dominated cost in the hot path; caching the
    /// resolved Function + `this` cuts JS↔wasm crossings to a single `call0`.
    /// `None` when `performance` isn't exposed (some AudioWorklet hosts).
    perf_now: Option<(js_sys::Function, wasm_bindgen::JsValue)>,
}

#[cfg(target_arch = "wasm32")]
impl Clock {
    fn new() -> Self {
        let perf_now = (|| -> Option<(js_sys::Function, wasm_bindgen::JsValue)> {
            let global = js_sys::global();
            let perf =
                js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("performance"))
                    .ok()?;
            if perf.is_undefined() || perf.is_null() {
                return None;
            }
            let now = js_sys::Reflect::get(&perf, &wasm_bindgen::JsValue::from_str("now")).ok()?;
            let func = now.dyn_into::<js_sys::Function>().ok()?;
            Some((func, perf))
        })();
        Self { perf_now }
    }

    fn now_us(&self) -> f64 {
        if let Some((func, this)) = &self.perf_now {
            if let Ok(v) = func.call0(this) {
                if let Some(ms) = v.as_f64() {
                    return ms * 1000.0;
                }
            }
        }
        js_sys::Date::now() * 1000.0
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct Clock;

#[cfg(not(target_arch = "wasm32"))]
impl Clock {
    fn new() -> Self {
        Self
    }
    fn now_us(&self) -> f64 {
        use std::sync::OnceLock;
        use std::time::Instant;
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        let epoch = EPOCH.get_or_init(Instant::now);
        epoch.elapsed().as_secs_f64() * 1_000_000.0
    }
}

pub struct PerfState {
    clock: Clock,
    /// EMA smoothing factor per call. 1 s time constant so the displayed
    /// value tracks reality without flickering with raw 1 ms timer steps.
    alpha: f32,
    prev_start_us: f64,
    total_ms_ema: f32,
    period_ms_ema: f32,
}

impl PerfState {
    pub fn new(dt_secs: f32) -> Self {
        let tau = 10.0_f32;
        let alpha = 1.0 - (-dt_secs / tau).exp();
        Self {
            clock: Clock::new(),
            alpha,
            prev_start_us: f64::NAN,
            total_ms_ema: f32::NAN,
            period_ms_ema: f32::NAN,
        }
    }

    #[inline]
    pub fn frame_start(&self) -> f64 {
        self.clock.now_us()
    }

    pub fn frame_end(&mut self, start_us: f64, out: &mut [f32]) {
        let end_us = self.clock.now_us();
        let elapsed_ms = ((end_us - start_us).max(0.0) / 1000.0) as f32;
        self.total_ms_ema = ema_step(self.total_ms_ema, elapsed_ms, self.alpha);

        if self.prev_start_us.is_finite() {
            let period_ms = ((start_us - self.prev_start_us).max(0.0) / 1000.0) as f32;
            self.period_ms_ema = ema_step(self.period_ms_ema, period_ms, self.alpha);
        }
        self.prev_start_us = start_us;

        out[PERF_TOTAL_MS] = self.total_ms_ema;
        out[PERF_FREQ_HZ] = if self.period_ms_ema.is_finite() && self.period_ms_ema > 0.0 {
            1000.0 / self.period_ms_ema
        } else {
            f32::NAN
        };
    }
}

#[inline]
fn ema_step(state: f32, sample: f32, alpha: f32) -> f32 {
    if state.is_nan() {
        sample
    } else {
        state + alpha * (sample - state)
    }
}
