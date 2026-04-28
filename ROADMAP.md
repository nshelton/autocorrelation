# Roadmap

Tracks features that are agreed-on but deferred. Each entry is a candidate for a future spec → plan → implementation cycle. Items are not in priority order within a section.

## In progress

(none)

## Next

### Developer experience
- Vite HMR so most code edits don't reload the page (biggest dev-loop win — page reload re-prompts for tab-capture permission)
- Synth demo source for debug content richer than a sine (chord, sweep, noise + tone, etc.)
- Trace the "Multiple instances of Three.js" warning that still appears in dev despite `resolve.dedupe`

### Performance
- Migrate autocorrelation to FFT-based (Wiener–Khinchin: ACF = IFFT(|FFT(x)|²)) once the v3 direct implementation has proven the visualization is correct. O(N log N) vs O(N²); reuses the existing realfft planner. Folds the RMS-ACF onto the same code path.

### Visual
- Particle system + post-processing (driven by spectrum / RMS / autocorrelation features)

## Shipped

- **v3** (tag `v3.0.0`): buffer autocorrelation + RMS-envelope autocorrelation (with mean-subtracted detrending) as two new line strips; two new camera presets (keys 5 and 6); five-line vertical layout
- **v2** (tag `v2.0.0`): FFT spectrum (log frequency, dB, smoothed), rolling RMS history, tab audio source, internal test source (T key), FPS overlay, four camera presets, overlapped FFT
- **v1** (tag `v1.0.0`): mic input → AudioWorklet → Rust/WASM → Three.js WebGPU LineRenderer with camera rig
