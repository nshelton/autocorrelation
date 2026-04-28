# Roadmap

Tracks features that are agreed-on but deferred. Each entry is a candidate for a future spec → plan → implementation cycle. Items are not in priority order within a section.

## In progress

(none)

## Next


1. paramter router - route levels and beat saw sine into other parameters, use this as a preset
2. preset controller, each module with its parameter set can have a bank of presets

## Beatdetector improvements
3. generate a sawtooth wave -assume 4 beats ? assume we got the measure ? 
4. onset detection for down beat - this is hard

## modules
1. camera
2. postproc
    hdr? tonemapper
    glitch
    halftone

3. physics? orbital mechanics, collisions
4. lighting
5. meshes - audio mesh ? Fft history buffer?


## autogain
- should this be on the signal? on on the rms 
- probably rms; per-level

sketch:
spawn sphere on beat/ threshold low 
do physics
then add little cubes for highs

### Developer experience

- Synth demo source for debug content richer than a sine (chord, sweep, noise + tone, etc.)
- Trace the "Multiple instances of Three.js" warning that still appears in dev despite `resolve.dedupe`

### Performance
- Migrate autocorrelation to FFT-based (Wiener–Khinchin: ACF = IFFT(|FFT(x)|²)) once the v3 direct implementation has proven the visualization is correct. O(N log N) vs O(N²); reuses the existing realfft planner. Folds the RMS-ACF onto the same code path.

### Visual
- Particle system + post-processing (driven by spectrum / RMS / autocorrelation features)

# Review
 need to do a code review checkin at some point;
 - dsp one big file; could it be modular? what are the modules really though


## Shipped

- **v3.2** (tag pending): rms_acf decaying EMA accumulator + top-10 sub-bin tempo peak picking with PeakMarkers visualization; new `dsp.accumTauSecs` analysis param
- **v3.1** (tag `v3.1.0`): runtime-tunable analysis parameters via ParamStore + tweakpane (window size, RMS history length, hop size, smoothing τ, dB floor); persisted to localStorage; live worklet reconfiguration with dispose+rebuild of LineRenderers; FPS overlay moved to top-left
- **v3** (tag `v3.0.0`): buffer autocorrelation + RMS-envelope autocorrelation (with mean-subtracted detrending) as two new line strips; two new camera presets (keys 5 and 6); five-line vertical layout
- **v2** (tag `v2.0.0`): FFT spectrum (log frequency, dB, smoothed), rolling RMS history, tab audio source, internal test source (T key), FPS overlay, four camera presets, overlapped FFT
- **v1** (tag `v1.0.0`): mic input → AudioWorklet → Rust/WASM → Three.js WebGPU LineRenderer with camera rig
