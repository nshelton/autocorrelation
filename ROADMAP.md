# Roadmap

Tracks features that are agreed-on but deferred. Each entry is a candidate for a future spec → plan → implementation cycle. Items are not in priority order within a section.

## In progress


1. Tau period does a "quadratic fit" on the score, but i don' thtink it's working right. I see that it could be off by sub-lag and that accumulates over say 16 beats, so the lastpeak in the autocorr could be like 4 lags away from the beat. not a huge deal in practice, but maybe we can do a subpixel refinement that actually takes into account K * tau_float 


2. halftime folding - we definitely see this. maybe fit an idea of a hierarchial grid; i do want something to see measure/ beat / downbeats whtever. 



## Next


1. paramter router - route levels and beat saw sine into other parameters, use this as a preset
2. preset controller, each module with its parameter set can have a bank of presets

## Beatdetector improvements
1. Spectral flux OSS (replace the half-wave-rectified RMS-diff proxy with the paper's per-bin log-magnitude flux)
2. Octave decider (paper §II-C step 4) — heuristic or simple ML to multiply/halve the reported tempo
3. `beats_per_measure` detection (deferred from rewrite)
4. Onset / downbeat detection (which `k·τ` of the beat grid is the bar boundary)

## modules
1. camera
2. postproc
    hdr? tonemapper
    glitch
    halftone

3. physics? orbital mechanics, collisions
4. lighting
5. meshes - audio mesh ? Fft history buffer?



then add little cubes for highs

### Developer experience
