//! Synthesized analog drum kit — the percussion counterpart to `PolySynth`.
//!
//! The codebase has no sample-loading infrastructure (everything is synthesis),
//! and a ReDrum-style step grid only needs a handful of one-shot percussion
//! voices, so each lane is *synthesized* from a sine "tone" component (with an
//! optional pitch sweep, for kicks and toms) plus a high-passed noise component
//! (snare, hats, claps, cymbals). Both components have independent exponential
//! decays.
//!
//! A `DrumKit` is a fixed bank of one `DrumVoice` per lane, addressed by MIDI
//! note: the step grid emits notes at the lane MIDI numbers (kick = 36, snare =
//! 38, …) and `note_on` routes each back to its lane. Voices are one-shots:
//! `note_on` retriggers, `note_off` is ignored (classic drum-machine "decay"
//! mode — the hit always rings its full envelope regardless of note length).
//!
//! The lane MIDI mapping here MUST agree with `src/sequencer/drumkit.ts`
//! (the UI side that authors the notes). It is the drum analogue of how
//! `INSTRUMENT_DEFAULTS` mirrors the `Voice` defaults.

/// One drum lane's type. The ordering of `LANES` is the kit's lane order (top
/// to bottom in the grid); adding a lane is a new variant + a `LANES` entry + a
/// `spec()` arm + the matching row in `drumkit.ts`.
#[derive(Clone, Copy, PartialEq)]
enum DrumType {
    Kick,
    Snare,
    ClosedHat,
    OpenHat,
    Clap,
    Rim,
    TomLo,
    TomMid,
    TomHi,
    Cymbal,
}

/// Lane order = grid row order. `DrumKit` allocates one voice per entry.
const LANES: [DrumType; 10] = [
    DrumType::Kick,
    DrumType::Snare,
    DrumType::ClosedHat,
    DrumType::OpenHat,
    DrumType::Clap,
    DrumType::Rim,
    DrumType::TomLo,
    DrumType::TomMid,
    DrumType::TomHi,
    DrumType::Cymbal,
];

/// Static synthesis recipe for a lane. Times (`*_tau`) are exponential-decay
/// time constants in seconds (level reaches 1/e after `tau`); the audible tail
/// is a few `tau`. Frequencies in Hz.
#[derive(Clone, Copy)]
struct DrumSpec {
    /// MIDI note that routes to this lane — must match `drumkit.ts`.
    midi: i32,
    // Tonal (sine) component.
    tone_freq0: f32, // start frequency
    tone_freq1: f32, // pitch sweeps toward this (only when pitch_tau > 0)
    pitch_tau: f32,  // pitch-envelope time constant; 0 → fixed at tone_freq0
    tone_level: f32, // 0 → no tonal component
    tone_tau: f32,   // tonal amp-decay time constant
    // Noise component (high-passed white noise).
    noise_level: f32, // 0 → no noise component
    noise_tau: f32,   // noise amp-decay time constant
    noise_hp: f32,    // one-pole high-pass cutoff; 0 → no filtering
    /// Voices sharing a nonzero choke group cut each other off (open/closed hat).
    choke_group: u8,
}

fn spec(kind: DrumType) -> DrumSpec {
    // Defaults: a silent, no-op recipe each arm overrides the relevant fields of.
    let base = DrumSpec {
        midi: 0,
        tone_freq0: 100.0,
        tone_freq1: 100.0,
        pitch_tau: 0.0,
        tone_level: 0.0,
        tone_tau: 0.1,
        noise_level: 0.0,
        noise_tau: 0.1,
        noise_hp: 0.0,
        choke_group: 0,
    };
    match kind {
        // Sine thump with a fast downward pitch sweep — the body of the kick.
        DrumType::Kick => DrumSpec {
            midi: 36,
            tone_freq0: 160.0,
            tone_freq1: 45.0,
            pitch_tau: 0.03,
            tone_level: 1.0,
            tone_tau: 0.12,
            ..base
        },
        // Tonal "ring" (~190 Hz) under a band of mid noise.
        DrumType::Snare => DrumSpec {
            midi: 38,
            tone_freq0: 190.0,
            tone_freq1: 190.0,
            tone_level: 0.5,
            tone_tau: 0.07,
            noise_level: 0.9,
            noise_tau: 0.10,
            noise_hp: 1200.0,
            ..base
        },
        // Short burst of bright noise.
        DrumType::ClosedHat => DrumSpec {
            midi: 42,
            noise_level: 0.7,
            noise_tau: 0.025,
            noise_hp: 7000.0,
            choke_group: 1,
            ..base
        },
        // Long bright noise — cut by the closed hat (shared choke group).
        DrumType::OpenHat => DrumSpec {
            midi: 46,
            noise_level: 0.6,
            noise_tau: 0.30,
            noise_hp: 7000.0,
            choke_group: 1,
            ..base
        },
        // Mid noise, snappy — a clap approximated as a single shaped burst.
        DrumType::Clap => DrumSpec {
            midi: 39,
            noise_level: 0.9,
            noise_tau: 0.10,
            noise_hp: 1200.0,
            ..base
        },
        // High click — short tonal tick plus a touch of noise.
        DrumType::Rim => DrumSpec {
            midi: 37,
            tone_freq0: 1700.0,
            tone_freq1: 1700.0,
            tone_level: 0.7,
            tone_tau: 0.02,
            noise_level: 0.3,
            noise_tau: 0.02,
            noise_hp: 2000.0,
            ..base
        },
        DrumType::TomLo => DrumSpec {
            midi: 41,
            tone_freq0: 110.0,
            tone_freq1: 80.0,
            pitch_tau: 0.10,
            tone_level: 1.0,
            tone_tau: 0.20,
            ..base
        },
        DrumType::TomMid => DrumSpec {
            midi: 45,
            tone_freq0: 160.0,
            tone_freq1: 120.0,
            pitch_tau: 0.10,
            tone_level: 1.0,
            tone_tau: 0.18,
            ..base
        },
        DrumType::TomHi => DrumSpec {
            midi: 48,
            tone_freq0: 230.0,
            tone_freq1: 170.0,
            pitch_tau: 0.09,
            tone_level: 1.0,
            tone_tau: 0.15,
            ..base
        },
        // Long shimmer of very bright noise.
        DrumType::Cymbal => DrumSpec {
            midi: 49,
            noise_level: 0.5,
            noise_tau: 0.60,
            noise_hp: 9000.0,
            ..base
        },
    }
}

/// Time constant → per-sample exponential-decay multiplier. `tau <= 0` returns
/// 0 so the component is silent immediately (callers also zero its env level).
fn decay_coef(tau: f32, sample_rate: f32) -> f32 {
    if tau <= 0.0 {
        0.0
    } else {
        (-1.0 / (tau * sample_rate)).exp()
    }
}

/// Cheap deterministic white-noise source (xorshift32). Avoids pulling the
/// default-hasher / `getrandom` RNG into the wasm build; seeded per lane so the
/// hats/snare don't share an identical (phase-locked) noise sequence.
struct Noise {
    state: u32,
}

impl Noise {
    fn new(seed: u32) -> Noise {
        Noise { state: seed | 1 } // never 0 — xorshift would lock up
    }

    fn next(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        // u32 → [-1, 1)
        (x as f32 / 2_147_483_648.0) - 1.0
    }
}

/// One drum lane: a sine "tone" oscillator + a high-passed noise generator, each
/// gated by its own exponential decay. Pre-allocated; `trigger` resets state, so
/// the realtime path never allocates.
struct DrumVoice {
    sample_rate: f32,
    spec: DrumSpec,

    // Tonal oscillator.
    phase: f32,
    pitch_env: f32,
    tone_env: f32,

    // Noise + its one-pole high-pass state.
    noise: Noise,
    noise_env: f32,
    hp_prev_in: f32,
    hp_prev_out: f32,
    hp_a: f32, // high-pass coefficient (1 → passthrough when no cutoff)

    // Per-trigger decay multipliers (copied from spec; `choke` overwrites them).
    pitch_coef: f32,
    tone_coef: f32,
    noise_coef: f32,

    velocity: f32,
}

impl DrumVoice {
    fn new(sample_rate: f32, spec: DrumSpec, seed: u32) -> DrumVoice {
        // One-pole high-pass coefficient a = RC / (RC + dt); y = a(y₋₁ + x − x₋₁).
        let hp_a = if spec.noise_hp > 0.0 {
            let rc = 1.0 / (2.0 * std::f32::consts::PI * spec.noise_hp);
            let dt = 1.0 / sample_rate;
            rc / (rc + dt)
        } else {
            1.0 // passthrough
        };
        DrumVoice {
            sample_rate,
            spec,
            phase: 0.0,
            pitch_env: 0.0,
            tone_env: 0.0,
            noise: Noise::new(seed),
            noise_env: 0.0,
            hp_prev_in: 0.0,
            hp_prev_out: 0.0,
            hp_a,
            pitch_coef: 0.0,
            tone_coef: 0.0,
            noise_coef: 0.0,
            velocity: 1.0,
        }
    }

    /// Strike the drum: reset envelopes/phase to the start of the hit.
    fn trigger(&mut self, velocity: f32) {
        self.velocity = velocity.clamp(0.0, 1.0);
        self.phase = 0.0;
        self.pitch_env = if self.spec.pitch_tau > 0.0 { 1.0 } else { 0.0 };
        self.tone_env = if self.spec.tone_level > 0.0 { 1.0 } else { 0.0 };
        self.noise_env = if self.spec.noise_level > 0.0 { 1.0 } else { 0.0 };
        self.pitch_coef = decay_coef(self.spec.pitch_tau, self.sample_rate);
        self.tone_coef = decay_coef(self.spec.tone_tau, self.sample_rate);
        self.noise_coef = decay_coef(self.spec.noise_tau, self.sample_rate);
        self.hp_prev_in = 0.0;
        self.hp_prev_out = 0.0;
    }

    /// Cut the voice quickly without fully muting (choke groups + pause/stop):
    /// replace the running decays with a fast ~4 ms fade so the tail doesn't pop.
    fn choke(&mut self) {
        let fast = decay_coef(0.004, self.sample_rate);
        self.tone_coef = self.tone_coef.min(fast);
        self.noise_coef = self.noise_coef.min(fast);
    }

    fn is_active(&self) -> bool {
        // ~-60 dBFS floor: below this a voice is inaudible, so free it (and stop
        // rendering it) rather than chasing the exponential tail toward zero.
        self.tone_env > 1e-3 || self.noise_env > 1e-3
    }

    /// Add this voice's contribution into `out` (does not clear it), scaled by
    /// the kit's master `gain`. No-op when the voice has decayed to silence.
    fn render_add(&mut self, out: &mut [f32], gain: f32) {
        if !self.is_active() {
            return;
        }
        let has_tone = self.spec.tone_level > 0.0;
        let has_noise = self.spec.noise_level > 0.0;
        let amp = self.velocity * gain;

        for s in out.iter_mut() {
            let mut sample = 0.0;

            if has_tone {
                let freq = if self.spec.pitch_tau > 0.0 {
                    self.spec.tone_freq1
                        + (self.spec.tone_freq0 - self.spec.tone_freq1) * self.pitch_env
                } else {
                    self.spec.tone_freq0
                };
                let osc = (self.phase * std::f32::consts::TAU).sin();
                self.phase += freq / self.sample_rate;
                if self.phase >= 1.0 {
                    self.phase -= 1.0;
                }
                self.pitch_env *= self.pitch_coef;
                sample += osc * self.tone_env * self.spec.tone_level;
                self.tone_env *= self.tone_coef;
            }

            if has_noise {
                let x = self.noise.next();
                // One-pole high-pass.
                let y = self.hp_a * (self.hp_prev_out + x - self.hp_prev_in);
                self.hp_prev_in = x;
                self.hp_prev_out = y;
                sample += y * self.noise_env * self.spec.noise_level;
                self.noise_env *= self.noise_coef;
            }

            *s += sample * amp;
        }
    }
}

/// A bank of one drum voice per lane, addressed by MIDI note. The sequencer
/// treats it like a `PolySynth` (note_on / render_add / set_param) so it can
/// live in the same per-track instrument slot.
pub struct DrumKit {
    voices: Vec<DrumVoice>,
    gain: f32,
}

impl DrumKit {
    pub fn new(sample_rate: f32) -> DrumKit {
        let voices = LANES
            .iter()
            .enumerate()
            // Seed each voice distinctly so noise lanes aren't correlated.
            .map(|(i, &kind)| {
                DrumVoice::new(sample_rate, spec(kind), 0x9E37_79B9u32.wrapping_mul(i as u32 + 1))
            })
            .collect();
        DrumKit { voices, gain: 0.9 }
    }

    /// Strike the lane whose MIDI note matches `midi`. Unknown notes are
    /// ignored. Triggering a choke-group member first cuts the others in its
    /// group (closed hat silences open hat).
    pub fn note_on(&mut self, midi: f32, velocity: f32) {
        let m = midi.round() as i32;
        let Some(idx) = self.voices.iter().position(|v| v.spec.midi == m) else {
            return;
        };
        let group = self.voices[idx].spec.choke_group;
        if group != 0 {
            for (j, v) in self.voices.iter_mut().enumerate() {
                if j != idx && v.spec.choke_group == group {
                    v.choke();
                }
            }
        }
        self.voices[idx].trigger(velocity);
    }

    /// One-shots ignore note-off (decay mode): a hit rings its full envelope.
    pub fn note_off(&mut self, _midi: f32) {}

    /// Choke every voice — used by the transport on pause/stop so a long tail
    /// (open hat, cymbal) doesn't drone after the clock stops.
    pub fn release_all(&mut self) {
        for v in &mut self.voices {
            v.choke();
        }
    }

    pub fn set_param(&mut self, key: &str, value: f32) {
        if key == "gain" {
            self.gain = value.max(0.0);
        }
    }

    /// Sum every lane's active voice into `out` additively (no clear, no clip) —
    /// the sequencer clears and master-clips around all instruments.
    pub fn render_add(&mut self, out: &mut [f32]) {
        let gain = self.gain;
        for v in &mut self.voices {
            v.render_add(out, gain);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn energy(buf: &[f32]) -> f32 {
        buf.iter().map(|x| x.abs()).sum()
    }

    fn render(kit: &mut DrumKit, n: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; n];
        kit.render_add(&mut buf);
        buf
    }

    #[test]
    fn kick_sounds_then_decays_to_silence() {
        let mut kit = DrumKit::new(SR);
        kit.note_on(36.0, 1.0);
        let head = render(&mut kit, 1024);
        assert!(energy(&head) > 1e-2, "kick should sound, got {}", energy(&head));

        // Run ~1.5 s; a 0.12 s-tau kick is well under the active floor by then.
        for _ in 0..70 {
            render(&mut kit, 1024);
        }
        let tail = render(&mut kit, 1024);
        assert!(energy(&tail) < 1e-3, "kick should decay to silence, got {}", energy(&tail));
    }

    #[test]
    fn unknown_midi_is_ignored() {
        let mut kit = DrumKit::new(SR);
        kit.note_on(60.0, 1.0); // no lane at MIDI 60
        let buf = render(&mut kit, 256);
        assert_eq!(energy(&buf), 0.0, "unmapped note must be silent");
    }

    #[test]
    fn velocity_scales_amplitude() {
        let mut loud = DrumKit::new(SR);
        let mut soft = DrumKit::new(SR);
        loud.note_on(36.0, 1.0);
        soft.note_on(36.0, 0.25);
        let el = energy(&render(&mut loud, 2048));
        let es = energy(&render(&mut soft, 2048));
        assert!(el > es * 2.0, "velocity should scale level: loud {el} vs soft {es}");
    }

    #[test]
    fn closed_hat_chokes_open_hat() {
        // Open hat alone rings long; a closed hat right after should cut it so
        // the later tail is much quieter than the un-choked reference.
        let mut choked = DrumKit::new(SR);
        choked.note_on(46.0, 1.0); // open hat
        render(&mut choked, 256); // let it start ringing
        choked.note_on(42.0, 1.0); // closed hat → chokes the open hat
        // Skip past the closed hat's own short (~25 ms) decay before measuring.
        render(&mut choked, 4096);
        let choked_tail = energy(&render(&mut choked, 4096));

        let mut free = DrumKit::new(SR);
        free.note_on(46.0, 1.0);
        render(&mut free, 256 + 4096);
        let free_tail = energy(&render(&mut free, 4096));

        assert!(
            choked_tail < free_tail * 0.5,
            "closed hat should choke open hat: choked {choked_tail} vs free {free_tail}"
        );
    }

    #[test]
    fn every_lane_makes_sound() {
        for &kind in LANES.iter() {
            let m = spec(kind).midi;
            let mut kit = DrumKit::new(SR);
            kit.note_on(m as f32, 1.0);
            let e = energy(&render(&mut kit, 4096));
            assert!(e > 1e-3, "lane at midi {m} should sound, got {e}");
        }
    }

    #[test]
    fn output_stays_bounded_with_a_full_hit() {
        // Every lane at once, master gain applied — individual voices shouldn't
        // each blow past unity before the sequencer's master soft-clip.
        let mut kit = DrumKit::new(SR);
        for &kind in LANES.iter() {
            kit.note_on(spec(kind).midi as f32, 1.0);
        }
        let buf = render(&mut kit, 1024);
        // Pre-clip we allow some headroom but not a runaway sum.
        assert!(buf.iter().all(|x| x.abs() < 8.0), "voice sum unexpectedly large");
    }
}
