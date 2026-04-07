//! Tiny xorshift64 PRNG used by seeded random built-ins.
//!
//! This is intentionally minimal: we do not pull in an external rand
//! crate. It is NOT cryptographically secure.

/// Xorshift64 state. Seeded deterministically from a u64.
pub(crate) struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG from a seed. A seed of 0 is remapped to a
    /// non-zero constant because xorshift cannot leave the zero state.
    pub(crate) fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }

    /// Raw 64-bit xorshift step.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f64 in [0, 1). Uses the top 53 bits for full mantissa.
    pub(crate) fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 bits
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard normal via Box-Muller. Uses two uniforms per sample;
    /// we just recompute each call for simplicity (no caching).
    pub(crate) fn next_normal(&mut self) -> f64 {
        // Avoid log(0) by clamping u1 away from zero.
        let mut u1 = self.next_f64();
        if u1 < 1e-300 {
            u1 = 1e-300;
        }
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }
}
