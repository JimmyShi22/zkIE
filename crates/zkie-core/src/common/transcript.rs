//! Fiat-Shamir transcript: a deterministic challenge source that binds every
//! challenge to the statement and to any proof messages absorbed so far.
//!
//! This replaces the free-running `XorShift64` PRNG used to sample evaluation
//! points. `Transcript` derives each challenge from a SHA-256 chain over
//! everything absorbed, so the prover cannot choose favourable points and the
//! verifier can replay the exact same challenges from the proof.

use crate::common::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64, P};
use sha2::{Digest, Sha256};

/// A deterministic challenge stream over absorbed bytes.
#[derive(Clone)]
pub struct Transcript {
    state: [u8; 32],
    counter: u64,
}

impl Transcript {
    /// Start a transcript with a domain separator (kept distinct per protocol).
    pub fn new(domain: &[u8]) -> Self {
        let state: [u8; 32] = Sha256::digest(domain).into();
        Transcript { state, counter: 0 }
    }

    /// Absorb arbitrary bytes into the challenge state.
    pub fn absorb(&mut self, data: &[u8]) {
        let mut h = Sha256::new();
        h.update(self.state);
        h.update(data);
        self.state = h.finalize().into();
    }

    /// Absorb a single canonical field element.
    pub fn absorb_field(&mut self, x: Goldilocks) {
        self.absorb(&x.as_canonical_u64().to_le_bytes());
    }

    /// Absorb a single `u64`.
    pub fn absorb_u64(&mut self, x: u64) {
        self.absorb(&x.to_le_bytes());
    }

    /// Absorb a slice of canonical field elements.
    pub fn absorb_fields(&mut self, xs: &[Goldilocks]) {
        for &x in xs {
            self.absorb_field(x);
        }
    }

    /// Sample one canonical `Goldilocks` element (rejection-sampled to `< P`).
    pub fn field(&mut self) -> Goldilocks {
        loop {
            let mut h = Sha256::new();
            h.update(self.state);
            h.update(self.counter.to_le_bytes());
            self.counter = self.counter.wrapping_add(1);
            let digest: [u8; 32] = h.finalize().into();
            let candidate = u64::from_le_bytes(digest[..8].try_into().unwrap());
            if candidate < P {
                return Goldilocks::from_u64(candidate);
            }
        }
    }

    /// Sample a vector of canonical field elements.
    pub fn field_vec(&mut self, n: usize) -> Vec<Goldilocks> {
        (0..n).map(|_| self.field()).collect()
    }

    /// Sample a raw `u64` (no rejection sampling); used for non-challenge
    /// randomness that only needs to be deterministic.
    pub fn next_u64(&mut self) -> u64 {
        let mut h = Sha256::new();
        h.update(self.state);
        h.update(self.counter.to_le_bytes());
        self.counter = self.counter.wrapping_add(1);
        let digest: [u8; 32] = h.finalize().into();
        u64::from_le_bytes(digest[..8].try_into().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_replay() {
        let mut a = Transcript::new(b"zkie-test");
        a.absorb(&[1u8, 2, 3]);
        let mut b = Transcript::new(b"zkie-test");
        b.absorb(&[1u8, 2, 3]);
        for _ in 0..16 {
            assert_eq!(a.field(), b.field());
        }
    }

    #[test]
    fn absorb_changes_challenges() {
        let mut a = Transcript::new(b"zkie-test");
        let mut b = Transcript::new(b"zkie-test");
        b.absorb(&[7u8]);
        assert_ne!(a.field(), b.field());
    }

    #[test]
    fn field_is_canonical() {
        let mut t = Transcript::new(b"zkie-test");
        for _ in 0..1000 {
            assert!(t.field().as_canonical_u64() < P);
        }
    }
}
impl crate::common::field::ChallengeSource for Transcript {
    fn field(&mut self) -> Goldilocks {
        Transcript::field(self)
    }
    fn next_u64(&mut self) -> u64 {
        Transcript::next_u64(self)
    }
}
