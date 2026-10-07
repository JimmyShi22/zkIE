//! Minimal Fiat–Shamir transcript for the extension-chain GKR protocol.
//!
//! Poseidon2 duplex over Goldilocks sampling FULL quadratic-extension
//! challenges (`EF`). Fully deterministic: every challenge derives from the
//! transcript contents — protocol label, dimensions, commitment roots, and
//! every round message absorbed BEFORE its challenge is sampled. No PRNG, no
//! caller-provided seeds. The same fixed public Poseidon2 permutation is used
//! on both prover and verifier sides.

use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_goldilocks::Poseidon2Goldilocks;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::common::field::{EF, Goldilocks, PrimeCharacteristicRing};
use crate::common::sumcheck::RoundPolyF;
use crate::pcs::whir::Commitment;

type Perm = Poseidon2Goldilocks<16>;
type Challenger = DuplexChallenger<Goldilocks, Perm, 16, 8>;

pub struct ETranscript {
    challenger: Challenger,
}

impl ETranscript {
    /// New transcript domain-separated by the protocol label, the dimensions
    /// (in the canonical order `[m, d, k, n]`), and the commitment roots (in
    /// the canonical order `[X, W1, H, W2, Y]`).
    pub fn new(protocol: &str, dims: &[usize], roots: &[&Commitment]) -> Self {
        // Fixed public permutation: the transcript must be reproducible by
        // the verifier, so no randomness here.
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let mut challenger = Challenger::new(perm);
        for &b in protocol.as_bytes() {
            challenger.observe(Goldilocks::from_u64(b as u64));
        }
        for &d in dims {
            challenger.observe(Goldilocks::from_u64(d as u64));
        }
        for r in roots {
            challenger.observe(*r);
        }
        ETranscript { challenger }
    }

    /// Absorb one extension-field element (a claim value or round coefficient).
    pub fn absorb(&mut self, v: EF) {
        self.challenger.observe_algebra_element(v);
    }

    /// Absorb a degree-2 round polynomial (three coefficients).
    pub fn absorb_round(&mut self, r: &RoundPolyF<EF>) {
        self.absorb(r.c0);
        self.absorb(r.c1);
        self.absorb(r.c2);
    }

    /// Sample one FULL extension-field challenge.
    pub fn sample(&mut self) -> EF {
        self.challenger.sample_algebra_element()
    }

    /// Sample `len` challenges.
    pub fn sample_vec(&mut self, len: usize) -> Vec<EF> {
        (0..len).map(|_| self.sample()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;
    use crate::pcs::whir::Whir;

    #[test]
    fn transcript_is_deterministic_and_roots_bind() {
        let mut rng = XorShift64::new(7);
        let whir = Whir::new_testing(6);
        let f: Vec<Goldilocks> = (0..64).map(|_| rng.field()).collect();
        let (root, _, _) = whir.commit(&f);
        let make = || {
            let mut t = ETranscript::new("test/v1", &[1, 2, 3, 4], &[&root]);
            t.sample_vec(4)
        };
        assert_eq!(make(), make(), "transcript must be deterministic");

        // A different root must yield different challenges.
        let mut g = f.clone();
        g[0] = g[0] + Goldilocks::ONE;
        let (root2, _, _) = whir.commit(&g);
        let mut t = ETranscript::new("test/v1", &[1, 2, 3, 4], &[&root2]);
        assert_ne!(make(), t.sample_vec(4));
    }
}
