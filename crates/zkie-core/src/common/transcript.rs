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

    /// Absorb a base-field element (public scalars like op factors/constants).
    pub fn absorb_base(&mut self, v: Goldilocks) {
        self.challenger.observe(v);
    }

    /// Absorb a commitment root (for two-phase transcripts where some roots
    /// are known only after earlier challenges are sampled).
    pub fn absorb_commitment(&mut self, c: &Commitment) {
        self.challenger.observe(c);
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

// ==== typed two-phase transcript schedule ====
//
// Fixes a prover/verifier challenge-stream divergence class discovered in the
// lookup reducer work: when a DERIVED commitment (one whose VALUES depend on
// challenges sampled from the transcript — e.g. a LogUp fractional-tree
// tensor that depends on alpha/beta) is absorbed into the transcript, the
// absorption order is part of the protocol and must be identical on both
// sides. The naive pattern "sample alpha/beta from one transcript, build the
// tree, then build a second transcript over all roots" silently produced two
// different streams.
//
// The state machine structurally enforces the PHASE ORDER ONLY:
//   Initial -> ChallengesSampled -> Ready, strictly, with no repeats/skips
//   (misuse panics). The per-segment absorb-then-sample order is NOT
//   enforced: the segment API is deliberately low-level, and the caller must
//   follow the protocol's segment schedule (absorb each round message BEFORE
//   sampling the challenges it feeds). No type-length framing is implied:
//   fixing the protocol shape, dims, and arity of the challenge stream is the
//   caller's responsibility.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Only `sample_derived_challenges` is allowed.
    Initial,
    /// Derived challenges sampled; only `absorb_derived_roots` is allowed.
    ChallengesSampled,
    /// Derived roots absorbed; only sequential segment activity is allowed.
    Ready,
}

/// Typed two-phase EF transcript schedule (see the module-level note above).
pub struct TwoPhaseTranscript {
    t: ETranscript,
    phase: Phase,
}

impl TwoPhaseTranscript {
    /// Phase 1: a fresh transcript domain-separated by the protocol label,
    /// dimensions, and the initial statement roots.
    pub fn new(protocol: &str, dims: &[usize], initial_roots: &[&Commitment]) -> Self {
        TwoPhaseTranscript {
            t: ETranscript::new(protocol, dims, initial_roots),
            phase: Phase::Initial,
        }
    }

    /// Sample the derived challenge pair (e.g. LogUp `alpha`, `beta`). Only
    /// valid in the `Initial` phase; moves to `ChallengesSampled` (so it can
    /// never be called twice).
    pub fn sample_derived_challenges(&mut self) -> (EF, EF) {
        assert_eq!(
            self.phase,
            Phase::Initial,
            "derived challenges must be sampled exactly once, before the derived roots are absorbed"
        );
        let out = (self.t.sample(), self.t.sample());
        self.phase = Phase::ChallengesSampled;
        out
    }

    /// Phase 2: absorb the derived commitment roots (commitments whose values
    /// depend on the derived challenges). Only valid in `ChallengesSampled`
    /// (never before the challenges are sampled, never twice); moves to
    /// `Ready`.
    pub fn absorb_derived_roots(&mut self, roots: &[&Commitment]) {
        assert_eq!(
            self.phase,
            Phase::ChallengesSampled,
            "derived roots must be absorbed exactly once, after the derived challenges are sampled"
        );
        for r in roots {
            self.t.absorb_commitment(r);
        }
        self.phase = Phase::Ready;
    }

    fn require_ready(&self) {
        assert_eq!(
            self.phase,
            Phase::Ready,
            "per-segment activity requires the derived roots to be absorbed"
        );
    }

    /// Absorb one extension-field element (a claim value or round message).
    pub fn absorb(&mut self, v: EF) {
        self.require_ready();
        self.t.absorb(v);
    }

    /// Absorb a base-field element (public scalars).
    pub fn absorb_base(&mut self, v: Goldilocks) {
        self.require_ready();
        self.t.absorb_base(v);
    }

    /// Absorb a degree-2 round polynomial (three coefficients).
    pub fn absorb_round(&mut self, r: &RoundPolyF<EF>) {
        self.require_ready();
        self.t.absorb_round(r);
    }

    /// Sample one FULL extension-field challenge.
    pub fn sample(&mut self) -> EF {
        self.require_ready();
        self.t.sample()
    }

    /// Sample `len` challenges.
    pub fn sample_vec(&mut self, len: usize) -> Vec<EF> {
        self.require_ready();
        self.t.sample_vec(len)
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

    /// Run one complete two-phase schedule and return the full challenge
    /// stream (derived pair + per-segment samples).
    fn run_two_phase(
        initial: &[&Commitment],
        derived: &[&Commitment],
        round_vals: &[EF],
    ) -> Vec<EF> {
        let mut s = TwoPhaseTranscript::new("test/two-phase/v1", &[7], initial);
        let (a, b) = s.sample_derived_challenges();
        s.absorb_derived_roots(derived);
        let mut stream = vec![a, b];
        for &v in round_vals {
            s.absorb(v);
            stream.push(s.sample());
        }
        stream
    }

    /// Same inputs must produce the identical complete challenge stream.
    #[test]
    fn two_phase_schedule_is_deterministic() {
        let mut rng = XorShift64::new(0x2A);
        let whir = Whir::new_testing(6);
        let f: Vec<Goldilocks> = (0..64).map(|_| rng.field()).collect();
        let (r1, _, _) = whir.commit(&f);
        let (r2, _, _) = whir.commit(&f);
        let mut g = f.clone();
        g[0] = g[0] + Goldilocks::ONE;
        let (dr, _, _) = whir.commit(&g);
        let vals: Vec<EF> = (0..3).map(|_| EF::from(rng.field())).collect();
        let a = run_two_phase(&[&r1, &r2], &[&dr], &vals);
        let b = run_two_phase(&[&r1, &r2], &[&dr], &vals);
        assert_eq!(a, b, "identical inputs must produce the identical stream");
        assert_eq!(a.len(), 2 + 3);
    }

    /// An altered INITIAL root must change the derived challenges AND every
    /// subsequent challenge.
    #[test]
    fn two_phase_initial_root_change_shifts_whole_stream() {
        let mut rng = XorShift64::new(0x2B);
        let whir = Whir::new_testing(6);
        let f: Vec<Goldilocks> = (0..64).map(|_| rng.field()).collect();
        let mut g = f.clone();
        g[0] = g[0] + Goldilocks::ONE;
        let (r1, _, _) = whir.commit(&f);
        let (r1b, _, _) = whir.commit(&g);
        let (r2, _, _) = whir.commit(&f);
        let (dr, _, _) = whir.commit(&g);
        let vals: Vec<EF> = (0..2).map(|_| EF::from(rng.field())).collect();
        let a = run_two_phase(&[&r1, &r2], &[&dr], &vals);
        let b = run_two_phase(&[&r1b, &r2], &[&dr], &vals);
        assert_ne!(a[0], b[0], "derived challenge alpha must change");
        assert_ne!(a[1], b[1], "derived challenge beta must change");
        assert_ne!(&a[2..], &b[2..], "segment challenges must change");
    }

    /// An altered DERIVED root must leave the derived challenges unchanged
    /// (they are sampled before the derived roots are absorbed) but shift
    /// every segment challenge.
    #[test]
    fn two_phase_derived_root_change_shifts_segment_challenges_only() {
        let mut rng = XorShift64::new(0x2C);
        let whir = Whir::new_testing(6);
        let f: Vec<Goldilocks> = (0..64).map(|_| rng.field()).collect();
        let (r1, _, _) = whir.commit(&f);
        let (r2, _, _) = whir.commit(&f);
        let (dr, _, _) = whir.commit(&f);
        let mut g = f.clone();
        g[0] = g[0] + Goldilocks::ONE;
        let (drb, _, _) = whir.commit(&g);
        let vals: Vec<EF> = (0..2).map(|_| EF::from(rng.field())).collect();
        let a = run_two_phase(&[&r1, &r2], &[&dr], &vals);
        let b = run_two_phase(&[&r1, &r2], &[&drb], &vals);
        assert_eq!(a[0], b[0], "alpha must not depend on the derived root");
        assert_eq!(a[1], b[1], "beta must not depend on the derived root");
        assert_ne!(&a[2..], &b[2..], "segment challenges must change");
    }

    /// A changed prior round message must shift the immediately following
    /// challenge.
    #[test]
    fn two_phase_prior_round_shifts_next_challenge() {
        let mut rng = XorShift64::new(0x2D);
        let whir = Whir::new_testing(6);
        let f: Vec<Goldilocks> = (0..64).map(|_| rng.field()).collect();
        let (r1, _, _) = whir.commit(&f);
        let (dr, _, _) = whir.commit(&f);
        let v1 = EF::from(rng.field());
        let v2 = v1 + EF::ONE;
        let a = run_two_phase(&[&r1], &[&dr], &[v1]);
        let b = run_two_phase(&[&r1], &[&dr], &[v2]);
        assert_ne!(a[2], b[2], "changed round message must change the next challenge");
    }

    /// Phase misuse is a programming error and must panic with a clear
    /// message (the schedule is a single canonical order).
    #[test]
    fn two_phase_misuse_panics() {
        let mut rng = XorShift64::new(0x2E);
        let whir = Whir::new_testing(6);
        let f: Vec<Goldilocks> = (0..64).map(|_| rng.field()).collect();
        let (r1, _, _) = whir.commit(&f);
        let (dr, _, _) = whir.commit(&f);

        // Derived roots absorbed BEFORE the derived challenges are sampled.
        let r = std::panic::catch_unwind(|| {
            let mut s = TwoPhaseTranscript::new("p/v1", &[7], &[&r1]);
            s.absorb_derived_roots(&[&dr]);
        });
        assert!(r.is_err());

        // Derived challenges sampled twice.
        let r = std::panic::catch_unwind(|| {
            let mut s = TwoPhaseTranscript::new("p/v1", &[7], &[&r1]);
            let _ = s.sample_derived_challenges();
            let _ = s.sample_derived_challenges();
        });
        assert!(r.is_err());

        // Absorbing derived roots twice.
        let r = std::panic::catch_unwind(|| {
            let mut s = TwoPhaseTranscript::new("p/v1", &[7], &[&r1]);
            let _ = s.sample_derived_challenges();
            s.absorb_derived_roots(&[&dr]);
            s.absorb_derived_roots(&[&dr]);
        });
        assert!(r.is_err());

        // Segment sampling before the derived roots are absorbed.
        let r = std::panic::catch_unwind(|| {
            let mut s = TwoPhaseTranscript::new("p/v1", &[7], &[&r1]);
            let _ = s.sample();
        });
        assert!(r.is_err());

        // Derived challenges after the derived roots are absorbed.
        let r = std::panic::catch_unwind(|| {
            let mut s = TwoPhaseTranscript::new("p/v1", &[7], &[&r1]);
            let _ = s.sample_derived_challenges();
            s.absorb_derived_roots(&[&dr]);
            let _ = s.sample_derived_challenges();
        });
        assert!(r.is_err());
    }
}
