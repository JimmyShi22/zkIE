//! Minimal executable schedule contract (stepping stone before the lookup
//! reducer): proves that the `TwoPhaseTranscript` schedule yields EXACTLY
//! identical EF opening points on the prover and the verifier, with real
//! WHIR openings.
//!
//! Protocol (no lookup algebra):
//! 1. Statement: dimensions plus the initial roots `[r1, r2]` and the derived
//!    root `dr`.
//! 2. `TwoPhaseTranscript::new(label, dims, [r1, r2])`; sample the derived
//!    challenge pair `(alpha, beta)`.
//! 3. Derive a base tensor `tree = t1 + t2 + (a0+a1) + (b0+b1)` from the
//!    challenges, commit it, absorb the derived root.
//! 4. Two named segments, each "absorb a round message, then sample a full-EF
//!    point": segment 1 carries `v1 = sum(t1) + alpha` and opens `t1` and the
//!    tree at `p1`; segment 2 carries `v2 = sum(t2) + beta` and opens `t2` at
//!    `p2`.
//! 5. The verifier reconstructs the SAME schedule from the statement plus the
//!    proof's round messages, and checks each WHIR opening at the
//!    reconstructed point against the claimed value. No witness, no
//!    `open_*` calls — the point identity is enforced by the opening values.
//!
//! This module is a schedule contract only: it does not prove any relation
//! between the tensors beyond the openings, and the round-message VALUES are
//! not themselves authenticated here (they only feed the transcript).

use zkie_core::common::field::{BasedVectorSpace, EF, Goldilocks, PrimeCharacteristicRing, PrimeField64};
use zkie_core::common::mle;
use zkie_core::common::transcript::TwoPhaseTranscript;
use zkie_core::pcs::whir::{Commitment, Point, Proof, Whir};

/// The canonical protocol label.
pub const PROTOCOL: &str = "zkie/ext-schedule/v1";

/// Public statement: tensor arity and the three canonical roots.
#[derive(Clone, Debug)]
pub struct ScheduleStatement {
    pub log_n: usize,
    pub root_1: Commitment,
    pub root_2: Commitment,
    pub root_tree: Commitment,
}

/// The transported proof: the two round messages, the two sampled point
/// vectors (checked against the verifier's reconstruction directly, not only
/// via the openings), and the three openings.
#[derive(Clone)]
pub struct ScheduleProof {
    pub v1: EF,
    pub v2: EF,
    pub p1: Vec<EF>,
    pub p2: Vec<EF>,
    pub open_1: (Proof, EF),
    pub open_2: (Proof, EF),
    pub open_tree: (Proof, EF),
}

fn to_p3_point(our_point: &[EF]) -> Point<EF> {
    Point::new(our_point.iter().rev().cloned().collect())
}

/// Sum of a base tensor as an EF element.
fn sum_ef(t: &[Goldilocks]) -> EF {
    t.iter().fold(EF::ZERO, |acc, &v| acc + EF::from(v))
}

/// Prove the schedule contract. Returns `(statement, proof)` or `None` for
/// malformed inputs.
pub fn prove(
    whir: &Whir,
    t1: &[Goldilocks],
    t2: &[Goldilocks],
    log_n: usize,
) -> Option<(ScheduleStatement, ScheduleProof)> {
    // Validate the WHIR arity first, then size via checked shift: a huge
    // `log_n` must yield `None`, never an overflowing shift.
    if whir.num_variables() != log_n || log_n >= usize::BITS as usize {
        return None;
    }
    let n = 1usize.checked_shl(log_n as u32)?;
    if t1.len() != n || t2.len() != n {
        return None;
    }
    let (root_1, pd_1, proto_1) = whir.commit(t1);
    let (root_2, pd_2, proto_2) = whir.commit(t2);

    let mut s = TwoPhaseTranscript::new(PROTOCOL, &[log_n], &[&root_1, &root_2]);
    let (alpha, beta) = s.sample_derived_challenges();
    // Derive the tree from the challenges (base values only).
    let ca: &[Goldilocks] = alpha.as_basis_coefficients_slice();
    let cb: &[Goldilocks] = beta.as_basis_coefficients_slice();
    let scalar = ca[0] + ca[1] + cb[0] + cb[1];
    let tree: Vec<Goldilocks> = (0..n).map(|i| t1[i] + t2[i] + scalar).collect();
    let (root_tree, pd_tree, proto_tree) = whir.commit(&tree);
    s.absorb_derived_roots(&[&root_tree]);

    // Segment 1: absorb v1, sample p1.
    let v1 = sum_ef(t1) + alpha;
    s.absorb(v1);
    let p1 = s.sample_vec(log_n);
    // Segment 2: absorb v2, sample p2.
    let v2 = sum_ef(t2) + beta;
    s.absorb(v2);
    let p2 = s.sample_vec(log_n);

    let open_1 = whir.open_ef(&root_1, pd_1, &proto_1, &to_p3_point(&p1));
    let open_2 = whir.open_ef(&root_2, pd_2, &proto_2, &to_p3_point(&p2));
    let open_tree = whir.open_ef(&root_tree, pd_tree, &proto_tree, &to_p3_point(&p1));

    Some((
        ScheduleStatement { log_n, root_1, root_2, root_tree },
        ScheduleProof {
            v1,
            v2,
            p1,
            p2,
            open_1,
            open_2,
            open_tree,
        },
    ))
}

/// Verify the schedule contract: reconstruct the identical challenge stream
/// from the statement and the proof's round messages, and check every WHIR
/// opening at the reconstructed point against the claimed value. Statement
/// and proof only — no witness, no recomputation, no `open` calls.
pub fn verify(whir: &Whir, stmt: &ScheduleStatement, proof: &ScheduleProof) -> bool {
    if whir.num_variables() != stmt.log_n {
        return false;
    }
    let mut s = TwoPhaseTranscript::new(PROTOCOL, &[stmt.log_n], &[&stmt.root_1, &stmt.root_2]);
    let _ = s.sample_derived_challenges();
    s.absorb_derived_roots(&[&stmt.root_tree]);

    s.absorb(proof.v1);
    let p1 = s.sample_vec(stmt.log_n);
    s.absorb(proof.v2);
    let p2 = s.sample_vec(stmt.log_n);
    // Direct point-vector equality with the prover's claimed points — the
    // schedule contract is enforced explicitly, not only via the openings.
    if proof.p1 != p1 || proof.p2 != p2 {
        return false;
    }

    let proto = whir.opening_protocol(stmt.log_n, 1);
    if whir
        .verify_ef(&stmt.root_1, &proof.open_1.0, &proto, &to_p3_point(&p1))
        .ok()
        != Some(proof.open_1.1)
    {
        return false;
    }
    if whir
        .verify_ef(&stmt.root_2, &proof.open_2.0, &proto, &to_p3_point(&p2))
        .ok()
        != Some(proof.open_2.1)
    {
        return false;
    }
    whir
        .verify_ef(&stmt.root_tree, &proof.open_tree.0, &proto, &to_p3_point(&p1))
        .ok()
        == Some(proof.open_tree.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::XorShift64;

    fn fixture(log_n: usize) -> (Whir, ScheduleStatement, ScheduleProof, Vec<Vec<Goldilocks>>) {
        let n = 1usize << log_n;
        let mut rng = XorShift64::new(0x5C1);
        let t1: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let t2: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let whir = Whir::new_target(log_n, 90, 0).expect("valid arity");
        let (stmt, proof) = prove(&whir, &t1, &t2, log_n).expect("honest prove");
        (whir, stmt, proof, vec![t1, t2])
    }

    #[test]
    fn honest_roundtrip() {
        let (whir, stmt, proof, _) = fixture(5);
        assert!(verify(&whir, &stmt, &proof));
    }

    /// The schedule is deterministic: an independent re-prove yields the
    /// identical claimed values (and therefore identical reconstructed
    /// points on both sides).
    #[test]
    fn schedule_is_deterministic() {
        let (whir, stmt, proof, tensors) = fixture(5);
        assert!(verify(&whir, &stmt, &proof));
        let (stmt2, proof2) = prove(&whir, &tensors[0], &tensors[1], 5).unwrap();
        assert_eq!(proof.v1, proof2.v1);
        assert_eq!(proof.v2, proof2.v2);
        assert_eq!(proof.open_1.1, proof2.open_1.1);
        assert_eq!(proof.open_tree.1, proof2.open_tree.1);
        assert!(verify(&whir, &stmt2, &proof2));
    }

    #[test]
    fn tampered_initial_root_rejected() {
        let (whir, stmt, proof, _) = fixture(5);
        let mut rng = XorShift64::new(0x5C2);
        let n = 1 << 5;
        let fake: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let (fake_root, _, _) = whir.commit(&fake);
        let mut bad = stmt.clone();
        bad.root_1 = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn tampered_derived_root_rejected() {
        let (whir, stmt, proof, _) = fixture(5);
        let mut rng = XorShift64::new(0x5C3);
        let n = 1 << 5;
        let fake: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let (fake_root, _, _) = whir.commit(&fake);
        let mut bad = stmt.clone();
        bad.root_tree = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn tampered_round_message_rejected() {
        let (whir, stmt, mut proof, _) = fixture(5);
        proof.v1 = proof.v1 + EF::ONE;
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn verify_never_opens() {
        let (whir, stmt, proof, _) = fixture(5);
        let before = whir.open_stats();
        assert!(verify(&whir, &stmt, &proof));
        assert_eq!(whir.open_stats(), before, "verifier must never open");
    }

    #[test]
    fn malformed_dims_rejected() {
        let (whir, stmt, proof, tensors) = fixture(5);
        assert!(verify(&whir, &stmt, &proof));
        let mut bad = stmt.clone();
        bad.log_n = 6;
        assert!(!verify(&whir, &bad, &proof));
        assert!(prove(&whir, &tensors[0][..16], &tensors[1], 5).is_none());
    }

    /// Huge `log_n` (at or beyond `usize::BITS`) must yield `None` from
    /// `prove` — never an overflowing shift.
    #[test]
    fn huge_log_n_rejected() {
        let (whir, _stmt, _proof, tensors) = fixture(5);
        assert!(prove(&whir, &tensors[0], &tensors[1], usize::BITS as usize).is_none());
        assert!(prove(&whir, &tensors[0], &tensors[1], 100).is_none());
    }

    /// Independent non-constant reference: the proof's point vectors must
    /// equal the test-side schedule reconstruction exactly, and every opening
    /// value must equal the independent `mle::eval_ef` of the corresponding
    /// tensor at that vector — point identity is asserted directly, not
    /// inferred from the openings alone.
    #[test]
    fn points_match_independent_schedule_and_reference_evals() {
        let (whir, stmt, proof, tensors) = fixture(5);
        assert!(verify(&whir, &stmt, &proof));
        let (t1, t2) = (&tensors[0], &tensors[1]);
        let n = 1 << 5;

        // Test-side schedule reconstruction (fresh commitments of the same
        // witnesses give the same roots — determinism).
        let (r1, _, _) = whir.commit(t1);
        let (r2, _, _) = whir.commit(t2);
        assert_eq!(r1, stmt.root_1);
        assert_eq!(r2, stmt.root_2);
        let mut s = TwoPhaseTranscript::new(PROTOCOL, &[5], &[&r1, &r2]);
        let (alpha, beta) = s.sample_derived_challenges();
        let ca: &[Goldilocks] = alpha.as_basis_coefficients_slice();
        let cb: &[Goldilocks] = beta.as_basis_coefficients_slice();
        let scalar = ca[0] + ca[1] + cb[0] + cb[1];
        let tree: Vec<Goldilocks> = (0..n).map(|i| t1[i] + t2[i] + scalar).collect();
        let (rt, _, _) = whir.commit(&tree);
        assert_eq!(rt, stmt.root_tree);
        s.absorb_derived_roots(&[&rt]);

        let v1e = sum_ef(t1) + alpha;
        s.absorb(v1e);
        let p1e = s.sample_vec(5);
        let v2e = sum_ef(t2) + beta;
        s.absorb(v2e);
        let p2e = s.sample_vec(5);

        assert_eq!(proof.v1, v1e);
        assert_eq!(proof.v2, v2e);
        assert_eq!(proof.p1, p1e, "prover point 1 must equal the reconstructed vector");
        assert_eq!(proof.p2, p2e, "prover point 2 must equal the reconstructed vector");
        assert_eq!(proof.open_1.1, mle::eval_ef(t1, &p1e));
        assert_eq!(proof.open_2.1, mle::eval_ef(t2, &p2e));
        assert_eq!(proof.open_tree.1, mle::eval_ef(&tree, &p1e));
    }
}
