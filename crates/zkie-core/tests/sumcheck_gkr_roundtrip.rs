//! End-to-end: prove a GKR sum-check over Goldilocks with `zkie-gkr`, then
//! verify the resulting transcript non-natively in the BN254 Halo2 circuit.

use halo2_proofs::dev::MockProver;
use zkie_core::field_convert::Fr;
use zkie_core::sumcheck_verify::{RoundClaim, SumcheckCircuit, SumcheckClaim};
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64, XorShift64};
use zkie_gkr::sumcheck;

#[test]
fn gkr_sumcheck_proof_verifies_in_halo2() {
    let mut rng = XorShift64::new(0x1234);
    let t = 6usize; // 2^6 = 64 evaluation points, 6 sum-check rounds
    let f: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
    let h: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
    let challenges: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();

    let claimed: Goldilocks = f
        .iter()
        .zip(&h)
        .fold(Goldilocks::ZERO, |acc, (&a, &b)| acc + a * b);
    let proof = sumcheck::prove(&f, &h, claimed, &challenges);

    // Convert the Goldilocks transcript into canonical u64 values for the
    // non-native Halo2 verifier.
    let rounds: Vec<RoundClaim> = proof
        .rounds
        .iter()
        .zip(&challenges)
        .map(|(rp, &r)| RoundClaim {
            c0: rp.c0.as_canonical_u64(),
            c1: rp.c1.as_canonical_u64(),
            c2: rp.c2.as_canonical_u64(),
            r: r.as_canonical_u64(),
        })
        .collect();
    let claim = SumcheckClaim {
        rounds,
        claimed: claimed.as_canonical_u64(),
        f: proof.f_eval.as_canonical_u64(),
        h: proof.h_eval.as_canonical_u64(),
    };

    let prover = MockProver::run(
        15,
        &SumcheckCircuit {
            claim: claim.clone(),
        },
        vec![vec![Fr::from(claim.claimed)]],
    )
    .unwrap();
    prover.assert_satisfied();
}

#[test]
fn tampered_gkr_proof_is_rejected_in_halo2() {
    let mut rng = XorShift64::new(0x5678);
    let t = 6usize;
    let f: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
    let h: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
    let challenges: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();

    let claimed: Goldilocks = f
        .iter()
        .zip(&h)
        .fold(Goldilocks::ZERO, |acc, (&a, &b)| acc + a * b);
    let proof = sumcheck::prove(&f, &h, claimed, &challenges);

    let mut rounds: Vec<RoundClaim> = proof
        .rounds
        .iter()
        .zip(&challenges)
        .map(|(rp, &r)| RoundClaim {
            c0: rp.c0.as_canonical_u64(),
            c1: rp.c1.as_canonical_u64(),
            c2: rp.c2.as_canonical_u64(),
            r: r.as_canonical_u64(),
        })
        .collect();
    // Corrupt the last round's constant term; the transcript is no longer valid.
    let last = rounds.last_mut().unwrap();
    last.c0 = last.c0.wrapping_add(1);

    let claim = SumcheckClaim {
        rounds,
        claimed: claimed.as_canonical_u64(),
        f: proof.f_eval.as_canonical_u64(),
        h: proof.h_eval.as_canonical_u64(),
    };

    let prover = MockProver::run(
        15,
        &SumcheckCircuit {
            claim: claim.clone(),
        },
        vec![vec![Fr::from(claim.claimed)]],
    )
    .unwrap();
    assert!(prover.verify().is_err());
}
