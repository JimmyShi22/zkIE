//! same_poly: prove a set of claims (r_i, y_i) are all evaluations of the SAME
//! MLE f (f(r_i)=y_i), and merge them into one claim at a fresh point. This is
//! the claim-merging primitive for reduction chaining (DeepProve iop/same_poly).
//!
//! S1: the batch coefficients are squeezed from the transcript (binding the
//! statement and prior messages), and the merged point is derived per-round
//! from the virtual sumcheck's own round messages.

use crate::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use crate::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use crate::common::transcript::Transcript;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SamePolyProof {
    pub coeffs: Vec<Goldilocks>,
    pub proof: VirtualProof,
    pub merged_point: Vec<Goldilocks>,
    pub merged_eval: Goldilocks,
}

pub fn prove_same_poly(
    f: &[Goldilocks],
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    transcript: &mut Transcript,
) -> SamePolyProof {
    let t = f.len().trailing_zeros() as usize;
    for (r_i, _) in claims {
        assert_eq!(r_i.len(), t);
    }
    let coeffs: Vec<Goldilocks> = (0..claims.len()).map(|_| transcript.field()).collect();
    let eqs: Vec<Vec<Goldilocks>> = claims
        .iter()
        .map(|(r_i, _)| crate::common::mle::eq_evals(r_i))
        .collect();
    let mut terms: Vec<(Goldilocks, Vec<usize>)> = Vec::new();
    let mut claimed = Goldilocks::from_u64(0);
    for (i, (_, y_i)) in claims.iter().enumerate() {
        terms.push((coeffs[i], vec![0usize, i + 1]));
        claimed = claimed + coeffs[i] * *y_i;
    }
    let mut mles: Vec<&[Goldilocks]> = vec![f];
    for e in &eqs {
        mles.push(e.as_slice());
    }
    let (proof, merged_point) = prove_virtual(&mles, &terms, claimed, transcript);
    let merged_eval = proof.final_evals[0];
    SamePolyProof {
        coeffs,
        proof,
        merged_point,
        merged_eval,
    }
}

pub fn verify_same_poly(
    proof: &SamePolyProof,
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    transcript: &mut Transcript,
) -> Option<Goldilocks> {
    if claims.is_empty() {
        return None;
    }
    let t = claims[0].0.len();
    let coeffs: Vec<Goldilocks> = (0..claims.len()).map(|_| transcript.field()).collect();
    if coeffs != proof.coeffs {
        return None;
    }
    let mut terms: Vec<(Goldilocks, Vec<usize>)> = Vec::new();
    let mut claimed = Goldilocks::from_u64(0);
    for (i, (_, y_i)) in claims.iter().enumerate() {
        terms.push((proof.coeffs[i], vec![0usize, i + 1]));
        claimed = claimed + proof.coeffs[i] * *y_i;
    }
    let (ok, r, final_claim) = verify_virtual(&proof.proof, &terms, claimed, t, transcript);
    if !ok || r != proof.merged_point {
        return None;
    }
    if final_claim != virtual_eval(&terms, &proof.proof.final_evals) {
        return None;
    }
    Some(proof.merged_eval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_poly_roundtrip() {
        let mut rng = XorShift64::new(0x5A1E);
        let t = 6usize;
        let n = 1usize << t;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let m = 3usize;
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = (0..m)
            .map(|_| {
                let r_i: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
                let y_i = crate::common::mle::eval(&f, &r_i);
                (r_i, y_i)
            })
            .collect();
        let mut tp = Transcript::new(b"zkie-same-poly-test");
        tp.absorb_u64(0x5A1E);
        let proof = prove_same_poly(&f, &claims, &mut tp);
        let mut tv = Transcript::new(b"zkie-same-poly-test");
        tv.absorb_u64(0x5A1E);
        let merged = verify_same_poly(&proof, &claims, &mut tv);
        assert!(merged.is_some());
        assert_eq!(
            merged.unwrap(),
            crate::common::mle::eval(&f, &proof.merged_point)
        );
        let mut bad = claims.clone();
        bad[0].1 = bad[0].1 + Goldilocks::from_u64(1);
        let mut tv2 = Transcript::new(b"zkie-same-poly-test");
        tv2.absorb_u64(0x5A1E);
        assert!(verify_same_poly(&proof, &bad, &mut tv2).is_none());
    }
}
