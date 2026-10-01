//! same_poly: prove a set of claims (r_i, y_i) are all evaluations of the SAME
//! MLE f (f(r_i)=y_i), and merge them into one claim at a fresh point. This is
//! the claim-merging primitive for reduction chaining (DeepProve iop/same_poly).
use crate::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
pub struct SamePolyProof {
    pub coeffs: Vec<Goldilocks>,
    pub proof: VirtualProof,
    pub merged_point: Vec<Goldilocks>,
    pub merged_eval: Goldilocks,
}
pub fn prove_same_poly(
    f: &[Goldilocks],
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    rng: &mut XorShift64,
) -> SamePolyProof {
    let t = f.len().trailing_zeros() as usize;
    for (r_i, _) in claims {
        assert_eq!(r_i.len(), t);
    }
    let coeffs: Vec<Goldilocks> = (0..claims.len()).map(|_| rng.field()).collect();
    let eqs: Vec<Vec<Goldilocks>> = claims.iter().map(|(r_i, _)| crate::common::mle::eq_evals(r_i)).collect();
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
    let merged_point: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
    let proof = prove_virtual(&mles, &terms, claimed, &merged_point);
    let merged_eval = proof.final_evals[0];
    SamePolyProof { coeffs, proof, merged_point, merged_eval }
}
pub fn verify_same_poly(
    proof: &SamePolyProof,
    f: &[Goldilocks],
    claims: &[(Vec<Goldilocks>, Goldilocks)],
) -> Option<Goldilocks> {
    let eqs: Vec<Vec<Goldilocks>> = claims.iter().map(|(r_i, _)| crate::common::mle::eq_evals(r_i)).collect();
    let mut terms: Vec<(Goldilocks, Vec<usize>)> = Vec::new();
    let mut claimed = Goldilocks::from_u64(0);
    for (i, (_, y_i)) in claims.iter().enumerate() {
        terms.push((proof.coeffs[i], vec![0usize, i + 1]));
        claimed = claimed + proof.coeffs[i] * *y_i;
    }
    let mut mles: Vec<&[Goldilocks]> = vec![f];
    for e in &eqs {
        mles.push(e.as_slice());
    }
    let final_evals: Vec<Goldilocks> = mles.iter().map(|m| crate::common::mle::eval(m, &proof.merged_point)).collect();
    if verify_virtual(&proof.proof, &terms, claimed, &proof.merged_point, &final_evals) {
        Some(final_evals[0])
    } else {
        None
    }
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
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = (0..m).map(|_| {
            let r_i: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
            let y_i = crate::common::mle::eval(&f, &r_i);
            (r_i, y_i)
        }).collect();
        let proof = prove_same_poly(&f, &claims, &mut rng);
        let merged = verify_same_poly(&proof, &f, &claims);
        assert!(merged.is_some());
        assert_eq!(merged.unwrap(), crate::common::mle::eval(&f, &proof.merged_point));
        let mut bad = claims.clone();
        bad[0].1 = bad[0].1 + Goldilocks::from_u64(1);
        assert!(verify_same_poly(&proof, &f, &bad).is_none());
    }
}
