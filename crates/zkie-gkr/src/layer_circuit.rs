//! Layer circuit: fold a set of elementwise constraints into ONE eq-weighted
//! sumcheck. This is the substrate for "one sumcheck per layer": each op's
//! arithmetic constraint is a virtual polynomial that must vanish, and all of
//! them are combined with a random linear combination plus the eq selector.
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::from_i64;
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
pub struct LayerCircuitProof {
    pub challenges: Vec<Goldilocks>,
    pub proof: VirtualProof,
}
pub fn prove_layer_circuit(
    mles: &[&[Goldilocks]],
    constraints: &[Vec<(Goldilocks, Vec<usize>)>],
    r: &[Goldilocks],
    rng: &mut XorShift64,
) -> LayerCircuitProof {
    let eq: Vec<Goldilocks> = crate::mle::eq_evals(r);
    let mut all: Vec<&[Goldilocks]> = vec![&eq];
    all.extend_from_slice(mles);
    let mut challenges = Vec::with_capacity(constraints.len());
    let mut terms: Vec<(Goldilocks, Vec<usize>)> = Vec::new();
    for constraint in constraints {
        let c = rng.field();
        challenges.push(c);
        for (coeff, idxs) in constraint {
            let mut nidxs = vec![0usize];
            nidxs.extend(idxs.iter().map(|&i| i + 1));
            terms.push((c * *coeff, nidxs));
        }
    }
    let proof = prove_virtual(&all, &terms, Goldilocks::from_u64(0), r);
    LayerCircuitProof { challenges, proof }
}
pub fn verify_layer_circuit(
    lcp: &LayerCircuitProof,
    mles: &[&[Goldilocks]],
    constraints: &[Vec<(Goldilocks, Vec<usize>)>],
    r: &[Goldilocks],
) -> bool {
    let eq: Vec<Goldilocks> = crate::mle::eq_evals(r);
    let mut all: Vec<&[Goldilocks]> = vec![&eq];
    all.extend_from_slice(mles);
    let mut terms: Vec<(Goldilocks, Vec<usize>)> = Vec::new();
    for (ci, constraint) in constraints.iter().enumerate() {
        let c = lcp.challenges[ci];
        for (coeff, idxs) in constraint {
            let mut nidxs = vec![0usize];
            nidxs.extend(idxs.iter().map(|&i| i + 1));
            terms.push((c * *coeff, nidxs));
        }
    }
    let final_evals: Vec<Goldilocks> = all.iter().map(|m| crate::mle::eval(m, r)).collect();
    verify_virtual(&lcp.proof, &terms, Goldilocks::from_u64(0), r, &final_evals)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_circuit_add_mul() {
        let mut rng = XorShift64::new(0xCAFE);
        let n = 1usize << 5;
        let x: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let c: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let y: Vec<Goldilocks> = x.iter().zip(&b).map(|(&xv, &bv)| xv + bv).collect();
        let z: Vec<Goldilocks> = y.iter().zip(&c).map(|(&yv, &cv)| yv * cv).collect();
        let t = n.trailing_zeros() as usize;
        let r: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let constraints: Vec<Vec<(Goldilocks, Vec<usize>)>> = vec![
            vec![
                (Goldilocks::from_u64(1), vec![2usize]),
                (from_i64(-1), vec![0usize]),
                (from_i64(-1), vec![1usize]),
            ],
            vec![
                (Goldilocks::from_u64(1), vec![4usize]),
                (from_i64(-1), vec![2usize, 3usize]),
            ],
        ];
        let mles: Vec<&[Goldilocks]> = vec![&x, &b, &y, &c, &z];
        let proof = prove_layer_circuit(&mles, &constraints, &r, &mut rng);
        assert!(verify_layer_circuit(&proof, &mles, &constraints, &r));
        let mut bad_z = z.clone();
        bad_z[0] = bad_z[0] + Goldilocks::from_u64(1);
        let mles_bad: Vec<&[Goldilocks]> = vec![&x, &b, &y, &c, &bad_z];
        assert!(!verify_layer_circuit(&proof, &mles_bad, &constraints, &r));
    }
}
