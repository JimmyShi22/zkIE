//! logUp GKR: prove sum_i num[i]/den[i] == C via the fraction-addition tree
//! with eq-weighted virtual sumchecks (DeepProve "fractional sumcheck").
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::from_i64;
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
pub struct FractionalLayer {
    pub r: Vec<Goldilocks>,
    pub c: Goldilocks,
    pub proof: VirtualProof,
}
pub struct FractionalProof {
    pub layers: Vec<FractionalLayer>,
    pub final_num: Goldilocks,
    pub final_den: Goldilocks,
}
fn build_terms(c: Goldilocks) -> Vec<(Goldilocks, Vec<usize>)> {
    let neg = from_i64(-1);
    vec![
        (Goldilocks::from_u64(1), vec![0usize, 1]),
        (neg, vec![0, 3, 6]),
        (neg, vec![0, 4, 5]),
        (c, vec![0, 2]),
        (neg * c, vec![0, 5, 6]),
    ]
}
pub fn prove_fractional(num: &[Goldilocks], den: &[Goldilocks], rng: &mut XorShift64) -> FractionalProof {
    assert_eq!(num.len(), den.len());
    assert!(num.len().is_power_of_two());
    let mut cur_num = num.to_vec();
    let mut cur_den = den.to_vec();
    let mut layers = Vec::new();
    while cur_num.len() > 1 {
        let half = cur_num.len() / 2;
        let num_low = cur_num[..half].to_vec();
        let num_high = cur_num[half..].to_vec();
        let den_low = cur_den[..half].to_vec();
        let den_high = cur_den[half..].to_vec();
        let next_num: Vec<Goldilocks> = (0..half).map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s]).collect();
        let next_den: Vec<Goldilocks> = (0..half).map(|s| den_low[s] * den_high[s]).collect();
        let m = half.trailing_zeros() as usize;
        let r: Vec<Goldilocks> = (0..m).map(|_| rng.field()).collect();
        let c = rng.field();
        let eq: Vec<Goldilocks> = crate::mle::eq_evals(&r);
        let mles: Vec<&[Goldilocks]> = vec![&eq, &next_num, &next_den, &num_low, &num_high, &den_low, &den_high];
        let terms = build_terms(c);
        let proof = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &r);
        layers.push(FractionalLayer { r, c, proof });
        cur_num = next_num;
        cur_den = next_den;
    }
    FractionalProof { layers, final_num: cur_num[0], final_den: cur_den[0] }
}
pub fn verify_fractional(proof: &FractionalProof, num: &[Goldilocks], den: &[Goldilocks], claimed: Goldilocks) -> bool {
    let mut cur_num = num.to_vec();
    let mut cur_den = den.to_vec();
    for layer in &proof.layers {
        let half = cur_num.len() / 2;
        let num_low = &cur_num[..half];
        let num_high = &cur_num[half..];
        let den_low = &cur_den[..half];
        let den_high = &cur_den[half..];
        let next_num: Vec<Goldilocks> = (0..half).map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s]).collect();
        let next_den: Vec<Goldilocks> = (0..half).map(|s| den_low[s] * den_high[s]).collect();
        let eq: Vec<Goldilocks> = crate::mle::eq_evals(&layer.r);
        let mles: Vec<&[Goldilocks]> = vec![&eq, &next_num, &next_den, num_low, num_high, den_low, den_high];
        let terms = build_terms(layer.c);
        let final_evals = vec![
            crate::mle::eval(&eq, &layer.r),
            crate::mle::eval(&next_num, &layer.r),
            crate::mle::eval(&next_den, &layer.r),
            crate::mle::eval(num_low, &layer.r),
            crate::mle::eval(num_high, &layer.r),
            crate::mle::eval(den_low, &layer.r),
            crate::mle::eval(den_high, &layer.r),
        ];
        if !verify_virtual(&layer.proof, &terms, Goldilocks::from_u64(0), &layer.r, &final_evals) {
            return false;
        }
        cur_num = next_num;
        cur_den = next_den;
    }
    proof.final_num == claimed * proof.final_den
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_roundtrip() {
        let mut rng = XorShift64::new(0xDEC0);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| acc + num[i] * den[i].inverse());
        let proof = prove_fractional(&num, &den, &mut rng);
        assert!(verify_fractional(&proof, &num, &den, claimed));
        let tampered = FractionalProof { final_num: proof.final_num + Goldilocks::from_u64(1), final_den: proof.final_den, layers: proof.layers };
        assert!(!verify_fractional(&tampered, &num, &den, claimed));
    }
}
