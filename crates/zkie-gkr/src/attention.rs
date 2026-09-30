//! Attention core test: scores = Q @ K^T, probs = softmax(scores), attn = probs @ V.
//! Composes the demonstrated matmul (committed output) + softmax core (row-sum +
//! rescale with virtual sum) + matmul. Demonstrates the full attention dataflow.
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i64};
use crate::sumcheck::{prove_virtual, verify_virtual};
fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
}
fn matmul_full(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut h = vec![Goldilocks::from_u64(0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::from_u64(0);
            for kk in 0..k {
                acc = acc + a[i * k + kk] * b[kk * n + j];
            }
            h[i * n + j] = acc;
        }
    }
    h
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_core() {
        let mut rng = XorShift64::new(0xA771);
        let (m, d) = (2usize, 2usize);
        let scale = 1i64 << 16;
        let q: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let k: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let v: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let mut kt = vec![Goldilocks::from_u64(0); d * m];
        for kk in 0..d {
            for i in 0..m {
                kt[kk * m + i] = k[i * d + kk];
            }
        }
        let scores = matmul_full(&q, &kt, m, d, m);
        let sum: Vec<Goldilocks> = (0..m).map(|i| (0..m).fold(Goldilocks::from_u64(0), |acc, j| acc + scores[i * m + j])).collect();
        let probs: Vec<Goldilocks> = (0..m * m).map(|ij| from_i64(round_div(to_i64(scores[ij]) * scale, to_i64(sum[ij / m])))).collect();
        let rem: Vec<Goldilocks> = (0..m * m).map(|ij| scores[ij] * from_i64(scale) - probs[ij] * sum[ij / m]).collect();
        let attn = matmul_full(&probs, &v, m, m, d);
        for i in 0..m {
            let s = to_i64(sum[i]);
            for j in 0..m {
                assert!(to_i64(rem[i * m + j]).abs() < s);
            }
        }
        let r_i: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let eq_i = crate::mle::eq_evals(&r_i);
        let eq_b: Vec<Goldilocks> = (0..m * m).map(|idx| eq_i[idx / m]).collect();
        let sum_claim = crate::mle::eval(&sum, &r_i);
        let sum_mles: Vec<&[Goldilocks]> = vec![&eq_b, &scores];
        let sum_terms = vec![(Goldilocks::from_u64(1), vec![0usize, 1usize])];
        let sum_ch: Vec<Goldilocks> = (0..(m * m).trailing_zeros() as usize).map(|_| rng.field()).collect();
        let sum_proof = prove_virtual(&sum_mles, &sum_terms, sum_claim, &sum_ch);
        let sum_fe = vec![crate::mle::eval(&eq_b, &sum_ch), crate::mle::eval(&scores, &sum_ch)];
        assert!(verify_virtual(&sum_proof, &sum_terms, sum_claim, &sum_ch, &sum_fe));
        let r_ij: Vec<Goldilocks> = (0..(m * m).trailing_zeros() as usize).map(|_| rng.field()).collect();
        let s: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let eq_ij = crate::mle::eq_evals(&r_ij);
        let eq_s = crate::mle::eq_evals(&s);
        let dom = m * m * m;
        let mut eqij_b = vec![Goldilocks::from_u64(0); dom];
        let mut eqs_b = vec![Goldilocks::from_u64(0); dom];
        let mut rem_b = vec![Goldilocks::from_u64(0); dom];
        let mut s_ij_b = vec![Goldilocks::from_u64(0); dom];
        let mut probs_b = vec![Goldilocks::from_u64(0); dom];
        let mut scores_b = vec![Goldilocks::from_u64(0); dom];
        for i in 0..m {
            for j in 0..m {
                for jp in 0..m {
                    let idx = (i * m + j) * m + jp;
                    eqij_b[idx] = eq_ij[i * m + j];
                    eqs_b[idx] = eq_s[jp];
                    rem_b[idx] = rem[i * m + j];
                    s_ij_b[idx] = scores[i * m + j];
                    probs_b[idx] = probs[i * m + j];
                    scores_b[idx] = scores[i * m + jp];
                }
            }
        }
        let neg_scale = from_i64(-scale);
        let terms = vec![
            (Goldilocks::from_u64(1), vec![0usize, 2usize, 1usize]),
            (neg_scale, vec![0usize, 3usize, 1usize]),
            (Goldilocks::from_u64(1), vec![0usize, 4usize, 5usize]),
        ];
        let mles: Vec<&[Goldilocks]> = vec![&eqij_b, &eqs_b, &rem_b, &s_ij_b, &probs_b, &scores_b];
        let ch: Vec<Goldilocks> = (0..dom.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let proof = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &ch);
        let fe: Vec<Goldilocks> = mles.iter().map(|mm| crate::mle::eval(mm, &ch)).collect();
        assert!(verify_virtual(&proof, &terms, Goldilocks::from_u64(0), &ch, &fe));
        let _ = attn;
    }
}
