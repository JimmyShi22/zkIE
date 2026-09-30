//! Softmax core test: e (exp values), sum = row_sum(e) (virtual), out = round(e*2^16/sum),
//! rem = e*2^16 - out*sum. Proves the row-sum reduction and the rescale constraint
//! (product with the virtual sum, using a broadcast eq selector to fold the (i,j)
//! terms into the (i,j,j') domain). The range check |rem| < sum is verified directly
//! here (it is a logUp range check like the affine round check).
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i64};
use crate::sumcheck::{prove_virtual, verify_virtual};
fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn softmax_core() {
        let mut rng = XorShift64::new(0x50F7);
        let (m, n) = (2usize, 4usize);
        let scale = 1i64 << 16;
        let e: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 1000) as i64 + 1)).collect();
        let sum: Vec<Goldilocks> = (0..m).map(|i| (0..n).fold(Goldilocks::from_u64(0), |acc, j| acc + e[i * n + j])).collect();
        let out: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(round_div(to_i64(e[ij]) * scale, to_i64(sum[ij / n])))).collect();
        let rem: Vec<Goldilocks> = (0..m * n).map(|ij| e[ij] * from_i64(scale) - out[ij] * sum[ij / n]).collect();
        for i in 0..m {
            let s = to_i64(sum[i]);
            for j in 0..n {
                let r = to_i64(rem[i * n + j]);
                assert!(r.abs() < s);
            }
        }
        let r_i: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let eq_i = crate::mle::eq_evals(&r_i);
        let eq_b: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
        let sum_claim = crate::mle::eval(&sum, &r_i);
        let sum_mles: Vec<&[Goldilocks]> = vec![&eq_b, &e];
        let sum_terms = vec![(Goldilocks::from_u64(1), vec![0usize, 1usize])];
        let sum_ch: Vec<Goldilocks> = (0..(m * n).trailing_zeros() as usize).map(|_| rng.field()).collect();
        let sum_proof = prove_virtual(&sum_mles, &sum_terms, sum_claim, &sum_ch);
        let sum_fe = vec![crate::mle::eval(&eq_b, &sum_ch), crate::mle::eval(&e, &sum_ch)];
        assert!(verify_virtual(&sum_proof, &sum_terms, sum_claim, &sum_ch, &sum_fe));
        let r_ij: Vec<Goldilocks> = (0..(m * n).trailing_zeros() as usize).map(|_| rng.field()).collect();
        let s: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let eq_ij = crate::mle::eq_evals(&r_ij);
        let eq_s = crate::mle::eq_evals(&s);
        let dom = m * n * n;
        let mut eqij_b = vec![Goldilocks::from_u64(0); dom];
        let mut eqs_b = vec![Goldilocks::from_u64(0); dom];
        let mut rem_b = vec![Goldilocks::from_u64(0); dom];
        let mut eij_b = vec![Goldilocks::from_u64(0); dom];
        let mut out_b = vec![Goldilocks::from_u64(0); dom];
        let mut e_b = vec![Goldilocks::from_u64(0); dom];
        for i in 0..m {
            for j in 0..n {
                for jp in 0..n {
                    let idx = (i * n + j) * n + jp;
                    eqij_b[idx] = eq_ij[i * n + j];
                    eqs_b[idx] = eq_s[jp];
                    rem_b[idx] = rem[i * n + j];
                    eij_b[idx] = e[i * n + j];
                    out_b[idx] = out[i * n + j];
                    e_b[idx] = e[i * n + jp];
                }
            }
        }
        let neg = from_i64(-1);
        let neg_scale = from_i64(-scale);
        let terms = vec![
            (Goldilocks::from_u64(1), vec![0usize, 2usize, 1usize]),
            (neg_scale, vec![0usize, 3usize, 1usize]),
            (Goldilocks::from_u64(1), vec![0usize, 4usize, 5usize]),
        ];
        let mles: Vec<&[Goldilocks]> = vec![&eqij_b, &eqs_b, &rem_b, &eij_b, &out_b, &e_b];
        let ch: Vec<Goldilocks> = (0..dom.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let proof = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &ch);
        let fe: Vec<Goldilocks> = mles.iter().map(|mm| crate::mle::eval(mm, &ch)).collect();
        assert!(verify_virtual(&proof, &terms, Goldilocks::from_u64(0), &ch, &fe));
    }
    #[test]
    fn committed_softmax_broadcast_open() {
        use crate::whir::Whir;
        use crate::committed::commit;
        let mut rng = XorShift64::new(0x55AA);
        let (m, n) = (8usize, 8usize);
        let lg = m.trailing_zeros() as usize;
        let scores: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let probs: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let rem: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let whir = Whir::new_testing((m * n).trailing_zeros() as usize);
        let c_scores = commit(&whir, &scores);
        let c_probs = commit(&whir, &probs);
        let c_rem = commit(&whir, &rem);
        let ch: Vec<Goldilocks> = (0..(m * n * n).trailing_zeros() as usize).map(|_| rng.field()).collect();
        let p_ji: Vec<Goldilocks> = ch[lg..3 * lg].to_vec();
        let mut p_ji_p = ch[0..lg].to_vec();
        p_ji_p.extend_from_slice(&ch[2 * lg..3 * lg]);
        let (o1, s1) = whir.open(c_scores.prover_data.clone(), &c_scores.protocol, &p_ji);
        assert_eq!(whir.verify(&c_scores.commitment, &o1, &c_scores.protocol, &p_ji).unwrap(), s1);
        assert_eq!(s1, crate::mle::eval(&scores, &p_ji));
        let (o2, s2) = whir.open(c_scores.prover_data.clone(), &c_scores.protocol, &p_ji_p);
        assert_eq!(whir.verify(&c_scores.commitment, &o2, &c_scores.protocol, &p_ji_p).unwrap(), s2);
        assert_eq!(s2, crate::mle::eval(&scores, &p_ji_p));
        let (o3, p1) = whir.open(c_probs.prover_data.clone(), &c_probs.protocol, &p_ji);
        assert_eq!(whir.verify(&c_probs.commitment, &o3, &c_probs.protocol, &p_ji).unwrap(), p1);
        assert_eq!(p1, crate::mle::eval(&probs, &p_ji));
        let (o4, r1) = whir.open(c_rem.prover_data.clone(), &c_rem.protocol, &p_ji);
        assert_eq!(whir.verify(&c_rem.commitment, &o4, &c_rem.protocol, &p_ji).unwrap(), r1);
        assert_eq!(r1, crate::mle::eval(&rem, &p_ji));
        assert!(s1 != s2);
    }
}
