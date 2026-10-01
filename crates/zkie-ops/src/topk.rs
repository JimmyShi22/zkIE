//! Top-k selection proof: `gate = scores` for the k largest entries of each
//! row and `0` otherwise, plus explicit "selected >= threshold >= unselected"
//! and "exactly k selected" checks.
//!
//! `x` is `[m, n]` Q16 scores. The prover provides the threshold (the k-th
//! largest score per row) and the 0/1 selection indicator. The proof checks
//!   gate = x * sel
//!   sel in {0,1}
//!   d1 = sel*(x - thr) and d2 = (1-sel)*(thr - x), both in [0, 2^16)
//!   sum_e sel[t,e] = k for every row  (exactly k selected)

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i32, to_i64};
use zkie_core::common::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional, FractionalProof};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

pub struct TopKSelectProof {
    pub gate_rel: VirtualProof,
    pub d1_rel: VirtualProof,
    pub d2_rel: VirtualProof,
    pub row_sum: VirtualProof,
    pub sel_range: FractionalProof,
    pub d1_range: FractionalProof,
    pub d2_range: FractionalProof,
    pub r: Vec<Goldilocks>,
    pub r_row: Vec<Goldilocks>,
    pub a_sel: Goldilocks,
    pub b_sel: Goldilocks,
    pub a_d1: Goldilocks,
    pub b_d1: Goldilocks,
    pub a_d2: Goldilocks,
    pub b_d2: Goldilocks,
}

/// `(thr, sel, gate, d1, d2)`.
pub fn topk_forward(
    x: &[Goldilocks],
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>) {
    let mut thr = vec![Goldilocks::ZERO; m];
    let mut sel = vec![Goldilocks::ZERO; m * n];
    let mut gate = vec![Goldilocks::ZERO; m * n];
    let mut d1 = vec![Goldilocks::ZERO; m * n];
    let mut d2 = vec![Goldilocks::ZERO; m * n];
    for t in 0..m {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            to_i64(x[t * n + b])
                .cmp(&to_i64(x[t * n + a]))
                .then(a.cmp(&b))
        });
        thr[t] = x[t * n + idx[k - 1]];
        for &e in &idx[..k] {
            sel[t * n + e] = Goldilocks::ONE;
        }
        for e in 0..n {
            let xe = x[t * n + e];
            let se = sel[t * n + e];
            let th = thr[t];
            gate[t * n + e] = xe * se;
            d1[t * n + e] = se * (xe - th);
            d2[t * n + e] = (Goldilocks::ONE - se) * (th - xe);
        }
    }
    (thr, sel, gate, d1, d2)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_topk(
    x: &[Goldilocks],
    sel: &[Goldilocks],
    thr: &[Goldilocks],
    gate: &[Goldilocks],
    d1: &[Goldilocks],
    d2: &[Goldilocks],
    m: usize,
    n: usize,
    k: usize,
    rng: &mut XorShift64,
) -> TopKSelectProof {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let r: Vec<Goldilocks> = (0..(m * n).trailing_zeros() as usize).map(|_| rng.field()).collect();

    // gate = x * sel  (degree-2)
    let gate_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![2usize]),
    ];
    let gate_rel = prove_virtual(&[x, sel, gate], &gate_terms, Goldilocks::ZERO, &r);

    // threshold broadcast to [m, n]
    let thr_b: Vec<Goldilocks> = thr.iter().flat_map(|&v| std::iter::repeat(v).take(n)).collect();

    // d1 = sel*(x - thr)  =>  sel*x - sel*thr - d1 = 0
    let d1_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (neg, vec![3usize]),
    ];
    let d1_rel = prove_virtual(&[sel, x, &thr_b, d1], &d1_terms, Goldilocks::ZERO, &r);

    // d2 = (1-sel)*(thr - x) => thr - x - sel*thr + sel*x - d2 = 0
    let d2_terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![1usize]),
        (neg, vec![0usize, 2usize]),
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![3usize]),
    ];
    let d2_rel = prove_virtual(&[sel, x, &thr_b, d2], &d2_terms, Goldilocks::ZERO, &r);

    // sum_e sel[t,e] = k for every row: prove row_sum(r_row) = k
    let r_row: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let eq_row = mle::eq_evals(&r_row);
    let eq_b: Vec<Goldilocks> = eq_row.iter().flat_map(|&v| std::iter::repeat(v).take(n)).collect();
    let row_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize])];
    let row_sum = prove_virtual(&[sel, &eq_b], &row_terms, from_i64(k as i64), &r);

    // sel in {0,1}
    let a_sel = rng.field();
    let b_sel = rng.field();
    let sel_idx: Vec<u32> = sel.iter().map(|&s| to_i32(s) as u32).collect();
    let bin_table = vec![Goldilocks::ZERO, Goldilocks::ONE];
    let sel_range = prove_lookup_fractional(&sel_idx, sel, &bin_table, a_sel, b_sel, rng);

    // d1, d2 in [0, 2^16)
    let a_d1 = rng.field();
    let b_d1 = rng.field();
    let d1_idx: Vec<u32> = d1.iter().map(|&v| to_i32(v) as u32).collect();
    let d1_table: Vec<Goldilocks> = (0..(1usize << 16)).map(|j| Goldilocks::from_u64(j as u64)).collect();
    let d1_range = prove_lookup_fractional(&d1_idx, d1, &d1_table, a_d1, b_d1, rng);

    let a_d2 = rng.field();
    let b_d2 = rng.field();
    let d2_idx: Vec<u32> = d2.iter().map(|&v| to_i32(v) as u32).collect();
    let d2_range = prove_lookup_fractional(&d2_idx, d2, &d1_table, a_d2, b_d2, rng);

    TopKSelectProof {
        gate_rel,
        d1_rel,
        d2_rel,
        row_sum,
        sel_range,
        d1_range,
        d2_range,
        r,
        r_row,
        a_sel,
        b_sel,
        a_d1,
        b_d1,
        a_d2,
        b_d2,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_topk(
    proof: &TopKSelectProof,
    x: &[Goldilocks],
    sel: &[Goldilocks],
    thr: &[Goldilocks],
    gate: &[Goldilocks],
    d1: &[Goldilocks],
    d2: &[Goldilocks],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    let gate_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![2usize]),
    ];
    let gate_fe = vec![
        mle::eval(x, &proof.r),
        mle::eval(sel, &proof.r),
        mle::eval(gate, &proof.r),
    ];
    if !verify_virtual(&proof.gate_rel, &gate_terms, Goldilocks::ZERO, &proof.r, &gate_fe) {
        return false;
    }

    let thr_b: Vec<Goldilocks> = thr.iter().flat_map(|&v| std::iter::repeat(v).take(n)).collect();

    let d1_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (neg, vec![3usize]),
    ];
    let d1_fe = vec![
        mle::eval(sel, &proof.r),
        mle::eval(x, &proof.r),
        mle::eval(&thr_b, &proof.r),
        mle::eval(d1, &proof.r),
    ];
    if !verify_virtual(&proof.d1_rel, &d1_terms, Goldilocks::ZERO, &proof.r, &d1_fe) {
        return false;
    }

    let d2_terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![1usize]),
        (neg, vec![0usize, 2usize]),
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![3usize]),
    ];
    let d2_fe = vec![
        mle::eval(sel, &proof.r),
        mle::eval(x, &proof.r),
        mle::eval(&thr_b, &proof.r),
        mle::eval(d2, &proof.r),
    ];
    if !verify_virtual(&proof.d2_rel, &d2_terms, Goldilocks::ZERO, &proof.r, &d2_fe) {
        return false;
    }

    let eq_row = mle::eq_evals(&proof.r_row);
    let eq_b: Vec<Goldilocks> = eq_row.iter().flat_map(|&v| std::iter::repeat(v).take(n)).collect();
    let row_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize])];
    let row_fe = vec![
        mle::eval(sel, &proof.r),
        mle::eval(&eq_b, &proof.r),
    ];
    if !verify_virtual(&proof.row_sum, &row_terms, from_i64(k as i64), &proof.r, &row_fe) {
        return false;
    }

    let sel_idx: Vec<u32> = sel.iter().map(|&s| to_i32(s) as u32).collect();
    let bin_table = vec![Goldilocks::ZERO, Goldilocks::ONE];
    if !verify_lookup_fractional(&proof.sel_range, &sel_idx, sel, &bin_table, proof.a_sel, proof.b_sel) {
        return false;
    }

    let d1_idx: Vec<u32> = d1.iter().map(|&v| to_i32(v) as u32).collect();
    let d1_table: Vec<Goldilocks> = (0..(1usize << 16)).map(|j| Goldilocks::from_u64(j as u64)).collect();
    if !verify_lookup_fractional(&proof.d1_range, &d1_idx, d1, &d1_table, proof.a_d1, proof.b_d1) {
        return false;
    }
    let d2_idx: Vec<u32> = d2.iter().map(|&v| to_i32(v) as u32).collect();
    verify_lookup_fractional(&proof.d2_range, &d2_idx, d2, &d1_table, proof.a_d2, proof.b_d2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_roundtrip() {
        let mut rng = XorShift64::new(0x7777);
        let (m, n, k) = (4usize, 8usize, 3usize);
        let x: Vec<Goldilocks> = (0..m * n).map(|i| from_i64((i as i64 % 53) - 26)).collect();
        let (thr, sel, gate, d1, d2) = topk_forward(&x, m, n, k);
        let proof = prove_topk(&x, &sel, &thr, &gate, &d1, &d2, m, n, k, &mut rng);
        assert!(verify_topk(&proof, &x, &sel, &thr, &gate, &d1, &d2, m, n, k));
        // corrupt the gate -> must fail
        let mut bad = gate.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_topk(&proof, &x, &sel, &thr, &bad, &d1, &d2, m, n, k));
        // corrupt d1 -> threshold relation must fail
        let mut bad_d1 = d1.clone();
        bad_d1[0] = bad_d1[0] + Goldilocks::ONE;
        assert!(!verify_topk(&proof, &x, &sel, &thr, &gate, &bad_d1, &d2, m, n, k));
    }
}
