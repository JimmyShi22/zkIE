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

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::{from_i64, to_i32, to_i64};
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional, verify_lookup_fractional, verify_lookup_fractional_chain,
    FractionalProof,
};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct TopKSelectProof {
    pub gate_rel: VirtualProof,
    pub d1_rel: VirtualProof,
    pub d2_rel: VirtualProof,
    pub row_sum: VirtualProof,
    pub sel_range: FractionalProof,
    pub d1_range: FractionalProof,
    pub d2_decomp: RangeDecompProof,
    pub r: Vec<Goldilocks>,
    pub r_row: Vec<Goldilocks>,
    pub a_sel: Goldilocks,
    pub b_sel: Goldilocks,
    pub a_d1: Goldilocks,
    pub b_d1: Goldilocks,
    pub range_bits: usize,
    pub sel: Vec<Goldilocks>,
}

/// `(thr, sel, gate, d1, d2)`.
pub fn topk_forward(
    x: &[Goldilocks],
    m: usize,
    n: usize,
    k: usize,
) -> (
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
) {
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
            d1[t * n + e] = se * from_i64(to_i64(xe) - to_i64(th));
            d2[t * n + e] = (Goldilocks::ONE - se) * from_i64(to_i64(th) - to_i64(xe));
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
    range_bits: usize,
    rng: &mut impl ChallengeSource,
) -> TopKSelectProof {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let r: Vec<Goldilocks> = (0..(m * n).trailing_zeros() as usize)
        .map(|_| rng.field())
        .collect();

    // gate = x * sel  (degree-2)
    let gate_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize]), (neg, vec![2usize])];
    let gate_rel = prove_virtual(&[x, sel, gate], &gate_terms, Goldilocks::ZERO, &r);

    // threshold broadcast to [m, n]
    let thr_b: Vec<Goldilocks> = thr
        .iter()
        .flat_map(|&v| std::iter::repeat(v).take(n))
        .collect();

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
    let r_row: Vec<Goldilocks> = (0..m.trailing_zeros() as usize)
        .map(|_| rng.field())
        .collect();
    let eq_row = mle::eq_evals(&r_row);
    let eq_b: Vec<Goldilocks> = eq_row
        .iter()
        .flat_map(|&v| std::iter::repeat(v).take(n))
        .collect();
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
    let d1_table: Vec<Goldilocks> = (0..(1usize << range_bits))
        .map(|j| Goldilocks::from_u64(j as u64))
        .collect();
    let d1_range = prove_lookup_fractional(&d1_idx, d1, &d1_table, a_d1, b_d1, rng);

    let d2_decomp = prove_range_decomp(d2, rng);

    TopKSelectProof {
        gate_rel,
        d1_rel,
        d2_rel,
        row_sum,
        sel_range,
        d1_range,
        d2_decomp,
        r,
        r_row,
        a_sel,
        b_sel,
        a_d1,
        b_d1,
        range_bits,
            sel: sel.to_vec(),
}
}

#[allow(clippy::too_many_arguments)]
pub fn verify_topk(proof: &TopKSelectProof, k: usize) -> bool {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    if !verify_virtual(
        &proof.gate_rel,
        &[(Goldilocks::ONE, vec![0usize, 1usize]), (neg, vec![2usize])],
        Goldilocks::ZERO,
        &proof.r,
        &proof.gate_rel.final_evals,
    ) {
        return false;
    }

    if !verify_virtual(
        &proof.d1_rel,
        &[
            (Goldilocks::ONE, vec![0usize, 1usize]),
            (neg, vec![0usize, 2usize]),
            (neg, vec![3usize]),
        ],
        Goldilocks::ZERO,
        &proof.r,
        &proof.d1_rel.final_evals,
    ) {
        return false;
    }

    if !verify_virtual(
        &proof.d2_rel,
        &[
            (Goldilocks::ONE, vec![2usize]),
            (neg, vec![1usize]),
            (neg, vec![0usize, 2usize]),
            (Goldilocks::ONE, vec![0usize, 1usize]),
            (neg, vec![3usize]),
        ],
        Goldilocks::ZERO,
        &proof.r,
        &proof.d2_rel.final_evals,
    ) {
        return false;
    }

    if !verify_virtual(
        &proof.row_sum,
        &[(Goldilocks::ONE, vec![0usize, 1usize])],
        from_i64(k as i64),
        &proof.r,
        &proof.row_sum.final_evals,
    ) {
        return false;
    }

    if !verify_lookup_fractional_chain(&proof.sel_range) {
        return false;
    }

    if !verify_lookup_fractional_chain(&proof.d1_range) {
        return false;
    }

    verify_range_decomp(&proof.d2_decomp)
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
        let mut proof = prove_topk(&x, &sel, &thr, &gate, &d1, &d2, m, n, k, 16, &mut rng);
        assert!(verify_topk(&proof, k));
        // corrupt sel_range leaf -> chained verify must fail
        proof.sel_range.layers[0].proof.final_evals[3] += Goldilocks::ONE;
        assert!(!verify_topk(&proof, k));
    }
}


/// Argmax (top-1) per row of an m x n matrix. Reuses TopKSelect's range/one-hot
/// proof with k=1, and binds the public index vector `out` to the one-hot `sel`
/// (checking sel[i][out[i]] == 1 for every row).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ArgmaxProof {
    pub topk: TopKSelectProof,
    pub out: Vec<u32>,
    pub sel_evals: Vec<Goldilocks>,
}

/// `(thr, sel, gate, d1, d2, out)`.
pub fn argmax_forward(
    x: &[Goldilocks],
    m: usize,
    n: usize,
) -> (
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<u32>,
) {
    let (thr, sel, gate, d1, d2) = topk_forward(x, m, n, 1);
    let mut out = vec![0u32; m];
    for i in 0..m {
        let mut best = 0usize;
        for j in 1..n {
            if to_i64(x[i * n + j]) > to_i64(x[i * n + best]) {
                best = j;
            }
        }
        out[i] = best as u32;
    }
    (thr, sel, gate, d1, d2, out)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_argmax(
    x: &[Goldilocks],
    sel: &[Goldilocks],
    thr: &[Goldilocks],
    gate: &[Goldilocks],
    d1: &[Goldilocks],
    d2: &[Goldilocks],
    out: &[u32],
    m: usize,
    n: usize,
    range_bits: usize,
    rng: &mut impl ChallengeSource,
) -> ArgmaxProof {
    let topk = prove_topk(x, sel, thr, gate, d1, d2, m, n, 1, range_bits, rng);
    ArgmaxProof {
        topk,
        out: out.to_vec(),
        sel_evals: Vec::new(),
    }
}

pub fn verify_argmax(proof: &ArgmaxProof, m: usize, n: usize) -> bool {
    if !verify_topk(&proof.topk, 1) {
        return false;
    }
    for i in 0..m {
        let j = proof.out[i] as usize;
        if j >= n {
            return false;
        }
        let v = if proof.sel_evals.is_empty() {
            proof.topk.sel[i * n + j]
        } else {
            proof.sel_evals[i]
        };
        if v != Goldilocks::ONE {
            return false;
        }
    }
    true
}


#[cfg(test)]
mod argmax_tests {
    use super::*;

    #[test]
    fn argmax_roundtrip() {
        let mut rng = XorShift64::new(0x9999);
        let (m, n) = (4usize, 8usize);
        let x: Vec<Goldilocks> = (0..m * n).map(|i| from_i64((i as i64 % 53) - 26)).collect();
        let (thr, sel, gate, d1, d2, out) = argmax_forward(&x, m, n);
        for i in 0..m {
            let mut best = 0usize;
            for j in 1..n {
                if to_i64(x[i * n + j]) > to_i64(x[i * n + best]) {
                    best = j;
                }
            }
            assert_eq!(out[i] as usize, best);
            assert_eq!(sel[i * n + best], Goldilocks::ONE);
        }
        let proof = prove_argmax(&x, &sel, &thr, &gate, &d1, &d2, &out, m, n, 16, &mut rng);
        assert!(verify_argmax(&proof, m, n));
        // corrupt out -> index binding must fail
        let mut bad = proof;
        bad.out[0] = (bad.out[0] + 1) % n as u32;
        assert!(!verify_argmax(&bad, m, n));
    }
}


/// 48-bit range decomposition: prove each value is in [0, 2^48) by splitting it
/// into three 16-bit limbs, range-checking each limb against a 2^16 table, and
/// recomposing. Avoids a single huge lookup table for wide (e.g. raw logit)
/// differences, so the argmax can run on un-shifted logits (no near-tie flip).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RangeDecompProof {
    pub c0_range: FractionalProof,
    pub c1_range: FractionalProof,
    pub c2_range: FractionalProof,
    pub recompose: VirtualProof,
    pub a0: Goldilocks,
    pub b0: Goldilocks,
    pub a1: Goldilocks,
    pub b1: Goldilocks,
    pub a2: Goldilocks,
    pub b2: Goldilocks,
    pub r: Vec<Goldilocks>,
}

pub fn prove_range_decomp(
    values: &[Goldilocks],
    rng: &mut impl ChallengeSource,
) -> RangeDecompProof {
    let c0: Vec<Goldilocks> = values
        .iter()
        .map(|&v| Goldilocks::from_u64((to_i64(v) as u64) & 0xFFFF))
        .collect();
    let c1: Vec<Goldilocks> = values
        .iter()
        .map(|&v| Goldilocks::from_u64(((to_i64(v) as u64) >> 16) & 0xFFFF))
        .collect();
    let c2: Vec<Goldilocks> = values
        .iter()
        .map(|&v| Goldilocks::from_u64(((to_i64(v) as u64) >> 32) & 0xFFFF))
        .collect();
    let table: Vec<Goldilocks> = (0..(1usize << 16))
        .map(|j| Goldilocks::from_u64(j as u64))
        .collect();
    let a0 = rng.field();
    let b0 = rng.field();
    let a1 = rng.field();
    let b1 = rng.field();
    let a2 = rng.field();
    let b2 = rng.field();
    let c0_idx: Vec<u32> = c0.iter().map(|&v| to_i32(v) as u32).collect();
    let c1_idx: Vec<u32> = c1.iter().map(|&v| to_i32(v) as u32).collect();
    let c2_idx: Vec<u32> = c2.iter().map(|&v| to_i32(v) as u32).collect();
    let c0_range = prove_lookup_fractional(&c0_idx, &c0, &table, a0, b0, rng);
    let c1_range = prove_lookup_fractional(&c1_idx, &c1, &table, a1, b1, rng);
    let c2_range = prove_lookup_fractional(&c2_idx, &c2, &table, a2, b2, rng);
    let r: Vec<Goldilocks> = (0..values.len().trailing_zeros() as usize)
        .map(|_| rng.field())
        .collect();
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let s16 = Goldilocks::from_u64(1 << 16);
    let s32 = Goldilocks::from_u64(1 << 32);
    let terms = vec![
        (Goldilocks::ONE, vec![0usize]),
        (neg, vec![1usize]),
        (neg * s16, vec![2usize]),
        (neg * s32, vec![3usize]),
    ];
    let recompose = prove_virtual(&[values, &c0, &c1, &c2], &terms, Goldilocks::ZERO, &r);
    RangeDecompProof {
        c0_range,
        c1_range,
        c2_range,
        recompose,
        a0,
        b0,
        a1,
        b1,
        a2,
        b2,
        r,
    }
}

pub fn verify_range_decomp(proof: &RangeDecompProof) -> bool {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let s16 = Goldilocks::from_u64(1 << 16);
    let s32 = Goldilocks::from_u64(1 << 32);
    let terms = vec![
        (Goldilocks::ONE, vec![0usize]),
        (neg, vec![1usize]),
        (neg * s16, vec![2usize]),
        (neg * s32, vec![3usize]),
    ];
    verify_lookup_fractional_chain(&proof.c0_range)
        && verify_lookup_fractional_chain(&proof.c1_range)
        && verify_lookup_fractional_chain(&proof.c2_range)
        && verify_virtual(
            &proof.recompose,
            &terms,
            Goldilocks::ZERO,
            &proof.r,
            &proof.recompose.final_evals,
        )
}
