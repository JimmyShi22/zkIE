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
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use zkie_core::common::transcript::Transcript;

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
    transcript: &mut Transcript,
) -> TopKSelectProof {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    // gate = x * sel  (degree-2). Each relation has its own per-round-derived
    // challenge vector (S1); `r` keeps gate_rel's vector for the op claims.
    let gate_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize]), (neg, vec![2usize])];
    let (gate_rel, r) = prove_virtual(&[x, sel, gate], &gate_terms, Goldilocks::ZERO, transcript);

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
    let (d1_rel, _) = prove_virtual(&[sel, x, &thr_b, d1], &d1_terms, Goldilocks::ZERO, transcript);

    // d2 = (1-sel)*(thr - x) => thr - x - sel*thr + sel*x - d2 = 0
    let d2_terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![1usize]),
        (neg, vec![0usize, 2usize]),
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![3usize]),
    ];
    let (d2_rel, _) = prove_virtual(&[sel, x, &thr_b, d2], &d2_terms, Goldilocks::ZERO, transcript);

    // sum_e sel[t,e] = k for every row: prove row_sum(r_row) = k.
    // r_row is the eq point (squeezed upfront); the sumcheck challenges are
    // derived per-round inside.
    let r_row: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    let eq_row = mle::eq_evals(&r_row);
    let eq_b: Vec<Goldilocks> = eq_row
        .iter()
        .flat_map(|&v| std::iter::repeat(v).take(n))
        .collect();
    let row_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize])];
    let (row_sum, _) = prove_virtual(&[sel, &eq_b], &row_terms, from_i64(k as i64), transcript);

    // sel in {0,1}
    let a_sel = transcript.field();
    let b_sel = transcript.field();
    let sel_idx: Vec<u32> = sel.iter().map(|&s| to_i32(s) as u32).collect();
    let bin_table = vec![Goldilocks::ZERO, Goldilocks::ONE];
    let sel_range = prove_lookup_fractional(&sel_idx, sel, &bin_table, a_sel, b_sel, transcript);

    // d1, d2 in [0, 2^16)
    let a_d1 = transcript.field();
    let b_d1 = transcript.field();
    let d1_idx: Vec<u32> = d1.iter().map(|&v| to_i32(v) as u32).collect();
    let d1_table: Vec<Goldilocks> = (0..(1usize << range_bits))
        .map(|j| Goldilocks::from_u64(j as u64))
        .collect();
    let d1_range = prove_lookup_fractional(&d1_idx, d1, &d1_table, a_d1, b_d1, transcript);

    let d2_decomp = prove_range_decomp(d2, transcript);

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
pub fn verify_topk(
    proof: &TopKSelectProof,
    k: usize,
    m: usize,
    n: usize,
    transcript: &mut Transcript,
) -> bool {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let arity_mn = (m * n).trailing_zeros() as usize;

    let gate_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize]), (neg, vec![2usize])];
    let (ok, r, fc) = verify_virtual(
        &proof.gate_rel,
        &gate_terms,
        Goldilocks::ZERO,
        arity_mn,
        transcript,
    );
    if !ok || r != proof.r || fc != virtual_eval(&gate_terms, &proof.gate_rel.final_evals) {
        return false;
    }

    let d1_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (neg, vec![3usize]),
    ];
    let (ok, _, fc) = verify_virtual(
        &proof.d1_rel,
        &d1_terms,
        Goldilocks::ZERO,
        arity_mn,
        transcript,
    );
    if !ok || fc != virtual_eval(&d1_terms, &proof.d1_rel.final_evals) {
        return false;
    }

    let d2_terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![1usize]),
        (neg, vec![0usize, 2usize]),
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![3usize]),
    ];
    let (ok, _, fc) = verify_virtual(
        &proof.d2_rel,
        &d2_terms,
        Goldilocks::ZERO,
        arity_mn,
        transcript,
    );
    if !ok || fc != virtual_eval(&d2_terms, &proof.d2_rel.final_evals) {
        return false;
    }

    let r_row: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    if r_row != proof.r_row {
        return false;
    }
    let row_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize])];
    let (ok, _, fc) = verify_virtual(
        &proof.row_sum,
        &row_terms,
        from_i64(k as i64),
        arity_mn,
        transcript,
    );
    if !ok || fc != virtual_eval(&row_terms, &proof.row_sum.final_evals) {
        return false;
    }

    if transcript.field() != proof.a_sel || transcript.field() != proof.b_sel {
        return false;
    }
    if !verify_lookup_fractional_chain(&proof.sel_range, transcript) {
        return false;
    }

    if transcript.field() != proof.a_d1 || transcript.field() != proof.b_d1 {
        return false;
    }
    if !verify_lookup_fractional_chain(&proof.d1_range, transcript) {
        return false;
    }

    verify_range_decomp(&proof.d2_decomp, transcript)
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
        let mut tp = Transcript::new(b"zkie-topk-test");
        tp.absorb_u64(0x7777);
        let mut proof = prove_topk(&x, &sel, &thr, &gate, &d1, &d2, m, n, k, 16, &mut tp);
        let mut tv = Transcript::new(b"zkie-topk-test");
        tv.absorb_u64(0x7777);
        assert!(verify_topk(&proof, k, m, n, &mut tv));
        // corrupt sel_range leaf -> chained verify must fail
        proof.sel_range.layers[0].proof.final_evals[3] += Goldilocks::ONE;
        let mut tv2 = Transcript::new(b"zkie-topk-test");
        tv2.absorb_u64(0x7777);
        assert!(!verify_topk(&proof, k, m, n, &mut tv2));
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
    transcript: &mut Transcript,
) -> ArgmaxProof {
    let topk = prove_topk(x, sel, thr, gate, d1, d2, m, n, 1, range_bits, transcript);
    ArgmaxProof {
        topk,
        out: out.to_vec(),
        sel_evals: Vec::new(),
    }
}

pub fn verify_argmax(
    proof: &ArgmaxProof,
    m: usize,
    n: usize,
    transcript: &mut Transcript,
) -> bool {
    if !verify_topk(&proof.topk, 1, m, n, transcript) {
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
        let mut tp = Transcript::new(b"zkie-argmax-test");
        tp.absorb_u64(0x9999);
        let proof = prove_argmax(&x, &sel, &thr, &gate, &d1, &d2, &out, m, n, 16, &mut tp);
        let mut tv = Transcript::new(b"zkie-argmax-test");
        tv.absorb_u64(0x9999);
        assert!(verify_argmax(&proof, m, n, &mut tv));
        // corrupt out -> index binding must fail
        let mut bad = proof;
        bad.out[0] = (bad.out[0] + 1) % n as u32;
        let mut tv2 = Transcript::new(b"zkie-argmax-test");
        tv2.absorb_u64(0x9999);
        assert!(!verify_argmax(&bad, m, n, &mut tv2));
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
    transcript: &mut Transcript,
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
    let a0 = transcript.field();
    let b0 = transcript.field();
    let a1 = transcript.field();
    let b1 = transcript.field();
    let a2 = transcript.field();
    let b2 = transcript.field();
    let c0_idx: Vec<u32> = c0.iter().map(|&v| to_i32(v) as u32).collect();
    let c1_idx: Vec<u32> = c1.iter().map(|&v| to_i32(v) as u32).collect();
    let c2_idx: Vec<u32> = c2.iter().map(|&v| to_i32(v) as u32).collect();
    let c0_range = prove_lookup_fractional(&c0_idx, &c0, &table, a0, b0, transcript);
    let c1_range = prove_lookup_fractional(&c1_idx, &c1, &table, a1, b1, transcript);
    let c2_range = prove_lookup_fractional(&c2_idx, &c2, &table, a2, b2, transcript);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let s16 = Goldilocks::from_u64(1 << 16);
    let s32 = Goldilocks::from_u64(1 << 32);
    let terms = vec![
        (Goldilocks::ONE, vec![0usize]),
        (neg, vec![1usize]),
        (neg * s16, vec![2usize]),
        (neg * s32, vec![3usize]),
    ];
    let (recompose, r) = prove_virtual(&[values, &c0, &c1, &c2], &terms, Goldilocks::ZERO, transcript);
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

pub fn verify_range_decomp(proof: &RangeDecompProof, transcript: &mut Transcript) -> bool {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let s16 = Goldilocks::from_u64(1 << 16);
    let s32 = Goldilocks::from_u64(1 << 32);
    let terms = vec![
        (Goldilocks::ONE, vec![0usize]),
        (neg, vec![1usize]),
        (neg * s16, vec![2usize]),
        (neg * s32, vec![3usize]),
    ];
    // The prover squeezes all six alpha/beta upfront, then runs the three
    // fractional proofs; replay in the same order.
    if transcript.field() != proof.a0 || transcript.field() != proof.b0
        || transcript.field() != proof.a1 || transcript.field() != proof.b1
        || transcript.field() != proof.a2 || transcript.field() != proof.b2
    {
        return false;
    }
    if !verify_lookup_fractional_chain(&proof.c0_range, transcript) {
        return false;
    }
    if !verify_lookup_fractional_chain(&proof.c1_range, transcript) {
        return false;
    }
    if !verify_lookup_fractional_chain(&proof.c2_range, transcript) {
        return false;
    }
    let (ok, r, fc) = verify_virtual(
        &proof.recompose,
        &terms,
        Goldilocks::ZERO,
        proof.r.len(),
        transcript,
    );
    ok && r == proof.r && fc == virtual_eval(&terms, &proof.recompose.final_evals)
}
