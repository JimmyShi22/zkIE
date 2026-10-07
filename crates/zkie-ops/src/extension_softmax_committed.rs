//! Root-bound committed softmax (rounded, GPT-2 semantics):
//!   e[ij]    = exp_table[idx[ij]]        (logUp lookup)
//!   sum[i]   = sum_j e[ij]               (reduction, virtual sumcheck)
//!   out[ij] * sum[i] = e[ij] * 2^16 - rem[ij]   (rescale, pointwise virtual w/ eq)
//!
//! Committed tensors: e, out, sum, rem (and idx via the lookup). Shared
//! commitments (same_poly, byte-equal roots): e (lookup.out + row_sum +
//! rescale), sum (row_sum + rescale broadcast), out (rescale). The `rem` is a
//! witness for the rounded rescale; its range check (exact round-half-up) is a
//! follow-up, as in the projection. The verifier holds `(whir, statement,
//! proof)` only: no witness, no Store, no forward recomputation.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
use zkie_core::pcs::whir::{Commitment, Proof as WhirProof, Whir};

use crate::extension_lookup_logup::{
    prove as lookup_prove, verify as lookup_verify, LookupProof, LookupStatement, LookupWhir,
};

pub const PROTOCOL: &str = "zkie/ext-softmax-committed/v1";

pub struct SoftmaxWhir {
    pub e: Whir,
    pub out: Whir,
    pub sum: Whir,
    pub rem: Whir,
    pub lookup: LookupWhir,
}

impl SoftmaxWhir {
    pub fn new(m: usize, n: usize, t: usize, security_level: usize, pow_budget: usize) -> Option<Self> {
        if !m.is_power_of_two() || !n.is_power_of_two() || !t.is_power_of_two() || m < 2 || n < 2 || t < 2 {
            return None;
        }
        let ar_mn = (m * n).trailing_zeros() as usize;
        let ar_m = m.trailing_zeros() as usize;
        if ar_mn < 5 || ar_m < 5 {
            return None;
        }
        let mk = |v: usize| Whir::new_target(v, security_level, pow_budget);
        Some(SoftmaxWhir {
            e: mk(ar_mn)?,
            out: mk(ar_mn)?,
            sum: mk(ar_m)?,
            rem: mk(ar_mn)?,
            lookup: LookupWhir::new(m * n, t, security_level, pow_budget)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SoftmaxStatement {
    pub m: usize,
    pub n: usize,
    pub root_e: Commitment,
    pub root_out: Commitment,
    pub root_sum: Commitment,
    pub root_rem: Commitment,
    pub lookup: LookupStatement,
}

pub struct SoftmaxCommittedProof {
    pub row_sum: VirtualProof,
    pub rescale: VirtualProof,
    pub lookup: LookupProof,
    pub open_e_sum: (WhirProof, Goldilocks),
    pub open_sum: (WhirProof, Goldilocks),
    pub open_e_scale: (WhirProof, Goldilocks),
    pub open_sum_scale: (WhirProof, Goldilocks),
    pub open_out: (WhirProof, Goldilocks),
    pub open_rem: (WhirProof, Goldilocks),
    pub r_sum: Vec<Goldilocks>,
    pub sum_ch: Vec<Goldilocks>,
    pub r_scale: Vec<Goldilocks>,
}

fn broadcast(sum: &[Goldilocks], n: usize) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; sum.len() * n];
    for (i, &v) in sum.iter().enumerate() {
        for j in 0..n {
            out[i * n + j] = v;
        }
    }
    out
}

fn row_index(point: &[Goldilocks], m_bits: usize) -> Vec<Goldilocks> {
    point[point.len() - m_bits..].to_vec()
}

pub fn prove(
    whir: &SoftmaxWhir,
    idx: &[u32],
    e: &[Goldilocks],
    out: &[Goldilocks],
    sum: &[Goldilocks],
    rem: &[Goldilocks],
    exp_table: &[u64],
    m: usize,
    n: usize,
    rng: &mut XorShift64,
) -> Option<(SoftmaxStatement, SoftmaxCommittedProof)> {
    assert_eq!(e.len(), m * n);
    assert_eq!(out.len(), m * n);
    assert_eq!(rem.len(), m * n);
    assert_eq!(sum.len(), m);
    assert_eq!(idx.len(), m * n);

    let (root_e, pd_e, proto_e) = whir.e.commit(e);
    let (root_out, pd_out, proto_out) = whir.out.commit(out);
    let (root_sum, pd_sum, proto_sum) = whir.sum.commit(sum);
    let (root_rem, pd_rem, proto_rem) = whir.rem.commit(rem);

    let m_bits = m.trailing_zeros() as usize;
    let mn_bits = (m * n).trailing_zeros() as usize;
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    // 1. exp lookup: e = table[idx].
    let e_u64: Vec<u64> = e.iter().map(|&v| to_i64(v) as u64).collect();
    let (lookup_stmt, lookup_proof) = lookup_prove(&whir.lookup, idx, &e_u64, exp_table)?;

    // 2. row_sum: sum[i] = sum_j e[ij].
    let r_sum: Vec<Goldilocks> = (0..m_bits).map(|_| rng.field()).collect();
    let eq_i = mle::eq_evals(&r_sum);
    let eq_b: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
    let sum_claim = mle::eval(sum, &r_sum);
    let sum_ch: Vec<Goldilocks> = (0..mn_bits).map(|_| rng.field()).collect();
    let row_sum = prove_virtual(&[&eq_b, e], &[(Goldilocks::ONE, vec![0usize, 1usize])], sum_claim, &sum_ch);
    let open_e_sum = whir.e.open(pd_e.clone(), &proto_e, &sum_ch);
    let open_sum = whir.sum.open(pd_sum.clone(), &proto_sum, &r_sum);

    // 3. rescale: out * sum_broadcast = e * 2^16 - rem, pointwise via eq.
    let sum_broadcast = broadcast(sum, n);
    let r_scale: Vec<Goldilocks> = (0..mn_bits).map(|_| rng.field()).collect();
    let eq_scale = mle::eq_evals(&r_scale);
    let two16 = Goldilocks::from_u64(1u64 << 16);
    let rescale = prove_virtual(
        &[&eq_scale, out, e, &sum_broadcast, rem],
        &[
            (Goldilocks::ONE, vec![0usize, 1usize, 3usize]),
            (neg * two16, vec![0usize, 2usize]),
            (Goldilocks::ONE, vec![0usize, 4usize]),
        ],
        Goldilocks::ZERO,
        &r_scale,
    );
    let r_scale_row = row_index(&r_scale, m_bits);
    let open_sum_scale = whir.sum.open(pd_sum, &proto_sum, &r_scale_row);
    let open_e_scale = whir.e.open(pd_e, &proto_e, &r_scale);
    let open_out = whir.out.open(pd_out, &proto_out, &r_scale);
    let open_rem = whir.rem.open(pd_rem, &proto_rem, &r_scale);

    Some((
        SoftmaxStatement { m, n, root_e, root_out, root_sum, root_rem, lookup: lookup_stmt },
        SoftmaxCommittedProof {
            row_sum, rescale, lookup: lookup_proof,
            open_e_sum, open_sum, open_e_scale, open_sum_scale, open_out, open_rem,
            r_sum, sum_ch, r_scale,
        },
    ))
}

pub fn verify(whir: &SoftmaxWhir, stmt: &SoftmaxStatement, proof: &SoftmaxCommittedProof) -> bool {
    let (m, n) = (stmt.m, stmt.n);
    let m_bits = m.trailing_zeros() as usize;
    let mn_bits = (m * n).trailing_zeros() as usize;
    let proto_mn = whir.e.opening_protocol(mn_bits, 1);
    let proto_m = whir.sum.opening_protocol(m_bits, 1);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    // same_poly binding for e (lookup out).
    if stmt.lookup.out.re != stmt.root_e {
        return false;
    }
    if !lookup_verify(&whir.lookup, &stmt.lookup, &proof.lookup) {
        return false;
    }

    // 2. row_sum.
    let eq_i = mle::eq_evals(&proof.r_sum);
    let eq_b: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
    let e_sum = match whir.e.verify(&stmt.root_e, &proof.open_e_sum.0, &proto_mn, &proof.sum_ch) {
        Ok(v) => v, Err(_) => return false,
    };
    let sum_val = match whir.sum.verify(&stmt.root_sum, &proof.open_sum.0, &proto_m, &proof.r_sum) {
        Ok(v) => v, Err(_) => return false,
    };
    if e_sum != proof.open_e_sum.1 || sum_val != proof.open_sum.1 {
        return false;
    }
    let row_sum_fe = vec![mle::eval(&eq_b, &proof.sum_ch), e_sum];
    if !verify_virtual(&proof.row_sum, &[(Goldilocks::ONE, vec![0usize, 1usize])], sum_val, &proof.sum_ch, &row_sum_fe) {
        return false;
    }

    // 3. rescale.
    let r_scale_row = row_index(&proof.r_scale, m_bits);
    let sum_scale = match whir.sum.verify(&stmt.root_sum, &proof.open_sum_scale.0, &proto_m, &r_scale_row) {
        Ok(v) => v, Err(_) => return false,
    };
    let e_scale = match whir.e.verify(&stmt.root_e, &proof.open_e_scale.0, &proto_mn, &proof.r_scale) {
        Ok(v) => v, Err(_) => return false,
    };
    let out_v = match whir.out.verify(&stmt.root_out, &proof.open_out.0, &proto_mn, &proof.r_scale) {
        Ok(v) => v, Err(_) => return false,
    };
    let rem_v = match whir.rem.verify(&stmt.root_rem, &proof.open_rem.0, &proto_mn, &proof.r_scale) {
        Ok(v) => v, Err(_) => return false,
    };
    if sum_scale != proof.open_sum_scale.1 || e_scale != proof.open_e_scale.1 || out_v != proof.open_out.1 || rem_v != proof.open_rem.1 {
        return false;
    }
    let eq_scale = mle::eq_evals(&proof.r_scale);
    let rescale_fe = vec![mle::eval(&eq_scale, &proof.r_scale), out_v, e_scale, sum_scale, rem_v];
    let two16 = Goldilocks::from_u64(1u64 << 16);
    let rescale_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize, 3usize]),
        (neg * two16, vec![0usize, 2usize]),
        (Goldilocks::ONE, vec![0usize, 4usize]),
    ];
    verify_virtual(&proof.rescale, &rescale_terms, Goldilocks::ZERO, &proof.r_scale, &rescale_fe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness(m: usize, n: usize, t: usize) -> (Vec<u32>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<u64>) {
        let mut rng = XorShift64::new(0x5EED);
        let table: Vec<u64> = (0..t as u64).map(|j| (j * 3 + 1) % 1000).collect();
        let idx: Vec<u32> = (0..m * n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let e: Vec<Goldilocks> = idx.iter().map(|&i| from_i64(table[i as usize] as i64)).collect();
        let sum: Vec<Goldilocks> = (0..m).map(|i| (0..n).fold(Goldilocks::ZERO, |a, j| a + e[i * n + j])).collect();
        let out: Vec<Goldilocks> = (0..m * n).map(|ij| {
            from_i64(round_div(to_i64(e[ij]) * (1i64 << 16), to_i64(sum[ij / n])))
        }).collect();
        let rem: Vec<Goldilocks> = (0..m * n).map(|ij| {
            from_i64(to_i64(e[ij]) * (1i64 << 16) - to_i64(out[ij]) * to_i64(sum[ij / n]))
        }).collect();
        (idx, e, out, sum, rem, table)
    }

    fn round_div(a: i64, b: i64) -> i64 {
        let q = a.div_euclid(b);
        let r = a.rem_euclid(b);
        if r * 2 >= b { q + 1 } else { q }
    }

    #[test]
    fn honest_roundtrip() {
        let (m, n, t) = (32usize, 8usize, 32usize);
        let whir = SoftmaxWhir::new(m, n, t, 32, 10).expect("valid dims");
        let (idx, e, out, sum, rem, table) = witness(m, n, t);
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &idx, &e, &out, &sum, &rem, &table, m, n, &mut rng).expect("prove");
        assert!(verify(&whir, &stmt, &proof));
    }

    #[test]
    fn wrong_out_rejected() {
        let (m, n, t) = (32usize, 8usize, 32usize);
        let whir = SoftmaxWhir::new(m, n, t, 32, 10).expect("valid dims");
        let (idx, e, mut out, sum, rem, table) = witness(m, n, t);
        out[0] = out[0] + Goldilocks::ONE;
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &idx, &e, &out, &sum, &rem, &table, m, n, &mut rng).expect("prove");
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_root_rejected() {
        let (m, n, t) = (32usize, 8usize, 32usize);
        let whir = SoftmaxWhir::new(m, n, t, 32, 10).expect("valid dims");
        let (idx, e, out, sum, rem, table) = witness(m, n, t);
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &idx, &e, &out, &sum, &rem, &table, m, n, &mut rng).expect("prove");
        let mut rng2 = XorShift64::new(0x888);
        let fake: Vec<Goldilocks> = (0..m * n).map(|_| rng2.field()).collect();
        let (fake_root, _, _) = whir.e.commit(&fake);
        let mut bad = stmt;
        bad.root_e = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn malformed_dims_rejected() {
        assert!(SoftmaxWhir::new(32, 8, 32, 32, 10).is_some());
        assert!(SoftmaxWhir::new(3, 8, 32, 32, 10).is_none());
        assert!(SoftmaxWhir::new(2, 2, 32, 32, 10).is_none());
    }
}

use crate::compose::{Op, Store};
use zkie_core::common::field::PrimeField64;

/// Op adapter: recognize `Op::Softmax` and materialize idx/e/out/table from a
/// real Store, compute the derived `sum`/`rem` intermediates (the forward), and
/// run the root-bound committed softmax. No forward recomputation on the
/// verifier side.
pub fn prove_op_softmax(
    whir: &SoftmaxWhir,
    op: &Op,
    store: &Store,
    rng: &mut XorShift64,
) -> Option<(SoftmaxStatement, SoftmaxCommittedProof)> {
    let Op::Softmax { idx, e, out, table, m, n } = op else {
        return None;
    };
    let idxs = store.idx.get(*idx)?;
    let ev = store.materialize(*e);
    let ov = store.materialize(*out);
    let tv = store.materialize(*table);
    if idxs.len() != *m * *n || ev.len() != *m * *n || ov.len() != *m * *n {
        return None;
    }
    let sum: Vec<Goldilocks> = (0..*m)
        .map(|i| (0..*n).fold(Goldilocks::ZERO, |a, j| a + ev[i * *n + j]))
        .collect();
    let two16 = Goldilocks::from_u64(1u64 << 16);
    let rem: Vec<Goldilocks> = (0..*m * *n)
        .map(|ij| ev[ij] * two16 - ov[ij] * sum[ij / *n])
        .collect();
    let table_u64: Vec<u64> = tv.iter().map(|&x| x.as_canonical_u64()).collect();
    prove(
        whir,
        idxs,
        ev.as_ref(),
        ov.as_ref(),
        &sum,
        &rem,
        &table_u64,
        *m,
        *n,
        rng,
    )
}
