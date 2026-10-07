//! Root-bound committed RMSNorm-style chain (no centering):
//!   mean_sq[i] = sum_j x[i,j]^2             (reduction, virtual sumcheck)
//!   mean_sq[i] = q[i] * t + idx[i], idx in [0,t)  (div-mod, committed)
//!   rsqrt[i]   = table[idx[i]]              (logUp lookup)
//!   scale[ij]  = rsqrt[i] * w[ij]           (pointwise, virtual sumcheck w/ eq)
//!   out[ij]    = x[ij]*scale[ij] + b[ij]    (pointwise, virtual sumcheck w/ eq)
//!
//! Committed tensors: x, w, b, out, mean_sq, q, idx, rsqrt, scale. Shared
//! commitments (same_poly, byte-equal roots): mean_sq (reduction + divmod.a),
//! idx (divmod.r + lookup.idx), rsqrt (lookup.out + scale), scale (scale +
//! out), x (reduction + out). The div-mod primitive supplies the truncation
//! `mean_sq % t`, so the chain is sound for arbitrary mean_sq. The verifier
//! holds `(whir, statement, proof)` only.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
use zkie_core::pcs::whir::{Commitment, Proof as WhirProof, Whir};

use crate::extension_divmod_committed::{
    prove as divmod_prove, verify as divmod_verify, DivmodProof, DivmodStatement, DivmodWhir,
};
use crate::extension_lookup_logup::{
    prove as lookup_prove, verify as lookup_verify, LookupProof, LookupStatement, LookupWhir,
};

pub const PROTOCOL: &str = "zkie/ext-layernorm-committed/v2";

pub struct LayernormWhir {
    pub x: Whir,
    pub w: Whir,
    pub b: Whir,
    pub out: Whir,
    pub mean_sq: Whir,
    pub q: Whir,
    pub idx: Whir,
    pub rsqrt: Whir,
    pub scale: Whir,
    pub divmod: DivmodWhir,
    pub lookup: LookupWhir,
}

impl LayernormWhir {
    pub fn new(m: usize, d: usize, t: usize, security_level: usize, pow_budget: usize) -> Option<Self> {
        if !m.is_power_of_two() || !d.is_power_of_two() || !t.is_power_of_two() || m < 2 || d < 2 || t < 2 {
            return None;
        }
        let ar_md = (m * d).trailing_zeros() as usize;
        let ar_m = m.trailing_zeros() as usize;
        if ar_md < 5 || ar_m < 5 {
            return None;
        }
        let mk = |v: usize| Whir::new_target(v, security_level, pow_budget);
        Some(LayernormWhir {
            x: mk(ar_md)?,
            w: mk(ar_md)?,
            b: mk(ar_md)?,
            out: mk(ar_md)?,
            mean_sq: mk(ar_m)?,
            q: mk(ar_m)?,
            idx: mk(ar_m)?,
            rsqrt: mk(ar_m)?,
            scale: mk(ar_md)?,
            divmod: DivmodWhir::new(m, t, security_level, pow_budget)?,
            lookup: LookupWhir::new(m, t, security_level, pow_budget)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct LayernormStatement {
    pub m: usize,
    pub d: usize,
    pub root_x: Commitment,
    pub root_w: Commitment,
    pub root_b: Commitment,
    pub root_out: Commitment,
    pub root_mean_sq: Commitment,
    pub root_q: Commitment,
    pub root_idx: Commitment,
    pub root_rsqrt: Commitment,
    pub root_scale: Commitment,
    pub divmod: DivmodStatement,
    pub lookup: LookupStatement,
}

pub struct LayernormCommittedProof {
    pub mean_sq: VirtualProof,
    pub scale: VirtualProof,
    pub out: VirtualProof,
    pub divmod: DivmodProof,
    pub lookup: LookupProof,
    pub open_x_mean: (WhirProof, Goldilocks),
    pub open_x_out: (WhirProof, Goldilocks),
    pub open_w: (WhirProof, Goldilocks),
    pub open_b: (WhirProof, Goldilocks),
    pub open_out: (WhirProof, Goldilocks),
    pub open_mean_sq: (WhirProof, Goldilocks),
    pub open_rsqrt: (WhirProof, Goldilocks),
    pub open_scale_own: (WhirProof, Goldilocks),
    pub open_scale_out: (WhirProof, Goldilocks),
    pub r_mean: Vec<Goldilocks>,
    pub mean_ch: Vec<Goldilocks>,
    pub r_scale: Vec<Goldilocks>,
    pub r_out: Vec<Goldilocks>,
}

fn broadcast(rsqrt: &[Goldilocks], d: usize) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; rsqrt.len() * d];
    for (i, &v) in rsqrt.iter().enumerate() {
        for j in 0..d {
            out[i * d + j] = v;
        }
    }
    out
}

fn row_index(point: &[Goldilocks], m_bits: usize) -> Vec<Goldilocks> {
    point[point.len() - m_bits..].to_vec()
}

pub fn prove(
    whir: &LayernormWhir,
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    out: &[Goldilocks],
    mean_sq: &[Goldilocks],
    q: &[Goldilocks],
    idx: &[Goldilocks],
    rsqrt: &[Goldilocks],
    scale: &[Goldilocks],
    rsqrt_table: &[u64],
    m: usize,
    d: usize,
    rng: &mut XorShift64,
) -> Option<(LayernormStatement, LayernormCommittedProof)> {
    assert_eq!(x.len(), m * d);
    assert_eq!(w.len(), m * d);
    assert_eq!(b.len(), m * d);
    assert_eq!(out.len(), m * d);
    assert_eq!(mean_sq.len(), m);
    assert_eq!(q.len(), m);
    assert_eq!(idx.len(), m);
    assert_eq!(rsqrt.len(), m);
    assert_eq!(scale.len(), m * d);

    let (root_x, pd_x, proto_x) = whir.x.commit(x);
    let (root_w, pd_w, proto_w) = whir.w.commit(w);
    let (root_b, pd_b, proto_b) = whir.b.commit(b);
    let (root_out, pd_out, proto_out) = whir.out.commit(out);
    let (root_mean_sq, pd_mean_sq, proto_mean_sq) = whir.mean_sq.commit(mean_sq);
    let (root_q, _, _) = whir.q.commit(q);
    let (root_idx, _, _) = whir.idx.commit(idx);
    let (root_rsqrt, pd_rsqrt, proto_rsqrt) = whir.rsqrt.commit(rsqrt);
    let (root_scale, pd_scale, proto_scale) = whir.scale.commit(scale);

    let m_bits = m.trailing_zeros() as usize;
    let md_bits = (m * d).trailing_zeros() as usize;
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let t = rsqrt_table.len();

    // 1. mean_sq reduction.
    let r_mean: Vec<Goldilocks> = (0..m_bits).map(|_| rng.field()).collect();
    let eq_i = mle::eq_evals(&r_mean);
    let eq_b: Vec<Goldilocks> = (0..m * d).map(|idx| eq_i[idx / d]).collect();
    let mean_claim = mle::eval(mean_sq, &r_mean);
    let mean_ch: Vec<Goldilocks> = (0..md_bits).map(|_| rng.field()).collect();
    let mean_sq_proof = prove_virtual(&[&eq_b, x], &[(Goldilocks::ONE, vec![0usize, 1usize, 1usize])], mean_claim, &mean_ch);
    let open_x_mean = whir.x.open(pd_x.clone(), &proto_x, &mean_ch);
    let open_mean_sq = whir.mean_sq.open(pd_mean_sq.clone(), &proto_mean_sq, &r_mean);

    // 2. div-mod: mean_sq = q * t + idx, idx in [0, t).
    let (divmod_stmt, divmod_proof) = divmod_prove(&whir.divmod, mean_sq, q, idx, t, rng)?;

    // 3. rsqrt lookup: rsqrt = table[idx].
    let idx_u32: Vec<u32> = idx.iter().map(|&v| to_i64(v) as u32).collect();
    let rsqrt_u64: Vec<u64> = rsqrt.iter().map(|&v| to_i64(v) as u64).collect();
    let (lookup_stmt, lookup_proof) = lookup_prove(&whir.lookup, &idx_u32, &rsqrt_u64, rsqrt_table)?;

    // 4. scale = rsqrt (broadcast) * w.
    let rsqrt_broadcast = broadcast(rsqrt, d);
    let r_scale: Vec<Goldilocks> = (0..md_bits).map(|_| rng.field()).collect();
    let eq_scale = mle::eq_evals(&r_scale);
    let scale_proof = prove_virtual(
        &[&eq_scale, scale, &rsqrt_broadcast, w],
        &[(Goldilocks::ONE, vec![0usize, 1usize]), (neg, vec![0usize, 2usize, 3usize])],
        Goldilocks::ZERO,
        &r_scale,
    );
    let r_scale_row = row_index(&r_scale, m_bits);
    let open_rsqrt = whir.rsqrt.open(pd_rsqrt.clone(), &proto_rsqrt, &r_scale_row);
    let open_w = whir.w.open(pd_w.clone(), &proto_w, &r_scale);
    let open_scale_own = whir.scale.open(pd_scale.clone(), &proto_scale, &r_scale);

    // 5. out = x * scale + b.
    let r_out: Vec<Goldilocks> = (0..md_bits).map(|_| rng.field()).collect();
    let eq_out = mle::eq_evals(&r_out);
    let out_proof = prove_virtual(
        &[&eq_out, x, scale, b, out],
        &[(Goldilocks::ONE, vec![0usize, 1usize, 2usize]), (Goldilocks::ONE, vec![0usize, 3usize]), (neg, vec![0usize, 4usize])],
        Goldilocks::ZERO,
        &r_out,
    );
    let open_x_out = whir.x.open(pd_x, &proto_x, &r_out);
    let open_scale_out = whir.scale.open(pd_scale, &proto_scale, &r_out);
    let open_b = whir.b.open(pd_b, &proto_b, &r_out);
    let open_out = whir.out.open(pd_out, &proto_out, &r_out);

    Some((
        LayernormStatement {
            m, d,
            root_x, root_w, root_b, root_out, root_mean_sq, root_q, root_idx, root_rsqrt, root_scale,
            divmod: divmod_stmt, lookup: lookup_stmt,
        },
        LayernormCommittedProof {
            mean_sq: mean_sq_proof, scale: scale_proof, out: out_proof,
            divmod: divmod_proof, lookup: lookup_proof,
            open_x_mean, open_x_out, open_w, open_b, open_out, open_mean_sq, open_rsqrt,
            open_scale_own, open_scale_out,
            r_mean, mean_ch, r_scale, r_out,
        },
    ))
}

pub fn verify(whir: &LayernormWhir, stmt: &LayernormStatement, proof: &LayernormCommittedProof) -> bool {
    let (m, d) = (stmt.m, stmt.d);
    let m_bits = m.trailing_zeros() as usize;
    let md_bits = (m * d).trailing_zeros() as usize;
    let proto_md = whir.x.opening_protocol(md_bits, 1);
    let proto_m = whir.mean_sq.opening_protocol(m_bits, 1);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    // same_poly bindings.
    if stmt.divmod.root_a != stmt.root_mean_sq || stmt.divmod.root_r != stmt.root_idx {
        return false;
    }
    if stmt.lookup.idx.re != stmt.root_idx || stmt.lookup.out.re != stmt.root_rsqrt {
        return false;
    }
    if !divmod_verify(&whir.divmod, &stmt.divmod, &proof.divmod) {
        return false;
    }
    if !lookup_verify(&whir.lookup, &stmt.lookup, &proof.lookup) {
        return false;
    }

    // 1. mean_sq reduction.
    let eq_i = mle::eq_evals(&proof.r_mean);
    let eq_b: Vec<Goldilocks> = (0..m * d).map(|idx| eq_i[idx / d]).collect();
    let x_mean = match whir.x.verify(&stmt.root_x, &proof.open_x_mean.0, &proto_md, &proof.mean_ch) {
        Ok(v) => v, Err(_) => return false,
    };
    let mean_val = match whir.mean_sq.verify(&stmt.root_mean_sq, &proof.open_mean_sq.0, &proto_m, &proof.r_mean) {
        Ok(v) => v, Err(_) => return false,
    };
    if x_mean != proof.open_x_mean.1 || mean_val != proof.open_mean_sq.1 {
        return false;
    }
    let mean_fe = vec![mle::eval(&eq_b, &proof.mean_ch), x_mean];
    if !verify_virtual(&proof.mean_sq, &[(Goldilocks::ONE, vec![0usize, 1usize, 1usize])], mean_val, &proof.mean_ch, &mean_fe) {
        return false;
    }

    // 4. scale = rsqrt * w.
    let r_scale_row = row_index(&proof.r_scale, m_bits);
    let rsqrt_v = match whir.rsqrt.verify(&stmt.root_rsqrt, &proof.open_rsqrt.0, &proto_m, &r_scale_row) {
        Ok(v) => v, Err(_) => return false,
    };
    let w_v = match whir.w.verify(&stmt.root_w, &proof.open_w.0, &proto_md, &proof.r_scale) {
        Ok(v) => v, Err(_) => return false,
    };
    let scale_own = match whir.scale.verify(&stmt.root_scale, &proof.open_scale_own.0, &proto_md, &proof.r_scale) {
        Ok(v) => v, Err(_) => return false,
    };
    if rsqrt_v != proof.open_rsqrt.1 || w_v != proof.open_w.1 || scale_own != proof.open_scale_own.1 {
        return false;
    }
    let eq_scale = mle::eq_evals(&proof.r_scale);
    let scale_fe = vec![mle::eval(&eq_scale, &proof.r_scale), scale_own, rsqrt_v, w_v];
    if !verify_virtual(&proof.scale, &[(Goldilocks::ONE, vec![0usize, 1usize]), (neg, vec![0usize, 2usize, 3usize])], Goldilocks::ZERO, &proof.r_scale, &scale_fe) {
        return false;
    }

    // 5. out = x * scale + b.
    let x_out = match whir.x.verify(&stmt.root_x, &proof.open_x_out.0, &proto_md, &proof.r_out) {
        Ok(v) => v, Err(_) => return false,
    };
    let scale_out = match whir.scale.verify(&stmt.root_scale, &proof.open_scale_out.0, &proto_md, &proof.r_out) {
        Ok(v) => v, Err(_) => return false,
    };
    let b_v = match whir.b.verify(&stmt.root_b, &proof.open_b.0, &proto_md, &proof.r_out) {
        Ok(v) => v, Err(_) => return false,
    };
    let out_v = match whir.out.verify(&stmt.root_out, &proof.open_out.0, &proto_md, &proof.r_out) {
        Ok(v) => v, Err(_) => return false,
    };
    if x_out != proof.open_x_out.1 || scale_out != proof.open_scale_out.1 || b_v != proof.open_b.1 || out_v != proof.open_out.1 {
        return false;
    }
    let eq_out = mle::eq_evals(&proof.r_out);
    let out_fe = vec![mle::eval(&eq_out, &proof.r_out), x_out, scale_out, b_v, out_v];
    let out_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
        (Goldilocks::ONE, vec![0usize, 3usize]),
        (neg, vec![0usize, 4usize]),
    ];
    verify_virtual(&proof.out, &out_terms, Goldilocks::ZERO, &proof.r_out, &out_fe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness(m: usize, d: usize, t: usize) -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<u64>) {
        let mut rng = XorShift64::new(0x5EED);
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let w: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
        let mean_sq: Vec<Goldilocks> = (0..m).map(|i| {
            (0..d).fold(Goldilocks::ZERO, |a, j| a + x[i * d + j] * x[i * d + j])
        }).collect();
        let table: Vec<u64> = (0..t as u64).map(|j| (j * 7 + 1) % 1000).collect();
        let q: Vec<Goldilocks> = mean_sq.iter().map(|&v| from_i64(to_i64(v) / t as i64)).collect();
        let idx: Vec<Goldilocks> = mean_sq.iter().map(|&v| from_i64(to_i64(v) % t as i64)).collect();
        let idx_u32: Vec<u32> = idx.iter().map(|&v| to_i64(v) as u32).collect();
        let rsqrt: Vec<Goldilocks> = idx_u32.iter().map(|&i| from_i64(table[i as usize] as i64)).collect();
        let scale: Vec<Goldilocks> = (0..m * d).map(|ij| rsqrt[ij / d] * w[ij]).collect();
        let out: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] * scale[ij] + b[ij]).collect();
        (x, w, b, out, mean_sq, q, idx, rsqrt, scale, table)
    }

    #[test]
    fn honest_roundtrip() {
        let (m, d, t) = (32usize, 8usize, 32usize);
        let whir = LayernormWhir::new(m, d, t, 32, 10).expect("valid dims");
        let (x, w, b, out, mean_sq, q, idx, rsqrt, scale, table) = witness(m, d, t);
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &x, &w, &b, &out, &mean_sq, &q, &idx, &rsqrt, &scale, &table, m, d, &mut rng).expect("prove");
        assert!(verify(&whir, &stmt, &proof));
    }

    #[test]
    fn wrong_out_rejected() {
        let (m, d, t) = (32usize, 8usize, 32usize);
        let whir = LayernormWhir::new(m, d, t, 32, 10).expect("valid dims");
        let (x, w, b, mut out, mean_sq, q, idx, rsqrt, scale, table) = witness(m, d, t);
        out[0] = out[0] + Goldilocks::ONE;
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &x, &w, &b, &out, &mean_sq, &q, &idx, &rsqrt, &scale, &table, m, d, &mut rng).expect("prove");
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_root_rejected() {
        let (m, d, t) = (32usize, 8usize, 32usize);
        let whir = LayernormWhir::new(m, d, t, 32, 10).expect("valid dims");
        let (x, w, b, out, mean_sq, q, idx, rsqrt, scale, table) = witness(m, d, t);
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &x, &w, &b, &out, &mean_sq, &q, &idx, &rsqrt, &scale, &table, m, d, &mut rng).expect("prove");
        let mut rng2 = XorShift64::new(0x999);
        let fake: Vec<Goldilocks> = (0..m * d).map(|_| rng2.field()).collect();
        let (fake_root, _, _) = whir.out.commit(&fake);
        let mut bad = stmt;
        bad.root_out = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn malformed_dims_rejected() {
        assert!(LayernormWhir::new(32, 8, 32, 32, 10).is_some());
        assert!(LayernormWhir::new(3, 8, 32, 32, 10).is_none());
        assert!(LayernormWhir::new(2, 2, 32, 32, 10).is_none());
    }
}
