//! Root-bound committed div-mod (truncation) primitive:
//!   a[i] = q[i] * d + r[i]        (pointwise quotient relation, virtual sumcheck)
//!   r[i] in [0, d)                (range check, root-bound logUp lookup)
//!
//! Three committed tensors: a, q, r. Shared commitment (same_poly) for `r`:
//! the range-check lookup's idx/out re-roots equal the quotient's `r` root, so a
//! prover cannot use one `r` in the quotient and another in the range check.
//! `d` is a public power-of-two divisor. The verifier holds `(whir, statement,
//! proof)` only: no witness, no Store, no forward recomputation.
//!
//! Reusable across the norm ops: layernorm / rmsnorm (idx = mean_sq % t),
//! softmax / rope (fractional index / range), projection rounding, etc.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
use zkie_core::pcs::whir::{Commitment, Proof as WhirProof, Whir};

use crate::extension_lookup_logup::{
    prove as lookup_prove, verify as lookup_verify, LookupProof, LookupStatement, LookupWhir,
};

pub const PROTOCOL: &str = "zkie/ext-divmod-committed/v1";

pub struct DivmodWhir {
    pub a: Whir,
    pub q: Whir,
    pub r: Whir,
    pub range: LookupWhir,
}

impl DivmodWhir {
    pub fn new(n: usize, d: usize, security_level: usize, pow_budget: usize) -> Option<Self> {
        if !n.is_power_of_two() || !d.is_power_of_two() || n < 2 || d < 2 {
            return None;
        }
        let ar = n.trailing_zeros() as usize;
        if ar < 5 {
            return None;
        }
        Some(DivmodWhir {
            a: Whir::new_target(ar, security_level, pow_budget)?,
            q: Whir::new_target(ar, security_level, pow_budget)?,
            r: Whir::new_target(ar, security_level, pow_budget)?,
            range: LookupWhir::new(n, d, security_level, pow_budget)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DivmodStatement {
    pub n: usize,
    pub d: usize,
    pub root_a: Commitment,
    pub root_q: Commitment,
    pub root_r: Commitment,
    pub lookup: LookupStatement,
}

pub struct DivmodProof {
    pub quotient: VirtualProof,
    pub lookup: LookupProof,
    pub open_a: (WhirProof, Goldilocks),
    pub open_q: (WhirProof, Goldilocks),
    pub open_r: (WhirProof, Goldilocks),
    pub pt: Vec<Goldilocks>,
}

pub fn prove(
    whir: &DivmodWhir,
    a: &[Goldilocks],
    q: &[Goldilocks],
    r: &[Goldilocks],
    d: usize,
    rng: &mut XorShift64,
) -> Option<(DivmodStatement, DivmodProof)> {
    let n = a.len();
    assert_eq!(q.len(), n);
    assert_eq!(r.len(), n);

    let (root_a, pd_a, proto_a) = whir.a.commit(a);
    let (root_q, pd_q, proto_q) = whir.q.commit(q);
    let (root_r, pd_r, proto_r) = whir.r.commit(r);

    let ar = n.trailing_zeros() as usize;
    let pt: Vec<Goldilocks> = (0..ar).map(|_| rng.field()).collect();
    let eq = mle::eq_evals(&pt);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let d_f = Goldilocks::from_u64(d as u64);
    // a - q*d - r = 0, pointwise via eq.
    let quotient = prove_virtual(
        &[&eq, a, q, r],
        &[
            (Goldilocks::ONE, vec![0usize, 1usize]),
            (neg * d_f, vec![0usize, 2usize]),
            (neg, vec![0usize, 3usize]),
        ],
        Goldilocks::ZERO,
        &pt,
    );
    let open_a = whir.a.open(pd_a, &proto_a, &pt);
    let open_q = whir.q.open(pd_q, &proto_q, &pt);
    let open_r = whir.r.open(pd_r, &proto_r, &pt);

    // range check r in [0, d): idx = r, out = r, table = [0, d).
    let idx: Vec<u32> = r.iter().map(|&v| to_i64(v) as u32).collect();
    let r_u64: Vec<u64> = r.iter().map(|&v| to_i64(v) as u64).collect();
    let table: Vec<u64> = (0..d as u64).collect();
    let (lookup_stmt, lookup_proof) = lookup_prove(&whir.range, &idx, &r_u64, &table)?;

    Some((
        DivmodStatement { n, d, root_a, root_q, root_r, lookup: lookup_stmt },
        DivmodProof { quotient, lookup: lookup_proof, open_a, open_q, open_r, pt },
    ))
}

pub fn verify(whir: &DivmodWhir, stmt: &DivmodStatement, proof: &DivmodProof) -> bool {
    // same_poly binding for r (quotient relation vs range check).
    if stmt.lookup.idx.re != stmt.root_r || stmt.lookup.out.re != stmt.root_r {
        return false;
    }
    if !lookup_verify(&whir.range, &stmt.lookup, &proof.lookup) {
        return false;
    }

    let ar = stmt.n.trailing_zeros() as usize;
    let proto = whir.a.opening_protocol(ar, 1);
    let a_v = match whir.a.verify(&stmt.root_a, &proof.open_a.0, &proto, &proof.pt) {
        Ok(v) => v, Err(_) => return false,
    };
    let q_v = match whir.q.verify(&stmt.root_q, &proof.open_q.0, &proto, &proof.pt) {
        Ok(v) => v, Err(_) => return false,
    };
    let r_v = match whir.r.verify(&stmt.root_r, &proof.open_r.0, &proto, &proof.pt) {
        Ok(v) => v, Err(_) => return false,
    };
    if a_v != proof.open_a.1 || q_v != proof.open_q.1 || r_v != proof.open_r.1 {
        return false;
    }
    let eq = mle::eq_evals(&proof.pt);
    let fe = vec![mle::eval(&eq, &proof.pt), a_v, q_v, r_v];
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let d_f = Goldilocks::from_u64(stmt.d as u64);
    let terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg * d_f, vec![0usize, 2usize]),
        (neg, vec![0usize, 3usize]),
    ];
    verify_virtual(&proof.quotient, &terms, Goldilocks::ZERO, &proof.pt, &fe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness(n: usize, d: usize) -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>) {
        let mut rng = XorShift64::new(0xD1F);
        let a: Vec<Goldilocks> = (0..n).map(|_| from_i64((rng.next_u64() % 100000) as i64)).collect();
        let q: Vec<Goldilocks> = a.iter().map(|&v| from_i64(to_i64(v) / d as i64)).collect();
        let r: Vec<Goldilocks> = a.iter().map(|&v| from_i64(to_i64(v) % d as i64)).collect();
        (a, q, r)
    }

    #[test]
    fn honest_roundtrip() {
        let (n, d) = (64usize, 32usize);
        let whir = DivmodWhir::new(n, d, 32, 10).expect("valid dims");
        let (a, q, r) = witness(n, d);
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &a, &q, &r, d, &mut rng).expect("prove");
        assert!(verify(&whir, &stmt, &proof));
    }

    #[test]
    fn wrong_quotient_rejected() {
        let (n, d) = (64usize, 32usize);
        let whir = DivmodWhir::new(n, d, 32, 10).expect("valid dims");
        let (a, mut q, r) = witness(n, d);
        q[0] = q[0] + Goldilocks::ONE;
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &a, &q, &r, d, &mut rng).expect("prove");
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn out_of_range_rejected() {
        let (n, d) = (64usize, 32usize);
        let whir = DivmodWhir::new(n, d, 32, 10).expect("valid dims");
        let (a, q, mut r) = witness(n, d);
        r[0] = from_i64(d as i64); // r == d, out of [0, d)
        let mut rng = XorShift64::new(0xABC);
        // prove returns None because idx = r is out of the table range.
        assert!(prove(&whir, &a, &q, &r, d, &mut rng).is_none());
    }

    #[test]
    fn tampered_root_rejected() {
        let (n, d) = (64usize, 32usize);
        let whir = DivmodWhir::new(n, d, 32, 10).expect("valid dims");
        let (a, q, r) = witness(n, d);
        let mut rng = XorShift64::new(0xABC);
        let (stmt, proof) = prove(&whir, &a, &q, &r, d, &mut rng).expect("prove");
        let mut rng2 = XorShift64::new(0x777);
        let fake: Vec<Goldilocks> = (0..n).map(|_| rng2.field()).collect();
        let (fake_root, _, _) = whir.a.commit(&fake);
        let mut bad = stmt;
        bad.root_a = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn malformed_dims_rejected() {
        assert!(DivmodWhir::new(64, 32, 32, 10).is_some());
        assert!(DivmodWhir::new(16, 32, 32, 10).is_none(), "arity below folding floor");
        assert!(DivmodWhir::new(64, 31, 32, 10).is_none(), "d not power of two");
    }
}
