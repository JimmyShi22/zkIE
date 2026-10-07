//! Root-bound committed single-contraction matmul: `C = A @ B`, proven by one
//! GKR sumcheck over the contraction index, with A / B / C committed and every
//! terminal claim opened from its commitment. The verifier holds
//! `(whir, statement, proof, u, v, ch)` only: no witness, no Store, no raw
//! tensors, no forward recomputation.
//!
//! Claim points mirror `compose::verify_shard_precomputed` (LSB-first MLE
//! convention):
//! - A opened at `[ch (k low), u (m high)]`  -> `f_eval = A(u, ch)`;
//! - B opened at `[v (n low), ch (k high)]`  -> `h_eval = B(ch, v)`;
//! - C opened at `[v (n low), u (m high)]`   -> `c_val  = C(u, v)`.
//!
//! Soundness: `c_val == gkr.claimed` binds the committed C to the sumcheck, and
//! `matmul::verify(gkr, ch, f_eval, h_eval)` binds the sumcheck to the opened
//! A / B terminal values. Every value the verifier uses comes from a WHIR
//! opening of a committed root, never from recomputing the witness.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::matmul::{
    prove as matmul_prove, verify as matmul_verify, MatmulProof as GkrProof,
};
use zkie_core::pcs::whir::{Commitment, Proof as WhirProof, Whir};

pub const PROTOCOL: &str = "zkie/ext-matmul-committed/v1";

/// One `Whir` instance per operand (each WHIR instance is bound to one arity).
pub struct MatmulWhir {
    pub a: Whir,
    pub b: Whir,
    pub c: Whir,
}

impl MatmulWhir {
    pub fn new(m: usize, k: usize, n: usize, security_level: usize, pow_budget: usize) -> Option<Self> {
        if !m.is_power_of_two() || !k.is_power_of_two() || !n.is_power_of_two() {
            return None;
        }
        let ar_a = (m * k).trailing_zeros() as usize;
        let ar_b = (k * n).trailing_zeros() as usize;
        let ar_c = (m * n).trailing_zeros() as usize;
        if ar_a < 5 || ar_b < 5 || ar_c < 5 {
            return None;
        }
        Some(MatmulWhir {
            a: Whir::new_target(ar_a, security_level, pow_budget)?,
            b: Whir::new_target(ar_b, security_level, pow_budget)?,
            c: Whir::new_target(ar_c, security_level, pow_budget)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct MatmulStatement {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub root_a: Commitment,
    pub root_b: Commitment,
    pub root_c: Commitment,
}

pub struct MatmulCommittedProof {
    pub gkr: GkrProof,
    pub open_a: (WhirProof, Goldilocks),
    pub open_b: (WhirProof, Goldilocks),
    pub open_c: (WhirProof, Goldilocks),
}

fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut at = vec![Goldilocks::ZERO; m * k];
    for i in 0..m {
        for j in 0..k {
            at[j * m + i] = a[i * k + j];
        }
    }
    at
}

fn claim_points(
    u: &[Goldilocks],
    v: &[Goldilocks],
    ch: &[Goldilocks],
) -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>) {
    let mut ap = ch.to_vec();
    ap.extend_from_slice(u);
    let mut bp = v.to_vec();
    bp.extend_from_slice(ch);
    let mut cp = v.to_vec();
    cp.extend_from_slice(u);
    (ap, bp, cp)
}

pub fn prove(
    whir: &MatmulWhir,
    a: &[Goldilocks],
    b: &[Goldilocks],
    c: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    ch: &[Goldilocks],
) -> (MatmulStatement, MatmulCommittedProof) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    let (root_a, pd_a, proto_a) = whir.a.commit(a);
    let (root_b, pd_b, proto_b) = whir.b.commit(b);
    let (root_c, pd_c, proto_c) = whir.c.commit(c);

    let at = transpose(a, m, k);
    let gkr = matmul_prove(&at, b, c, m, k, n, u, v, ch);

    let (ap, bp, cp) = claim_points(u, v, ch);
    let open_a = whir.a.open(pd_a, &proto_a, &ap);
    let open_b = whir.b.open(pd_b, &proto_b, &bp);
    let open_c = whir.c.open(pd_c, &proto_c, &cp);

    (
        MatmulStatement { m, k, n, root_a, root_b, root_c },
        MatmulCommittedProof { gkr, open_a, open_b, open_c },
    )
}

pub fn verify(
    whir: &MatmulWhir,
    stmt: &MatmulStatement,
    proof: &MatmulCommittedProof,
    u: &[Goldilocks],
    v: &[Goldilocks],
    ch: &[Goldilocks],
) -> bool {
    let (ap, bp, cp) = claim_points(u, v, ch);

    let ar_a = (stmt.m * stmt.k).trailing_zeros() as usize;
    let ar_b = (stmt.k * stmt.n).trailing_zeros() as usize;
    let ar_c = (stmt.m * stmt.n).trailing_zeros() as usize;
    let proto_a = whir.a.opening_protocol(ar_a, 1);
    let proto_b = whir.b.opening_protocol(ar_b, 1);
    let proto_c = whir.c.opening_protocol(ar_c, 1);

    let f_eval = match whir.a.verify(&stmt.root_a, &proof.open_a.0, &proto_a, &ap) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let h_eval = match whir.b.verify(&stmt.root_b, &proof.open_b.0, &proto_b, &bp) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let c_val = match whir.c.verify(&stmt.root_c, &proof.open_c.0, &proto_c, &cp) {
        Ok(v) => v,
        Err(_) => return false,
    };

    if f_eval != proof.open_a.1 || h_eval != proof.open_b.1 || c_val != proof.open_c.1 {
        return false;
    }
    if c_val != proof.gkr.claimed {
        return false;
    }

    matmul_verify(&proof.gkr, ch, f_eval, h_eval)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gold_mm(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
        let mut c = vec![Goldilocks::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = Goldilocks::ZERO;
                for t in 0..k {
                    s = s + a[i * k + t] * b[t * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn challenges(rng: &mut XorShift64, bits: usize) -> Vec<Goldilocks> {
        (0..bits).map(|_| rng.field()).collect()
    }

    #[test]
    fn honest_roundtrip() {
        let (m, k, n) = (8usize, 8usize, 8usize);
        let whir = MatmulWhir::new(m, k, n, 32, 10).expect("valid dims");
        let mut rng = XorShift64::new(0xA11CE);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let c = gold_mm(&a, &b, m, k, n);
        let u = challenges(&mut rng, m.trailing_zeros() as usize);
        let v = challenges(&mut rng, n.trailing_zeros() as usize);
        let ch = challenges(&mut rng, k.trailing_zeros() as usize);
        let (stmt, proof) = prove(&whir, &a, &b, &c, m, k, n, &u, &v, &ch);
        assert!(verify(&whir, &stmt, &proof, &u, &v, &ch));
    }

    #[test]
    fn wrong_output_rejected() {
        let (m, k, n) = (8usize, 8usize, 8usize);
        let whir = MatmulWhir::new(m, k, n, 32, 10).expect("valid dims");
        let mut rng = XorShift64::new(0xBEEF);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let mut c = gold_mm(&a, &b, m, k, n);
        c[0] = c[0] + Goldilocks::ONE;
        let u = challenges(&mut rng, m.trailing_zeros() as usize);
        let v = challenges(&mut rng, n.trailing_zeros() as usize);
        let ch = challenges(&mut rng, k.trailing_zeros() as usize);
        let (stmt, proof) = prove(&whir, &a, &b, &c, m, k, n, &u, &v, &ch);
        assert!(!verify(&whir, &stmt, &proof, &u, &v, &ch));
    }

    #[test]
    fn tampered_root_rejected() {
        let (m, k, n) = (8usize, 8usize, 8usize);
        let whir = MatmulWhir::new(m, k, n, 32, 10).expect("valid dims");
        let mut rng = XorShift64::new(0xC0FFEE);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let c = gold_mm(&a, &b, m, k, n);
        let u = challenges(&mut rng, m.trailing_zeros() as usize);
        let v = challenges(&mut rng, n.trailing_zeros() as usize);
        let ch = challenges(&mut rng, k.trailing_zeros() as usize);
        let (stmt, proof) = prove(&whir, &a, &b, &c, m, k, n, &u, &v, &ch);
        let fake: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let (fake_root, _, _) = whir.c.commit(&fake);
        let mut bad = stmt;
        bad.root_c = fake_root;
        assert!(!verify(&whir, &bad, &proof, &u, &v, &ch));
    }

    #[test]
    fn malformed_dims_rejected() {
        assert!(MatmulWhir::new(8, 8, 8, 32, 10).is_some());
        assert!(MatmulWhir::new(7, 8, 8, 32, 10).is_none(), "m not power of two");
        assert!(MatmulWhir::new(2, 2, 2, 32, 10).is_none(), "arity below folding floor");
    }
}

use crate::compose::{Op, Store};

/// Op adapter: recognize `Op::MatMul` and materialize a/b/c from a real Store,
/// then run the root-bound committed matmul. Returns the committed statement,
/// proof, and the three challenge vectors (u, v, ch) the verifier needs — the
/// caller (a shard-DAG verifier) must carry them. No forward recomputation.
pub fn prove_op_matmul(
    whir: &MatmulWhir,
    op: &Op,
    store: &Store,
    rng: &mut XorShift64,
) -> Option<(
    MatmulStatement,
    MatmulCommittedProof,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
)> {
    let Op::MatMul { a, b, c, m, k, n } = op else {
        return None;
    };
    let av = store.materialize(*a);
    let bv = store.materialize(*b);
    let cv = store.materialize(*c);
    if av.len() != *m * *k || bv.len() != *k * *n || cv.len() != *m * *n {
        return None;
    }
    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let (stmt, proof) = prove(whir, av.as_ref(), bv.as_ref(), cv.as_ref(), *m, *k, *n, &u, &v, &ch);
    Some((stmt, proof, u, v, ch))
}
