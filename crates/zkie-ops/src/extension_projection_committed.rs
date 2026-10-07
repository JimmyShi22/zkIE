//! Root-bound committed projection block (matmul + affine), with the matmul
//! output `h = x @ W` kept VIRTUAL (never committed — its value at the affine
//! point is the GKR claim). Five committed tensors: x, w, bias, out, rem.
//!
//! Relations proven:
//! - matmul: `h = x @ W` via one GKR sumcheck, leaving terminal claims on x and w;
//! - affine: `rem = h - (out - bias) * 2^shift + half` via one virtual sumcheck,
//!   with `h` supplied by the GKR claim and out/rem/bias by committed openings.
//!
//! The verifier holds `(whir, statement, proof)` only — no witness, no Store,
//! no forward recomputation. Every value it uses comes from a WHIR opening or a
//! public constant.
//!
//! NOTE (soundness of "rounding"): this module proves the affine + matmul
//! relation, but the RANGE CHECK `rem in [0, 2^shift)` — which is what makes
//! `out` the actual round-half-up value rather than an arbitrary affine image —
//! is a separate committed logUp lookup layered on top (reusing
//! `extension_lookup_logup`). Without it a cheating prover could pick a
//! non-rounded `out` and a compensating `rem`.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_core::common::matmul::{
    prove as matmul_prove, verify as matmul_verify, MatmulProof as GkrProof,
};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
use zkie_core::pcs::whir::{Commitment, Proof as WhirProof, Whir};

pub const PROTOCOL: &str = "zkie/ext-projection-committed/v1";

pub struct ProjectionWhir {
    pub x: Whir,
    pub w: Whir,
    pub bias: Whir,
    pub out: Whir,
    pub rem: Whir,
}

impl ProjectionWhir {
    pub fn new(m: usize, k: usize, n: usize, security_level: usize, pow_budget: usize) -> Option<Self> {
        if !m.is_power_of_two() || !k.is_power_of_two() || !n.is_power_of_two() {
            return None;
        }
        let ar_x = (m * k).trailing_zeros() as usize;
        let ar_w = (k * n).trailing_zeros() as usize;
        let ar_y = (m * n).trailing_zeros() as usize;
        if ar_x < 5 || ar_w < 5 || ar_y < 5 {
            return None;
        }
        Some(ProjectionWhir {
            x: Whir::new_target(ar_x, security_level, pow_budget)?,
            w: Whir::new_target(ar_w, security_level, pow_budget)?,
            bias: Whir::new_target(ar_y, security_level, pow_budget)?,
            out: Whir::new_target(ar_y, security_level, pow_budget)?,
            rem: Whir::new_target(ar_y, security_level, pow_budget)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ProjectionStatement {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub shift: u32,
    pub root_x: Commitment,
    pub root_w: Commitment,
    pub root_bias: Commitment,
    pub root_out: Commitment,
    pub root_rem: Commitment,
}

pub struct ProjectionCommittedProof {
    pub matmul: GkrProof,
    pub affine: VirtualProof,
    pub open_x: (WhirProof, Goldilocks),
    pub open_w: (WhirProof, Goldilocks),
    pub open_rem: (WhirProof, Goldilocks),
    pub open_out: (WhirProof, Goldilocks),
    pub open_bias: (WhirProof, Goldilocks),
    pub u: Vec<Goldilocks>,
    pub v: Vec<Goldilocks>,
    pub ch: Vec<Goldilocks>,
    pub pt: Vec<Goldilocks>,
}

fn transpose(x: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut t = vec![Goldilocks::ZERO; m * k];
    for i in 0..m {
        for j in 0..k {
            t[j * m + i] = x[i * k + j];
        }
    }
    t
}

fn matmul_full(x: &[Goldilocks], w: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = Goldilocks::ZERO;
            for t in 0..k {
                s = s + x[i * k + t] * w[t * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

pub fn prove(
    whir: &ProjectionWhir,
    x: &[Goldilocks],
    w: &[Goldilocks],
    bias: &[Goldilocks],
    out: &[Goldilocks],
    rem: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    shift: u32,
    rng: &mut XorShift64,
) -> (ProjectionStatement, ProjectionCommittedProof) {
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), k * n);
    assert_eq!(bias.len(), m * n);
    assert_eq!(out.len(), m * n);
    assert_eq!(rem.len(), m * n);

    let (root_x, pd_x, proto_x) = whir.x.commit(x);
    let (root_w, pd_w, proto_w) = whir.w.commit(w);
    let (root_bias, pd_bias, proto_bias) = whir.bias.commit(bias);
    let (root_out, pd_out, proto_out) = whir.out.commit(out);
    let (root_rem, pd_rem, proto_rem) = whir.rem.commit(rem);

    let h = matmul_full(x, w, m, k, n);

    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let mut pt = v.clone();
    pt.extend_from_slice(&u);

    let at = transpose(x, m, k);
    let matmul = matmul_prove(&at, w, &h, m, k, n, &u, &v, &ch);

    // Matmul terminal points (LSB-first): x at [ch, u], w at [v, ch].
    let mut xp = ch.clone();
    xp.extend_from_slice(&u);
    let mut wp = v.clone();
    wp.extend_from_slice(&ch);
    let open_x = whir.x.open(pd_x, &proto_x, &xp);
    let open_w = whir.w.open(pd_w, &proto_w, &wp);

    // Affine relation over [eq, rem, h, out, bias, ones] at point pt.
    let eq = mle::eq_evals(&pt);
    let ones = vec![Goldilocks::from_u64(1); m * n];
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let neg = from_i64(-1);
    let terms = vec![
        (Goldilocks::from_u64(1), vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (two_shift, vec![0usize, 3usize]),
        (neg * two_shift, vec![0usize, 4usize]),
        (neg * half, vec![0usize, 5usize]),
    ];
    let mles: Vec<&[Goldilocks]> = vec![&eq, rem, &h, out, bias, &ones];
    let affine = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &pt);

    let open_rem = whir.rem.open(pd_rem, &proto_rem, &pt);
    let open_out = whir.out.open(pd_out, &proto_out, &pt);
    let open_bias = whir.bias.open(pd_bias, &proto_bias, &pt);

    (
        ProjectionStatement { m, k, n, shift, root_x, root_w, root_bias, root_out, root_rem },
        ProjectionCommittedProof {
            matmul, affine, open_x, open_w, open_rem, open_out, open_bias, u, v, ch, pt,
        },
    )
}

pub fn verify(
    whir: &ProjectionWhir,
    stmt: &ProjectionStatement,
    proof: &ProjectionCommittedProof,
) -> bool {
    let (m, k, n) = (stmt.m, stmt.k, stmt.n);

    // Matmul terminal points.
    let mut xp = proof.ch.clone();
    xp.extend_from_slice(&proof.u);
    let mut wp = proof.v.clone();
    wp.extend_from_slice(&proof.ch);

    let ar_x = (m * k).trailing_zeros() as usize;
    let ar_w = (k * n).trailing_zeros() as usize;
    let ar_y = (m * n).trailing_zeros() as usize;
    let proto_x = whir.x.opening_protocol(ar_x, 1);
    let proto_w = whir.w.opening_protocol(ar_w, 1);
    let proto_y = whir.bias.opening_protocol(ar_y, 1);

    let f_eval = match whir.x.verify(&stmt.root_x, &proof.open_x.0, &proto_x, &xp) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let h_eval = match whir.w.verify(&stmt.root_w, &proof.open_w.0, &proto_w, &wp) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if f_eval != proof.open_x.1 || h_eval != proof.open_w.1 {
        return false;
    }
    if !matmul_verify(&proof.matmul, &proof.ch, f_eval, h_eval) {
        return false;
    }

    let rem_pt = match whir.rem.verify(&stmt.root_rem, &proof.open_rem.0, &proto_y, &proof.pt) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let out_pt = match whir.out.verify(&stmt.root_out, &proof.open_out.0, &proto_y, &proof.pt) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let bias_pt = match whir.bias.verify(&stmt.root_bias, &proof.open_bias.0, &proto_y, &proof.pt) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if rem_pt != proof.open_rem.1 || out_pt != proof.open_out.1 || bias_pt != proof.open_bias.1 {
        return false;
    }

    let eq = mle::eq_evals(&proof.pt);
    let ones = vec![Goldilocks::from_u64(1); m * n];
    let fe = vec![
        mle::eval(&eq, &proof.pt),   // eq(pt) == 1
        rem_pt,                       // rem(pt)
        proof.matmul.claimed,         // h(pt) from the GKR claim
        out_pt,                       // out(pt)
        bias_pt,                      // bias(pt)
        mle::eval(&ones, &proof.pt),  // ones(pt) == 1
    ];
    let half = Goldilocks::from_u64(1u64 << (stmt.shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << stmt.shift);
    let neg = from_i64(-1);
    let terms = vec![
        (Goldilocks::from_u64(1), vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (two_shift, vec![0usize, 3usize]),
        (neg * two_shift, vec![0usize, 4usize]),
        (neg * half, vec![0usize, 5usize]),
    ];
    verify_virtual(&proof.affine, &terms, Goldilocks::from_u64(0), &proof.pt, &fe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn div_round(a: i64, b: i64) -> i64 {
        let q = a.div_euclid(b);
        let r = a.rem_euclid(b);
        if r * 2 >= b { q + 1 } else { q }
    }

    fn witness(rng: &mut XorShift64, m: usize, k: usize, n: usize, shift: u32)
        -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>)
    {
        let half = Goldilocks::from_u64(1u64 << (shift - 1));
        let two_shift = Goldilocks::from_u64(1u64 << shift);
        let x: Vec<Goldilocks> = (0..m * k).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let w: Vec<Goldilocks> = (0..k * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
        let h = matmul_full(&x, &w, m, k, n);
        let out: Vec<Goldilocks> = (0..m * n).map(|ij| {
            from_i64(div_round(to_i64(h[ij]), 1i64 << shift) + to_i64(bias[ij]))
        }).collect();
        let rem: Vec<Goldilocks> = (0..m * n).map(|ij| {
            let h_i = to_i64(h[ij]);
            let o_i = to_i64(out[ij]);
            let b_i = to_i64(bias[ij]);
            from_i64(h_i - (o_i - b_i) * (1i64 << shift) + (1i64 << (shift - 1)))
        }).collect();
        let _ = (half, two_shift);
        (x, w, bias, out, rem)
    }

    #[test]
    fn honest_roundtrip() {
        let (m, k, n, shift) = (8usize, 8usize, 8usize, 8u32);
        let whir = ProjectionWhir::new(m, k, n, 32, 10).expect("valid dims");
        let mut rng = XorShift64::new(0xABCD);
        let (x, w, bias, out, rem) = witness(&mut rng, m, k, n, shift);
        let (stmt, proof) = prove(&whir, &x, &w, &bias, &out, &rem, m, k, n, shift, &mut rng);
        assert!(verify(&whir, &stmt, &proof));
    }

    #[test]
    fn wrong_out_rejected() {
        let (m, k, n, shift) = (8usize, 8usize, 8usize, 8u32);
        let whir = ProjectionWhir::new(m, k, n, 32, 10).expect("valid dims");
        let mut rng = XorShift64::new(0xBEEF);
        let (x, w, bias, mut out, rem) = witness(&mut rng, m, k, n, shift);
        out[0] = out[0] + Goldilocks::ONE;
        let (stmt, proof) = prove(&whir, &x, &w, &bias, &out, &rem, m, k, n, shift, &mut rng);
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_root_rejected() {
        let (m, k, n, shift) = (8usize, 8usize, 8usize, 8u32);
        let whir = ProjectionWhir::new(m, k, n, 32, 10).expect("valid dims");
        let mut rng = XorShift64::new(0xC0FFEE);
        let (x, w, bias, out, rem) = witness(&mut rng, m, k, n, shift);
        let (stmt, proof) = prove(&whir, &x, &w, &bias, &out, &rem, m, k, n, shift, &mut rng);
        let fake: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let (fake_root, _, _) = whir.out.commit(&fake);
        let mut bad = stmt;
        bad.root_out = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn malformed_dims_rejected() {
        assert!(ProjectionWhir::new(8, 8, 8, 32, 10).is_some());
        assert!(ProjectionWhir::new(3, 8, 8, 32, 10).is_none());
        assert!(ProjectionWhir::new(2, 2, 2, 32, 10).is_none());
    }
}
