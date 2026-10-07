//! Root-bound claim-driven two-op shard: MatMul (C = A @ B) feeding a Lookup
//! (out = table[idx]) where the matmul output C IS the lookup table. The shared
//! tensor is committed once; the two committed op verifiers are bound by
//! byte-equal roots (`matmul.root_c == lookup.table.re`), so the composed shard
//! is sound: a prover cannot use one C in the matmul and a different table in
//! the lookup. The verifier holds `(whir, statement, proof)` only — no witness,
//! no Store, no forward recomputation.
//!
//! This is the "wire committed ops into a shard DAG" integration shape: each op
//! keeps its own root-only proof, and cross-op tensors are bound by shared
//! commitments (same_poly), exactly the binding the aggregation envelope uses.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::pcs::whir::Commitment;

use crate::extension_lookup_logup::{
    prove as lookup_prove, verify as lookup_verify, LookupProof, LookupStatement, LookupWhir,
};
use crate::extension_matmul_committed::{
    prove as matmul_prove, verify as matmul_verify, MatmulCommittedProof, MatmulStatement, MatmulWhir,
};

pub const PROTOCOL: &str = "zkie/ext-shard-matmul-lookup/v1";

pub struct ShardMatmulLookupWhir {
    pub matmul: MatmulWhir,
    pub lookup: LookupWhir,
}

impl ShardMatmulLookupWhir {
    pub fn new(m: usize, k: usize, n: usize, security_level: usize, pow_budget: usize) -> Option<Self> {
        // m*n must equal the lookup table size so C == table (same arity).
        let t = m * n;
        if t < 2 || !t.is_power_of_two() {
            return None;
        }
        Some(ShardMatmulLookupWhir {
            matmul: MatmulWhir::new(m, k, n, security_level, pow_budget)?,
            lookup: LookupWhir::new(t, t, security_level, pow_budget)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ShardMatmulLookupStatement {
    pub matmul: MatmulStatement,
    pub lookup: LookupStatement,
}

pub struct ShardMatmulLookupProof {
    pub matmul: MatmulCommittedProof,
    pub lookup: LookupProof,
    pub u: Vec<Goldilocks>,
    pub v: Vec<Goldilocks>,
    pub ch: Vec<Goldilocks>,
}

pub fn prove(
    whir: &ShardMatmulLookupWhir,
    a: &[Goldilocks],
    b: &[Goldilocks],
    c: &[Goldilocks],
    idx: &[u32],
    out: &[u64],
    m: usize,
    k: usize,
    n: usize,
    rng: &mut XorShift64,
) -> Option<(ShardMatmulLookupStatement, ShardMatmulLookupProof)> {
    let t = m * n;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), t);
    assert_eq!(idx.len(), t);
    assert_eq!(out.len(), t);

    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let (matmul_stmt, matmul_proof) = matmul_prove(&whir.matmul, a, b, c, m, k, n, &u, &v, &ch);

    // c is the lookup table (byte-identical committed values -> identical root).
    let c_u64: Vec<u64> = c.iter().map(|&x| to_u64(x)).collect();
    let (lookup_stmt, lookup_proof) = lookup_prove(&whir.lookup, idx, out, &c_u64)?;

    Some((
        ShardMatmulLookupStatement { matmul: matmul_stmt, lookup: lookup_stmt },
        ShardMatmulLookupProof { matmul: matmul_proof, lookup: lookup_proof, u, v, ch },
    ))
}

fn to_u64(x: Goldilocks) -> u64 {
    use zkie_core::common::field::PrimeField64;
    x.as_canonical_u64()
}

pub fn verify(whir: &ShardMatmulLookupWhir, stmt: &ShardMatmulLookupStatement, proof: &ShardMatmulLookupProof) -> bool {
    // same_poly: matmul output C == lookup table.
    if stmt.matmul.root_c != stmt.lookup.table.re {
        return false;
    }
    if !matmul_verify(&whir.matmul, &stmt.matmul, &proof.matmul, &proof.u, &proof.v, &proof.ch) {
        return false;
    }
    lookup_verify(&whir.lookup, &stmt.lookup, &proof.lookup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honest_roundtrip() {
        let (m, k, n) = (8usize, 8usize, 8usize);
        let whir = ShardMatmulLookupWhir::new(m, k, n, 32, 10).expect("valid dims");
        let mut rng = XorShift64::new(0xABC);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
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
        let c_u64: Vec<u64> = c.iter().map(|&x| to_u64(x)).collect();
        let t = m * n;
        let idx: Vec<u32> = (0..t).map(|i| (i as u32 * 7 + 3) % t as u32).collect();
        let out: Vec<u64> = idx.iter().map(|&i| c_u64[i as usize]).collect();
        let (stmt, proof) = prove(&whir, &a, &b, &c, &idx, &out, m, k, n, &mut rng).expect("prove");
        assert!(verify(&whir, &stmt, &proof));
    }

    #[test]
    fn mismatched_binding_rejected() {
        let (m, k, n) = (8usize, 8usize, 8usize);
        let whir = ShardMatmulLookupWhir::new(m, k, n, 32, 10).expect("valid dims");
        let mut rng = XorShift64::new(0xABC);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
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
        let c_u64: Vec<u64> = c.iter().map(|&x| to_u64(x)).collect();
        let t = m * n;
        let idx: Vec<u32> = (0..t).map(|i| (i as u32 * 7 + 3) % t as u32).collect();
        let out: Vec<u64> = idx.iter().map(|&i| c_u64[i as usize]).collect();
        let (stmt, proof) = prove(&whir, &a, &b, &c, &idx, &out, m, k, n, &mut rng).expect("prove");
        // Corrupt the matmul output root so it no longer equals the table root.
        let mut bad = stmt;
        let fake_c: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let (fake_root, _, _) = whir.matmul.c.commit(&fake_c);
        bad.matmul.root_c = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn malformed_dims_rejected() {
        assert!(ShardMatmulLookupWhir::new(8, 8, 8, 32, 10).is_some());
        assert!(ShardMatmulLookupWhir::new(3, 8, 8, 32, 10).is_none());
    }
}
