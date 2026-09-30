//! Shard = a group of ops folded into one `g` (one sumcheck) with a boundary.
//!
//! Shard granularity (how many ops per shard) is a public parameter, *not*
//! hardcoded to "one transformer layer": a shard can be one op, one layer, a
//! few layers, or the whole model. A shard DAG chains shards in topological
//! order; each boundary tensor (the output of shard i == the input of shard i+1)
//! is claimed at two points and merged via `same_poly`, so the composed proof
//! is sound iff every shard is individually sound AND every boundary is
//! consistent. In the committed model the boundary is a single commitment
//! shared across the two shards (cross-shard binding).

use crate::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::layer::{compile, ElemOp};
use crate::layer_circuit::{prove_layer_circuit, verify_layer_circuit, LayerCircuitProof};
use crate::mle;
use crate::same_poly::{prove_same_poly, verify_same_poly, SamePolyProof};

/// A single shard's proof: the ops' constraints folded into one `g` at point `r`.
pub struct ShardProof {
    pub circuit: LayerCircuitProof,
    pub r: Vec<Goldilocks>,
}

/// A shard DAG proof: every shard, plus a `same_poly` binding for each boundary.
pub struct ShardDagProof {
    pub shards: Vec<ShardProof>,
    pub bindings: Vec<SamePolyProof>,
    pub boundaries: Vec<usize>,
}

pub fn prove_shard(
    tensors: &[&[Goldilocks]],
    ops: &[ElemOp],
    rng: &mut XorShift64,
) -> ShardProof {
    assert!(!tensors.is_empty(), "shard needs at least one tensor");
    let t = tensors[0].len().trailing_zeros() as usize;
    let r: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
    let circuit = prove_layer_circuit(tensors, &compile(ops), &r, rng);
    ShardProof { circuit, r }
}

pub fn verify_shard(p: &ShardProof, tensors: &[&[Goldilocks]], ops: &[ElemOp]) -> bool {
    verify_layer_circuit(&p.circuit, tensors, &compile(ops), &p.r)
}

/// Prove a chain of shards where `boundaries[i]` is the global tensor index
/// shared between shard `i` and shard `i+1`. Granularity is the caller's choice:
/// the shards are whatever op-groups the caller decided to fold.
pub fn prove_shard_dag(
    tensors: &[&[Goldilocks]],
    shards: &[Vec<ElemOp>],
    boundaries: &[usize],
    rng: &mut XorShift64,
) -> ShardDagProof {
    assert_eq!(boundaries.len(), shards.len() - 1, "one boundary per shard junction");
    let shard_proofs: Vec<ShardProof> =
        shards.iter().map(|ops| prove_shard(tensors, ops, rng)).collect();
    let mut bindings = Vec::new();
    for (i, &b) in boundaries.iter().enumerate() {
        let t = tensors[b];
        let c0 = (shard_proofs[i].r.clone(), mle::eval(t, &shard_proofs[i].r));
        let c1 = (shard_proofs[i + 1].r.clone(), mle::eval(t, &shard_proofs[i + 1].r));
        bindings.push(prove_same_poly(t, &[c0, c1], rng));
    }
    ShardDagProof {
        shards: shard_proofs,
        bindings,
        boundaries: boundaries.to_vec(),
    }
}

pub fn verify_shard_dag(
    p: &ShardDagProof,
    tensors: &[&[Goldilocks]],
    shards: &[Vec<ElemOp>],
) -> bool {
    for (sp, ops) in p.shards.iter().zip(shards) {
        if !verify_shard(sp, tensors, ops) {
            return false;
        }
    }
    for (i, &b) in p.boundaries.iter().enumerate() {
        let t = tensors[b];
        let c0 = (p.shards[i].r.clone(), mle::eval(t, &p.shards[i].r));
        let c1 = (p.shards[i + 1].r.clone(), mle::eval(t, &p.shards[i + 1].r));
        if verify_same_poly(&p.bindings[i], t, &[c0, c1]).is_none() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_dag_roundtrip() {
        let mut rng = XorShift64::new(0x5A5A);
        let n = 1usize << 6;
        let x: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let w: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let y: Vec<Goldilocks> = (0..n).map(|i| x[i] + w[i]).collect();
        let z: Vec<Goldilocks> = (0..n).map(|i| y[i] * w[i]).collect();
        let tensors: Vec<&[Goldilocks]> = vec![&x, &w, &y, &z];

        // shard 0: y = x + w; shard 1: z = y * w. Boundary = y (index 2).
        let shard0 = vec![ElemOp::Add { a: 0, b: 1, c: 2 }];
        let shard1 = vec![ElemOp::Mul { a: 2, b: 1, c: 3 }];
        let proof = prove_shard_dag(&tensors, &[shard0.clone(), shard1.clone()], &[2], &mut rng);
        assert!(verify_shard_dag(&proof, &tensors, &[shard0, shard1]));

        // Tamper with the output of the last shard; the DAG must reject.
        let bad_z: Vec<Goldilocks> = z.iter().map(|&v| v + Goldilocks::ONE).collect();
        let bad_tensors: Vec<&[Goldilocks]> = vec![&x, &w, &y, &bad_z];
        let s0 = vec![ElemOp::Add { a: 0, b: 1, c: 2 }];
        let s1 = vec![ElemOp::Mul { a: 2, b: 1, c: 3 }];
        assert!(!verify_shard_dag(&proof, &bad_tensors, &[s0, s1]));
    }

    #[test]
    fn single_shard_is_op_granularity() {
        // Granularity is a parameter: one shard per op is the degenerate
        // "op-granularity" case; the same code proves a whole group at once.
        let mut rng = XorShift64::new(0x6B6B);
        let n = 1usize << 6;
        let x: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let w: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let y: Vec<Goldilocks> = (0..n).map(|i| x[i] + w[i]).collect();
        let z: Vec<Goldilocks> = (0..n).map(|i| y[i] * w[i]).collect();
        let tensors: Vec<&[Goldilocks]> = vec![&x, &w, &y, &z];

        // whole graph in one shard (coarse granularity)
        let coarse = vec![
            ElemOp::Add { a: 0, b: 1, c: 2 },
            ElemOp::Mul { a: 2, b: 1, c: 3 },
        ];
        let p = prove_shard(&tensors, &coarse, &mut rng);
        assert!(verify_shard(&p, &tensors, &coarse));
    }
}
