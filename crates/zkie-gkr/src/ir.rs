//! Minimal program IR + two-phase executor.
//!
//! Ops are declared imperatively; the executor records each tensor into a
//! BatchBuilder (by size/group), then the prove pass walks the same op list and
//! opens each tensor by its (size, group, index) to run the batch proof.

use std::collections::HashMap;

use crate::committed::{prove_add_batch, prove_matmul_batch, BatchCtx};
use crate::field::{Goldilocks, XorShift64};
use crate::ops::{add_vec, dense_m, BatchBuilder};
use crate::whir::Whir;

/// A committed proof step. Composite ops are added later.
pub enum Op {
    MatMul { a: usize, b: usize, c: usize, m: usize, k: usize, n: usize },
    Add { a: usize, b: usize, c: usize },
}

pub struct Exec {
    plain: Vec<Vec<Goldilocks>>,
    meta: Vec<(usize, usize, usize)>,
    ops: Vec<Op>,
    bb: BatchBuilder,
}

impl Exec {
    pub fn new() -> Self {
        Exec { plain: Vec::new(), meta: Vec::new(), ops: Vec::new(), bb: BatchBuilder::new() }
    }

    pub fn input(&mut self, plain: Vec<Goldilocks>, group: usize) -> usize {
        let (size, group, index) = self.bb.push_group(plain.clone(), group);
        let id = self.plain.len();
        self.plain.push(plain);
        self.meta.push((size, group, index));
        id
    }

    pub fn matmul(&mut self, a: usize, b: usize, m: usize, k: usize, n: usize) -> usize {
        let c_plain = dense_m(&self.plain[a], &self.plain[b], m, k, n);
        let c = self.input(c_plain, 0);
        self.ops.push(Op::MatMul { a, b, c, m, k, n });
        c
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let c_plain = add_vec(&self.plain[a], &self.plain[b]);
        let c = self.input(c_plain, 0);
        self.ops.push(Op::Add { a, b, c });
        c
    }

    pub fn prove(&self, whir: &Whir, rng: &mut XorShift64) {
        let batches: HashMap<(usize, usize), BatchCtx> = self.bb.commit(whir);
        for op in &self.ops {
            match op {
                Op::MatMul { a, b, c, m, k, n } => {
                    let (a_sz, a_g, a_i) = self.meta[*a];
                    let (b_sz, b_g, b_i) = self.meta[*b];
                    let (c_sz, c_g, c_i) = self.meta[*c];
                    let a_b = &batches[&(a_sz, a_g)];
                    let b_b = &batches[&(b_sz, b_g)];
                    let c_b = &batches[&(c_sz, c_g)];
                    assert!(prove_matmul_batch(
                        a_b, a_i, b_b, b_i, c_b, c_i,
                        &self.plain[*a], &self.plain[*b], &self.plain[*c], *m, *k, *n, rng,
                    ));
                }
                Op::Add { a, b, c } => {
                    let (a_sz, a_g, a_i) = self.meta[*a];
                    let (b_sz, b_g, b_i) = self.meta[*b];
                    let (c_sz, c_g, c_i) = self.meta[*c];
                    let a_b = &batches[&(a_sz, a_g)];
                    let b_b = &batches[&(b_sz, b_g)];
                    let c_b = &batches[&(c_sz, c_g)];
                    assert!(prove_add_batch(
                        a_b, a_i, b_b, b_i, c_b, c_i, self.plain[*c].len(), rng,
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Goldilocks, PrimeCharacteristicRing};
    use crate::fixed_point::from_i32;

    #[test]
    fn ir_two_phase_matmul_add() {
        let mut rng = XorShift64::new(0x4444);
        let (m, k, n) = (4usize, 16usize, 16usize);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| from_i32((rng.next_u64() % 1000) as i32)).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| from_i32((rng.next_u64() % 1000) as i32)).collect();
        let d: Vec<Goldilocks> = (0..m * n).map(|_| from_i32((rng.next_u64() % 1000) as i32)).collect();

        let mut ex = Exec::new();
        let a_id = ex.input(a, 1);
        let b_id = ex.input(b, 1);
        let c_id = ex.matmul(a_id, b_id, m, k, n);
        let d_id = ex.input(d, 1);
        let _e_id = ex.add(c_id, d_id);

        let whir = Whir::new_testing(6);
        ex.prove(&whir, &mut rng);
    }
}
