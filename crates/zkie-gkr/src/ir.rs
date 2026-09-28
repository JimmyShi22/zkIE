//! Minimal program IR + two-phase executor.
//!
//! Ops are declared imperatively; the executor records each tensor into a
//! BatchBuilder (by size/group), then the prove pass walks the same op list and
//! opens each tensor by its (size, group, index) to run the batch proof.

use std::collections::HashMap;

use crate::committed::{affine_raw, prove_add_batch, prove_affine_batch, prove_matmul_batch, BatchCtx};
use crate::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i32, from_i64, to_i32, to_i64};
use crate::ops::{add_vec, dense_m, BatchBuilder};
use crate::whir::Whir;

/// A committed proof step. Composite ops are added later.
pub enum Op {
    MatMul { a: usize, b: usize, c: usize, m: usize, k: usize, n: usize },
    Add { a: usize, b: usize, c: usize },
    Affine { x: usize, out: usize, bits_group: (usize, usize), bias: Vec<Goldilocks>, shift: u32 },
}

pub struct Exec {
    plain: Vec<Vec<Goldilocks>>,
    meta: Vec<(usize, usize, usize)>,
    ops: Vec<Op>,
    bb: BatchBuilder,
    next_group: usize,
}

impl Exec {
    pub fn new() -> Self {
        Exec { plain: Vec::new(), meta: Vec::new(), ops: Vec::new(), bb: BatchBuilder::new(), next_group: 2 }
    }

    fn fresh_group(&mut self) -> usize {
        let g = self.next_group;
        self.next_group += 1;
        g
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

    pub fn affine(&mut self, x: usize, bias: Vec<Goldilocks>, shift: u32) -> usize {
        let out_plain = affine_raw(&self.plain[x], &bias, shift, false);
        let n = bias.len();
        let g = self.fresh_group();
        let mut bit_columns: Vec<Vec<Goldilocks>> = vec![vec![Goldilocks::ZERO; n]; shift as usize];
        for i in 0..n {
            let in_i = to_i64(self.plain[x][i]);
            let out_i = to_i32(out_plain[i]) as i64;
            let b_i = to_i32(bias[i]) as i64;
            let rem = in_i - (out_i - b_i) * (1i64 << shift);
            let rem_off = rem + (1i64 << (shift - 1));
            for (j, bj) in bit_columns.iter_mut().enumerate() {
                bj[i] = from_i32(((rem_off >> j) & 1) as i32);
            }
        }
        for bc in &bit_columns {
            self.input(bc.clone(), g);
        }
        let out = self.input(out_plain, 0);
        self.ops.push(Op::Affine { x, out, bits_group: (n, g), bias, shift });
        out
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
                Op::Affine { x, out, bits_group, bias, shift } => {
                    let (x_sz, x_g, x_i) = self.meta[*x];
                    let (o_sz, o_g, o_i) = self.meta[*out];
                    let x_b = &batches[&(x_sz, x_g)];
                    let o_b = &batches[&(o_sz, o_g)];
                    let bits_b = &batches[&bits_group];
                    assert!(prove_affine_batch(
                        x_b, x_i, &self.plain[*x], o_b, o_i, &self.plain[*out], bits_b, bias, *shift, rng,
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
    fn ir_two_phase_affine() {
        let mut rng = XorShift64::new(0x5555);
        let n = 64usize;
        let x: Vec<Goldilocks> = (0..n).map(|_| from_i64((rng.next_u64() % (1u64 << 32)) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..n).map(|_| from_i32((rng.next_u64() % 1000) as i32)).collect();
        let mut ex = Exec::new();
        let x_id = ex.input(x, 1);
        let _out = ex.affine(x_id, bias, 16);
        let whir = Whir::new_testing(6);
        ex.prove(&whir, &mut rng);
    }

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
