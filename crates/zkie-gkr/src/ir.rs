//! Minimal program IR + two-phase executor.
//!
//! Ops are declared imperatively; the executor records each tensor into a
//! BatchBuilder (by size/group), then the prove pass walks the same op list and
//! opens each tensor by its (size, group, index) to run the batch proof.

use std::collections::HashMap;

use crate::committed::{affine_raw, layer_norm_raw, prove_add_batch, prove_affine_batch, prove_layer_norm_batch, prove_matmul_batch, prove_relu_batch, prove_rms_norm_batch, prove_scale_batch, rms_norm_raw, scale_raw, BatchCtx};
use crate::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i32, from_i64, to_i32, to_i64};
use crate::ops::{add_vec, dense_m, BatchBuilder};
use crate::whir::Whir;

/// A committed proof step. Composite ops are added later.
fn round_half_up(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
}

pub enum Op {
    MatMul { a: usize, b: usize, c: usize, m: usize, k: usize, n: usize },
    Add { a: usize, b: usize, c: usize },
    Affine { x: usize, out: usize, bits_group: (usize, usize), bias: Vec<Goldilocks>, shift: u32 },
    Scale { x: usize, out: usize, bits_group: (usize, usize), scale: i64, bias: Vec<Goldilocks> },
    Relu { x: usize, out: usize, mid_group: (usize, usize), bits_group: (usize, usize), zero_bits_group: (usize, usize), bias: Vec<Goldilocks> },
    RmsNorm { x: usize, raw: usize, weight: Vec<Goldilocks>, n_real: usize },
    LayerNorm { x: usize, raw: usize, weight: Vec<Goldilocks>, n_real: usize },
}

pub struct Exec {
    plain: Vec<Vec<Goldilocks>>,
    meta: Vec<(usize, usize, usize)>,
    ops: Vec<Op>,
    bb: BatchBuilder,
    next_group: usize,
    rsqrt_table: Option<Vec<Goldilocks>>,
}

impl Exec {
    pub fn new() -> Self {
        Exec { plain: Vec::new(), meta: Vec::new(), ops: Vec::new(), bb: BatchBuilder::new(), next_group: 2, rsqrt_table: None }
    }

    fn fresh_group(&mut self) -> usize {
        let g = self.next_group;
        self.next_group += 1;
        g
    }

    pub fn set_rsqrt(&mut self, table: Vec<Goldilocks>) {
        self.rsqrt_table = Some(table);
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

    pub fn scale(&mut self, x: usize, scale: i64, bias: Vec<Goldilocks>) -> usize {
        let out_plain = scale_raw(&self.plain[x], scale, &bias);
        let n = bias.len();
        let g = self.fresh_group();
        let mut bit_columns: Vec<Vec<Goldilocks>> = vec![vec![Goldilocks::ZERO; n]; 16];
        for i in 0..n {
            let in_i = to_i64(self.plain[x][i]);
            let out_i = to_i32(out_plain[i]) as i64;
            let b_i = to_i32(bias[i]) as i64;
            let rem = in_i * scale - (out_i - b_i) * (1i64 << 16);
            let rem_off = rem + (1i64 << 15);
            for (j, bj) in bit_columns.iter_mut().enumerate() {
                bj[i] = from_i32(((rem_off >> j) & 1) as i32);
            }
        }
        for bc in &bit_columns {
            self.input(bc.clone(), g);
        }
        let out = self.input(out_plain, 0);
        self.ops.push(Op::Scale { x, out, bits_group: (n, g), scale, bias });
        out
    }

    pub fn relu(&mut self, x: usize, bias: Vec<Goldilocks>) -> usize {
        let n = bias.len();
        let out_plain = affine_raw(&self.plain[x], &bias, 16, true);

        let rescaled: Vec<Goldilocks> = self.plain[x]
            .iter()
            .map(|&iv| from_i32(round_half_up(to_i64(iv), 1i64 << 16) as i32))
            .collect();
        let pre: Vec<Goldilocks> = rescaled.iter().zip(&bias).map(|(&r, &b)| r + b).collect();
        let abs_pre: Vec<Goldilocks> = pre.iter().map(|&p| from_i32((to_i32(p) as i64).abs() as i32)).collect();

        let mut bits: Vec<Vec<Goldilocks>> = vec![vec![Goldilocks::ZERO; n]; 31];
        for i in 0..n {
            let a = (to_i32(pre[i]) as i64).abs();
            for (j, bj) in bits.iter_mut().enumerate() {
                bj[i] = from_i32(((a >> j) & 1) as i32);
            }
        }
        let mut zbits: Vec<Vec<Goldilocks>> = vec![vec![Goldilocks::ZERO; n]; 16];
        for i in 0..n {
            let in_i = to_i64(self.plain[x][i]);
            let out_i = to_i32(rescaled[i]) as i64;
            let rem = in_i - out_i * (1i64 << 16);
            let rem_off = rem + (1i64 << 15);
            for (j, bj) in zbits.iter_mut().enumerate() {
                bj[i] = from_i32(((rem_off >> j) & 1) as i32);
            }
        }

        let g_mid = self.fresh_group();
        self.input(rescaled, g_mid);
        self.input(pre, g_mid);
        self.input(abs_pre, g_mid);

        let g_bits = self.fresh_group();
        for bc in &bits {
            self.input(bc.clone(), g_bits);
        }
        let g_zb = self.fresh_group();
        for bc in &zbits {
            self.input(bc.clone(), g_zb);
        }

        let out = self.input(out_plain, 0);
        self.ops.push(Op::Relu {
            x,
            out,
            mid_group: (n, g_mid),
            bits_group: (n, g_bits),
            zero_bits_group: (n, g_zb),
            bias,
        });
        out
    }

    pub fn rms_norm(&mut self, x: usize, weight: Vec<Goldilocks>, n_real: usize) -> usize {
        let rsqrt = self.rsqrt_table.as_ref().expect("set_rsqrt first");
        let (raw, _, _) = rms_norm_raw(&self.plain[x], &weight, n_real, rsqrt);
        let raw_id = self.input(raw, 0);
        self.ops.push(Op::RmsNorm { x, raw: raw_id, weight, n_real });
        raw_id
    }

    pub fn layer_norm(&mut self, x: usize, weight: Vec<Goldilocks>, n_real: usize) -> usize {
        let rsqrt = self.rsqrt_table.as_ref().expect("set_rsqrt first");
        let (raw, _, _, _) = layer_norm_raw(&self.plain[x], &weight, n_real, rsqrt);
        let raw_id = self.input(raw, 0);
        self.ops.push(Op::LayerNorm { x, raw: raw_id, weight, n_real });
        raw_id
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
                Op::Scale { x, out, bits_group, scale, bias } => {
                    let (x_sz, x_g, x_i) = self.meta[*x];
                    let (o_sz, o_g, o_i) = self.meta[*out];
                    let x_b = &batches[&(x_sz, x_g)];
                    let o_b = &batches[&(o_sz, o_g)];
                    let bits_b = &batches[&bits_group];
                    assert!(prove_scale_batch(
                        x_b, x_i, &self.plain[*x], o_b, o_i, &self.plain[*out], bits_b, *scale, bias, rng,
                    ));
                }
                Op::Relu { x, out, mid_group, bits_group, zero_bits_group, bias } => {
                    let (x_sz, x_g, x_i) = self.meta[*x];
                    let (o_sz, o_g, o_i) = self.meta[*out];
                    let x_b = &batches[&(x_sz, x_g)];
                    let o_b = &batches[&(o_sz, o_g)];
                    let mid_b = &batches[&mid_group];
                    let bits_b = &batches[&bits_group];
                    let zb_b = &batches[&zero_bits_group];
                    assert!(prove_relu_batch(
                        x_b, x_i, &self.plain[*x], o_b, o_i, mid_b, bits_b, zb_b, bias, rng,
                    ));
                }
                Op::RmsNorm { x, raw, weight, n_real } => {
                    let rsqrt = self.rsqrt_table.as_ref().unwrap();
                    let alpha = rng.field();
                    let beta = rng.field();
                    let (x_sz, x_g, x_i) = self.meta[*x];
                    let (r_sz, r_g, r_i) = self.meta[*raw];
                    let x_b = &batches[&(x_sz, x_g)];
                    let r_b = &batches[&(r_sz, r_g)];
                    assert!(prove_rms_norm_batch(
                        x_b, x_i, &self.plain[*x], r_b, r_i, &self.plain[*raw], weight, *n_real, rsqrt, alpha, beta, rng,
                    ));
                }
                Op::LayerNorm { x, raw, weight, n_real } => {
                    let rsqrt = self.rsqrt_table.as_ref().unwrap();
                    let alpha = rng.field();
                    let beta = rng.field();
                    let (x_sz, x_g, x_i) = self.meta[*x];
                    let (r_sz, r_g, r_i) = self.meta[*raw];
                    let x_b = &batches[&(x_sz, x_g)];
                    let r_b = &batches[&(r_sz, r_g)];
                    assert!(prove_layer_norm_batch(
                        x_b, x_i, &self.plain[*x], r_b, r_i, &self.plain[*raw], weight, *n_real, rsqrt, alpha, beta, rng,
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
    fn ir_two_phase_scale_relu() {
        let mut rng = XorShift64::new(0x6666);
        let n = 64usize;
        let x: Vec<Goldilocks> = (0..n).map(|_| from_i32((rng.next_u64() % 65536) as i32)).collect();
        let scale = (rng.next_u64() % 1000) as i64 + 1;
        let bias: Vec<Goldilocks> = (0..n).map(|_| from_i32((rng.next_u64() % 1000) as i32)).collect();

        let mut ex = Exec::new();
        let x_id = ex.input(x, 1);
        let _s = ex.scale(x_id, scale, bias);

        let relu_in: Vec<Goldilocks> = (0..n).map(|_| from_i64((rng.next_u64() % (1u64 << 32)) as i64)).collect();
        let relu_bias: Vec<Goldilocks> = (0..n).map(|_| from_i32((rng.next_u64() % 1000) as i32 - 500)).collect();
        let r_id = ex.input(relu_in, 1);
        let _r = ex.relu(r_id, relu_bias);

        let whir = Whir::new_testing(6);
        ex.prove(&whir, &mut rng);
    }

    #[test]
    fn ir_two_phase_norm() {
        let mut rng = XorShift64::new(0x7777);
        let n = 64usize;
        let n_real = n;
        let x: Vec<Goldilocks> = (0..n).map(|_| from_i32((rng.next_u64() % 65536) as i32)).collect();
        let weight: Vec<Goldilocks> = (0..n).map(|_| from_i32((rng.next_u64() % 65536) as i32)).collect();
        let table_size = 1usize << 17;
        let rsqrt_table: Vec<Goldilocks> = (0..table_size)
            .map(|j| {
                let s = j as f64 / 16384.0 + 1e-6;
                from_i32((1.0 / s.sqrt() * 65536.0).round() as i32)
            })
            .collect();

        let mut ex = Exec::new();
        ex.set_rsqrt(rsqrt_table);
        let x_id = ex.input(x, 1);
        let _r = ex.rms_norm(x_id, weight.clone(), n_real);
        let _l = ex.layer_norm(x_id, weight, n_real);

        let whir = Whir::new_testing(6);
        ex.prove(&whir, &mut rng);
    }

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
