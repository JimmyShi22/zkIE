//! Two-phase proving demo: forward pass collects raw tensors into a
//! BatchBuilder, then a single commit pass groups them by size, then the
//! proving phase opens tensors by (batch, table index).

use zkie_gkr::committed::{affine_raw, prove_affine_batch, prove_matmul_batch};
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::{from_i32, to_i32, to_i64};
use zkie_gkr::ops::{dense_m, BatchBuilder};
use zkie_gkr::whir::Whir;

fn main() {
    let mut rng = XorShift64::new(0x3333);
    let (m, k, n) = (4usize, 16usize, 16usize);

    // --- phase 1: forward (compute raw only) ---
    let a: Vec<Goldilocks> = (0..m * k).map(|_| from_i32((rng.next_u64() % 1000) as i32)).collect();
    let b: Vec<Goldilocks> = (0..k * n).map(|_| from_i32((rng.next_u64() % 1000) as i32)).collect();
    let c = dense_m(&a, &b, m, k, n);
    let bias = vec![from_i32(0); m * n];
    let out = affine_raw(&c, &bias, 16, false);

    // affine round bits (zero bias, shift 16) for prove_affine_batch.
    let mut zbits: Vec<Vec<Goldilocks>> = vec![vec![Goldilocks::ZERO; m * n]; 16];
    for i in 0..m * n {
        let in_i = to_i64(c[i]);
        let out_i = to_i32(out[i]) as i64;
        let rem = in_i - out_i * (1i64 << 16);
        let rem_off = rem + (1i64 << 15);
        for (j, bj) in zbits.iter_mut().enumerate() {
            bj[i] = from_i32(((rem_off >> j) & 1) as i32);
        }
    }

    // --- phase 2: batch commit, grouped by size ---
    let mut bb = BatchBuilder::new();
    let (a_sz, a_i) = bb.push(a.clone());
    let (b_sz, b_i) = bb.push(b.clone());
    let (c_sz, c_i) = bb.push(c.clone());
    let (o_sz, o_i) = bb.push(out.clone());
    let whir = Whir::new_testing(8);
    let batches = bb.commit(&whir);

    let mut zbb = BatchBuilder::new();
    for z in &zbits {
        zbb.push(z.clone());
    }
    let zbatches = zbb.commit(&whir);

    // --- phase 3: prove ---
    let a_b = &batches[&a_sz];
    let b_b = &batches[&b_sz];
    let c_b = &batches[&c_sz];
    let o_b = &batches[&o_sz];
    assert!(prove_matmul_batch(a_b, a_i, b_b, b_i, c_b, c_i, &a, &b, &c, m, k, n, &mut rng));

    let zb = &zbatches[&(m * n)];
    assert!(prove_affine_batch(c_b, c_i, &c, o_b, o_i, &out, zb, &bias, 16, &mut rng));

    println!("two-phase matmul + affine proved over batch commitments");
}
