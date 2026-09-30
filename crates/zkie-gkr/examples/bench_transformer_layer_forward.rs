//! Measure the full transformer layer forward (witness) with the row-parallel
//! matmul, at GPT-2 scale. This is the bottleneck that the parallel matmul
//! targets.

use std::time::Instant;
use zkie_gkr::field::{Field, Goldilocks, XorShift64};
use zkie_gkr::fixed_point::from_i64;
use zkie_gkr::transformer_chain::transformer_layer_forward;

fn main() {
    let (m, d, ffn) = (512usize, 1024usize, 4096usize);
    let shift = 16u32;
    let mut rng = XorShift64::new(0xF00D);
    let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let wq: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let wk: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let wv: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let wo: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let bias: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let exp_table: Vec<Goldilocks> = (0..(1usize << 18)).map(|_| rng.field()).collect();
    let fc_w: Vec<Goldilocks> = (0..d * ffn).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let fc_b: Vec<Goldilocks> = (0..m * ffn).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let proj_w: Vec<Goldilocks> = (0..ffn * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let proj_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();
    let ln_w: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect();
    let ln_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let rsqrt_table: Vec<Goldilocks> = (0..(1usize << 18)).map(|_| rng.field()).collect();

    let t0 = Instant::now();
    let _out = transformer_layer_forward(
        &x, &wq, &wk, &wv, &wo, &bias, &exp_table, &fc_w, &fc_b, &proj_w, &proj_b,
        &gelu_table, &ln_w, &ln_b, &rsqrt_table, m, d, ffn, shift,
    );
    println!(
        "transformer-layer forward (parallel mm) m={} d={} ffn={} time={:.2}s",
        m,
        d,
        ffn,
        t0.elapsed().as_secs_f64()
    );
}
