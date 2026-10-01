//! Time the forward matmul (witness generation) for the GPT-2 lm_head, which is
//! the single largest matmul (m=512, k=1024, n=65536). This isolates the
//! `mm_par` cost from the GKR sumcheck cost to locate the prove-time bottleneck.

use std::fs;
use std::time::Instant;

use zkie_gkr::field::Goldilocks;
use zkie_gkr::fixed_point::from_i32;
use zkie_gkr::par::{mm_par, mm_par_fixed};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn main() {
    let (m, k, n) = (512usize, 1024usize, 65536usize);
    let lm_head_w = load_i32("models/gpt2_stack/lm_head_w_i32.bin");
    assert_eq!(lm_head_w.len(), k * n);

    // A synthetic activation (h_final) of the right shape.
    let a: Vec<Goldilocks> = (0..m * k).map(|i| from_i32((i % 997) as i32 - 500)).collect();

    for run in 0..3 {
        let t = Instant::now();
        let c = mm_par(&a, &lm_head_w, m, k, n, 64);
        let dt = t.elapsed();
        println!("lm_head mm_par (field) run {}: {:?}", run, dt);
    }
    for run in 0..3 {
        let t = Instant::now();
        let c = mm_par_fixed(&a, &lm_head_w, m, k, n);
        let dt = t.elapsed();
        println!("lm_head mm_par_fixed (i64) run {}: {:?} ({} elements)", run, dt, c.len());
    }
}
