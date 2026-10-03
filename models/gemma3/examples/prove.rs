//! End-to-end Gemma 3 270M prove + verify, printing the final argmax and its
//! match count against the ground-truth reference.

use std::fs;

use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i32, to_i32, to_i64};
use zkie_models_gemma3::{build_gemma3, VOCAB};
use zkie_ops::compose::{prove_shard_dag, verify_shard_dag, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn main() {
    let m = std::env::var("GEMMA_SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
    let dir = "models/gemma3/weights";
    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, logits) = build_gemma3(&mut store, &mut ops, dir, m, 16);

    let mut rng = XorShift64::new(0xC0FFEE);
    let t0 = std::time::Instant::now();
    let proof = prove_shard_dag(&mut store, &ops, 53, &mut rng);
    let prove_t = t0.elapsed();
    let t1 = std::time::Instant::now();
    assert!(verify_shard_dag(&mut store, &ops, 53, &proof));
    let verify_t = t1.elapsed();

    let gt = load_i32(&format!("{dir}/gt_argmax_i32.bin"));
    let lg = store.get(logits);
    let mut matches = 0;
    for i in 0..m {
        let mut best = 0usize;
        let mut best_v = lg[i * VOCAB];
        for j in 1..VOCAB {
            let v = lg[i * VOCAB + j];
            if to_i64(v) > to_i64(best_v) {
                best_v = v;
                best = j;
            }
        }
        if best as i32 == to_i32(gt[i]) {
            matches += 1;
        }
    }
    println!("prove {prove_t:?} verify {verify_t:?} argmax {matches}/{m}");
}
