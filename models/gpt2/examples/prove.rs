//! Prove the full GPT-2 512 (12 layers, 12 heads, real weights) end to end
//! using the reusable `zkie_engine::models::gpt2` builder.

use zkie_core::common::field::XorShift64;
use zkie_core::common::fixed_point::{to_i32, to_i64};
use zkie_models_gpt2::{build_gpt2, load_i32};
use zkie_ops::compose::{prove_shard_dag, verify_shard_dag, Store};

fn main() {
    let (m, shift) = (512usize, 16u32);
    let dir = "models/gpt2/weights";

    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, logits) = build_gpt2(&mut store, &mut ops, dir, m, shift);

    let ops_per_shard = 176; // 13 shards (one transformer layer each)
    let mut rng = XorShift64::new(0xBEEF);
    let t0 = std::time::Instant::now();
    let proof = prove_shard_dag(&mut store, &ops, ops_per_shard, &mut rng);
    let prove_t = t0.elapsed();
    let t1 = std::time::Instant::now();
    assert!(verify_shard_dag(&mut store, &ops, ops_per_shard, &proof));
    let verify_t = t1.elapsed();
    println!("prove {:?}, verify {:?}", prove_t, verify_t);

    let gt = load_i32(&format!("{dir}/gt_argmax_512_i32.bin"));
    let logits_v = store.get(logits);
    let mut matches = 0;
    for i in 0..m {
        let mut best = 0usize;
        let mut best_v = logits_v[i * 65536];
        for j in 1..50257 {
            let v = logits_v[i * 65536 + j];
            if to_i64(v) > to_i64(best_v) {
                best_v = v;
                best = j;
            }
        }
        if best as i32 == to_i32(gt[i]) {
            matches += 1;
        }
    }
    println!("argmax matches: {}/{}", matches, m);
}
