//! End-to-end DeepSeek-V2-Lite prove (prove + argmax sanity). Verify is optional
//! (it clones the full store and needs ~2x memory).

use std::fs;

use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i32, to_i32, to_i64};
use zkie_models_deepseek_v2_lite::{build_deepseek, VOCAB, VOCAB_PAD};
use zkie_ops::compose::{prove_shard_dag, verify_shard_dag, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn peak_rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            return v.trim().trim_end_matches(" kB").parse().unwrap_or(0);
        }
    }
    0
}

fn main() {
    let m = 16usize;
    let dir = "models/deepseek-v2-lite/weights";
    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, logits) = build_deepseek(&mut store, &mut ops, dir, m, 16);
    println!("total ops: {}", ops.len());

    let gt = load_i32(&format!("{dir}/gt_argmax_i32.bin"));
    let mut rng = XorShift64::new(0xD5EED);
    let t0 = std::time::Instant::now();
    let proof = prove_shard_dag(&mut store, &ops, 718, &mut rng);
    let prove_t = t0.elapsed();

    let lg = store.get(logits);
    let mut matches = 0;
    for i in 0..m {
        let mut best = 0usize;
        let mut best_v = lg[i * VOCAB_PAD];
        for j in 1..VOCAB {
            let v = lg[i * VOCAB_PAD + j];
            if to_i64(v) > to_i64(best_v) {
                best_v = v;
                best = j;
            }
        }
        if best as i32 == to_i32(gt[i]) {
            matches += 1;
        }
    }
    let t1 = std::time::Instant::now();
    assert!(verify_shard_dag(&store, &ops, 718, &proof), "verify failed");
    let verify_t = t1.elapsed();
    println!(
        "prove {prove_t:?} verify {verify_t:?} shards={} argmax={matches}/{m} rss={}kB",
        proof.shards.len(),
        peak_rss_kb()
    );
}
