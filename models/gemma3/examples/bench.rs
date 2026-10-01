//! Prove the full Gemma 3 270M op graph with the shard-DAG composer at several
//! `ops_per_shard` granularities, verifying cross-shard binding and measuring
//! prove/verify wall time + peak RSS.

use std::fs;

use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i32, to_i32, to_i64};
use zkie_models_gemma3::{build_gemma3, H_PAD, VOCAB};
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
    let m = std::env::var("GEMMA_SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
    let shift = 16u32;
    let dir = "models/gemma3/weights";

    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, logits) = build_gemma3(&mut store, &mut ops, dir, m, shift);
    let total = ops.len();
    println!("seq={m} total ops: {total}");

    let gt = load_i32(&format!("{dir}/gt_argmax_i32.bin"));

    let mut best_time = f64::MAX;
    let mut best_pps = 0usize;
    for &pps in &[53usize, 106, 212, 424, 848] {
        let mut rng = XorShift64::new(0xBEEF);
        let t0 = std::time::Instant::now();
        let proof = prove_shard_dag(&mut store, &ops, pps, &mut rng);
        let prove_t = t0.elapsed();
        let t1 = std::time::Instant::now();
        assert!(verify_shard_dag(&store, &ops, pps, &proof), "sharded proof failed");
        let verify_t = t1.elapsed();

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
        let total_t = (prove_t + verify_t).as_secs_f64();
        println!(
            "pps={pps:5} shards={:3} prove={prove_t:?} verify={verify_t:?} total={total_t:.2}s argmax={matches}/{m} rss={}kB",
            proof.shards.len(),
            peak_rss_kb()
        );
        if total_t < best_time {
            best_time = total_t;
            best_pps = pps;
        }
    }
    println!("best ops_per_shard: {best_pps} ({best_time:.2}s total)");
}
