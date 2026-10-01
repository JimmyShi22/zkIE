//! Prove the full DeepSeek-V2-Lite graph at per-layer granularity and report
//! prove time + peak RSS.

use zkie_core::common::field::XorShift64;
use zkie_models_deepseek_v2_lite::build_deepseek;
use zkie_ops::compose::{prove_shard_dag, Store};

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
    build_deepseek(&mut store, &mut ops, dir, m, 16);
    let total = ops.len();
    println!("seq={m} total ops: {total}");

    let mut rng = XorShift64::new(0xD5EED);
    let t0 = std::time::Instant::now();
    let proof = prove_shard_dag(&mut store, &ops, 718, &mut rng);
    let prove_t = t0.elapsed();
    println!("prove {prove_t:?} shards={} rss={}kB", proof.shards.len(), peak_rss_kb());
}
