//! End-to-end GPT-2 prove (prove + argmax sanity). Verify is optional
//! (it clones the full store and needs ~2x memory).


use zkie_core::common::fixed_point::{to_i32, to_i64};
use zkie_models_gpt2::{build_gpt2, load_i32};
use zkie_ops::compose::{prove_shard_dag, verify_shard_dag, Store};

const VOCAB: usize = 50257;
const VOCAB_PAD: usize = 65536;

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
    let m: usize = std::env::var("M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let dir = "models/gpt2/weights";
    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, logits, _tokens, _t) = build_gpt2(&mut store, &mut ops, dir, m, 16);

    let gt = load_i32(&format!("{dir}/gt_argmax_{}i32.bin", if m <= 16 { "" } else { "512_" }));
    let t0 = std::time::Instant::now();
    let proof = prove_shard_dag(&mut store, &ops, 176);
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
    println!(
        "prove {prove_t:?} shards={} argmax={matches}/{m} rss={}kB",
        proof.shards.len(),
        peak_rss_kb()
    );
    use std::io::Write;
    let _ = std::io::stdout().flush();
    if std::env::var("VERIFY").is_ok() {
        let t1 = std::time::Instant::now();
        assert!(
            verify_shard_dag(&store.owned_slices(), &ops, 176, &proof),
            "verify failed"
        );
        println!("verify {:?}", t1.elapsed());
        let _ = std::io::stdout().flush();
    }
}
