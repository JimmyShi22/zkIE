//! Committed store-free proof for a single layer-0 block-quantized matmul.
//!
//! This is the smallest end-to-end check that exercises the quantized weight
//! codec, `commit_model_weights`, `prove_committed_shard_dag`, and the root-only
//! verifier on a real DeepSeek-V4.1-Flash weight.

use zkie_core::common::field::{XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;
use zkie_models_deepseek_v41_flash::{build_quantized_matmul, open_shard, push_quantized_fp8};
use zkie_ops::compose::{
    commit_model_weights, prove_committed_shard_dag_with_ranges,
    verify_committed_shard_dag_with_ranges, Store,
};

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
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let file = open_shard(&path);
    let m = 1usize;
    let k = 4096usize;
    let n = 8192usize;

    let mut store = Store::new();
    let x = store.push(vec![from_i64(1); m * k]);
    let w = push_quantized_fp8(
        &mut store,
        file,
        "layers.0.attn.wo_a.weight",
        "layers.0.attn.wo_a.scale",
    );
    let mut ops = Vec::new();
    build_quantized_matmul(&mut store, &mut ops, x, w, m, k, n);

    let tokens_t = store.push_idx(vec![]);
    let argmax_t = store.push_idx(vec![]);
    let whir = Whir::new_testing(10);
    let mut rng = XorShift64::new(0xBEEF);

    let t0 = std::time::Instant::now();
    let cached = commit_model_weights(
        &store,
        &ops,
        "/data/jimmyshi/zkie-cache/deepseek-v41-flash-layer0",
    );
    let commit_t = t0.elapsed();

    let t1 = std::time::Instant::now();
    let proof = prove_committed_shard_dag_with_ranges(
        &mut store,
        &ops,
        &[(0, 1)],
        &whir,
        &cached,
        &mut rng,
        tokens_t,
        argmax_t,
    );
    let prove_t = t1.elapsed();

    let t2 = std::time::Instant::now();
    let bytes = bincode::serialize(&proof).unwrap();
    assert!(verify_committed_shard_dag_with_ranges(
        &store.owned_slices(),
        &ops,
        &[(0, 1)],
        &whir,
        &proof,
    ));
    let verify_t = t2.elapsed();

    println!(
        "commit={commit_t:?} prove={prove_t:?} verify={verify_t:?} rss={}kB proof_bytes={}",
        peak_rss_kb(),
        bytes.len()
    );
}
