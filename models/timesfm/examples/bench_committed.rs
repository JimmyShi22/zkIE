//! TimesFM 200M committed (root-only) path, end to end: weights and shard
//! boundaries are WHIR-committed (90-bit) and opened, verifier holds only roots.

use zkie_core::common::field::XorShift64;
use zkie_core::pcs::whir::Whir;
use zkie_models_timesfm::build_timesfm;
use zkie_ops::compose::{
    commit_model_weights, prove_committed_shard_dag, verify_committed_shard_dag_bytes, Store,
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
    let dir = "models/timesfm/weights";
    let mut store = Store::new();
    let mut ops = Vec::new();
    let _out = build_timesfm(&mut store, &mut ops, dir, 16);
    let tokens = store.push_idx(vec![]);
    let argmax = store.push_idx(vec![]);
    println!("total ops: {}", ops.len());

    // Boundary tensors are seq x H_PAD = 16 x 2048 = 2^15.
    let whir = Whir::new_testing(11);
    let mut rng = XorShift64::new(0xBEEF);

    let t0 = std::time::Instant::now();
    let cached = commit_model_weights(&store, &ops, "/data/jimmyshi/zkie-cache/timesfm-model-batches-v2");
    let commit_t = t0.elapsed();

    let t1 = std::time::Instant::now();
    let proof = prove_committed_shard_dag(&mut store, &ops, 108, &whir, &cached, tokens, argmax);
    let prove_t = t1.elapsed();

    let t2 = std::time::Instant::now();
    let bytes = bincode::serialize(&proof).unwrap();
    assert!(
        verify_committed_shard_dag_bytes(&bytes, &store.owned_slices(), &ops, 108, &whir),
        "committed proof failed"
    );
    let verify_t = t2.elapsed();

    println!(
        "commit={commit_t:?} prove={prove_t:?} verify={verify_t:?} rss={}kB proof_bytes={}",
        peak_rss_kb(),
        bytes.len()
    );
}
