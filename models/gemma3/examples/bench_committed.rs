//! Gemma 3 270M committed (root-only) path, end to end.

use std::fs;

use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i32, to_i32, to_i64};
use zkie_core::pcs::whir::Whir;
use zkie_models_gemma3::{build_gemma3, VOCAB};
use zkie_ops::compose::{
    commit_model_weights, prove_committed_shard_dag, verify_committed_shard_dag_bytes,
    verify_sequence_shift, Store,
};

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
    let m = std::env::var("GEMMA_SEQ").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let shift = 16u32;
    let dir = "models/gemma3/weights";

    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, logits, tokens, t) = build_gemma3(&mut store, &mut ops, dir, m, shift);
    let gt = load_i32(&format!("{dir}/gt_argmax_{}i32.bin", if m <= 16 { "" } else { "512_" }));

    // Boundary tensors are seq x H_PAD = 16 x 1024 = 2^14.
    let whir = Whir::new_testing(14);
    let mut rng = XorShift64::new(0xBEEF);

    let t0 = std::time::Instant::now();
    let cached = commit_model_weights(&store, &ops, "/data/jimmyshi/zkie-cache/gemma3-model-batches-v2");
    let commit_t = t0.elapsed();

    let t1 = std::time::Instant::now();
    let proof = prove_committed_shard_dag(&mut store, &ops, 53, &whir, &cached, tokens, t);
    let prove_t = t1.elapsed();

    let t2 = std::time::Instant::now();
    let bytes = bincode::serialize(&proof).unwrap();
    assert!(
        verify_committed_shard_dag_bytes(&bytes, &store.owned_slices(), &ops, 53, &whir),
        "committed proof failed"
    );
    let verify_t = t2.elapsed();

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

    // Argmax-op sanity: the proven index must match the post-hoc argmax of the
    // scaled logits (the actual input to the Argmax op).
    let arg_idx = &store.idx[t];
    let mut argmax_ok = true;
    for i in 0..m {
        let mut best = 0usize;
        let mut best_v = to_i64(lg[i * VOCAB]) / 8192;
        for j in 1..VOCAB {
            let v = to_i64(lg[i * VOCAB + j]) / 8192;
            if v > best_v {
                best_v = v;
                best = j;
            }
        }
        if arg_idx[i] as usize != best {
            argmax_ok = false;
        }
    }
    println!("argmax_op_matches={argmax_ok}");
    println!(
        "sequence_shift_ok={}",
        verify_sequence_shift(&store.idx[tokens], &store.idx[t], &[])
    );
    println!(
        "commit={commit_t:?} prove={prove_t:?} verify={verify_t:?} argmax={matches}/{m} rss={}kB proof_bytes={}",
        peak_rss_kb(),
        bytes.len()
    );
}
