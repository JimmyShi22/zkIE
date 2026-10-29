//! GPT-2 committed (root-only) shard-DAG path, end to end: weights and shard
//! boundaries are WHIR-committed (90-bit) and opened, the verifier holds only
//! commitment roots + opening proofs. Reports commit / prove / verify separately.

use std::fs;

use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i32, to_i32, to_i64};
use zkie_core::pcs::whir::Whir;
use zkie_models_gpt2::build_gpt2;
use zkie_ops::compose::{
    commit_model_weights, prove_committed_shard_dag, verify_committed_shard_dag_bytes,
    verify_sequence_shift, PublicStatement, Store,
};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
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
    let m = std::env::var("M").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let shift = 16u32;
    let dir = "models/gpt2/weights";

    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, logits, tokens, t) = build_gpt2(&mut store, &mut ops, dir, m, shift);
    let gt = load_i32(&format!("{dir}/gt_argmax_{}i32.bin", if m <= 16 { "" } else { "512_" }));

    // Boundary tensors are m x D = 512 x 1024 = 2^19.
    let whir = Whir::new_testing(19);
    let mut rng = XorShift64::new(0xBEEF);

    let t0 = std::time::Instant::now();
    let cached =
        commit_model_weights(&store, &ops, "/data/jimmyshi/zkie-cache/gpt2-model-batches-v2");
    let commit_t = t0.elapsed();

    let t1 = std::time::Instant::now();
    let proof = prove_committed_shard_dag(&mut store, &ops, 176, &whir, &cached, tokens, t);
    let prove_t = t1.elapsed();

    let t2 = std::time::Instant::now();
    let bytes = bincode::serialize(&proof).unwrap();
    // S6: this benchmark replays extracted (non-generated) tokens, so the
    // language-model sequence statement is not asserted here; real public
    // sequence verification passes a `PublicStatement::LanguageModel` with
    // the caller-confirmed prompt + response (see the s6_* tests).
    assert!(
        verify_committed_shard_dag_bytes(
            &bytes,
            &store.owned_slices(),
            &ops,
            176,
            &whir,
            &PublicStatement::Generic
        ),
        "committed proof failed"
    );
    let verify_t = t2.elapsed();

    let lg = store.get(logits);
    let mut matches = 0;
    for i in 0..m {
        let mut best = 0usize;
        let mut best_v = lg[i * 65536];
        for j in 1..50257 {
            let v = lg[i * 65536 + j];
            if to_i64(v) > to_i64(best_v) {
                best_v = v;
                best = j;
            }
        }
        if best as i32 == to_i32(gt[i]) {
            matches += 1;
        }
    }

    // Argmax-op sanity: the proven index vector must match the post-hoc argmax.
    // Argmax-op sanity: the proven index t must equal the argmax of the *scaled*
    // logits (the actual input to the Argmax op, right-shifted by 8).
    let arg_idx = &store.idx[t];
    let mut argmax_ok = true;
    for i in 0..m {
        let mut best = 0usize;
        let mut best_v = to_i64(lg[i * 65536]);
        for j in 1..50257 {
            let v = to_i64(lg[i * 65536 + j]);
            if v > best_v {
                best_v = v;
                best = j;
            }
        }
        if arg_idx[i] as usize != best {
            argmax_ok = false;
        }
    }
    let mut n_mismatch = 0;
    for i in 0..m {
        let mut best = 0usize;
        let mut best_v = to_i64(lg[i * 65536]);
        for j in 1..50257 {
            let v = to_i64(lg[i * 65536 + j]);
            if v > best_v { best_v = v; best = j; }
        }
        if arg_idx[i] as usize != best {
            n_mismatch += 1;
        }
    }
    println!("argmax_op_matches={argmax_ok} n_mismatch={n_mismatch}");
    // Sequence self-consistency is a verifier-side check on the public tokens/argmax
    // (only meaningful for a real generated sequence; random tokens will not hold).
    println!(
        "sequence_shift_ok={}",
        verify_sequence_shift(&store.idx[tokens], &store.idx[t], &store.idx[tokens][..3])
    );
    println!(
        "commit={commit_t:?} prove={prove_t:?} verify={verify_t:?} argmax={matches}/{m} rss={}kB proof_bytes={}",
        peak_rss_kb(),
        bytes.len()
    );
    let shards_b = bincode::serialize(&proof.shards).unwrap().len();
    let wbo_b = bincode::serialize(&proof.weight_batch_openings).unwrap().len();
    let bound_b = bincode::serialize(&(&proof.boundary_commitments, &proof.cross_binds)).unwrap().len();
    println!("breakdown: shards={} weight_openings={} boundary={}", shards_b, wbo_b, bound_b);
    for (i, wbo) in proof.weight_batch_openings.iter().enumerate() {
        if let Some(p) = &wbo.proof {
            println!("  group{i}: open={} sumcheck={} evals={} n_claims={}",
                bincode::serialize(&p.open).unwrap().len(),
                bincode::serialize(&p.sumcheck).unwrap().len(),
                bincode::serialize(&p.evals).unwrap().len(),
                wbo.table_idxs.len());
        }
    }
}
