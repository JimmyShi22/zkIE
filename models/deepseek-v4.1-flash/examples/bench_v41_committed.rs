//! Committed store-free roundtrip for the full DeepSeek-V4.1-Flash text path.
//!
//! Builds the 40-layer Hyper-Connection + MLA + MoE + Engram forward and the
//! output head, commits every weight group once, proves with one shard per
//! layer boundary, and verifies root-only. Env: `ZKIE_LAYERS` (default 1),
//! `ZKIE_M` (default 16), `ZKIE_NO_ROUTED` to disable routed experts,
//! `ZKIE_REAL` to use the real prompt embedding.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;
use zkie_models_deepseek_v41_flash::{
    build_final_logits, build_v41_forward, push_embedding, DIM_PAD,
};
use zkie_ops::compose::{
    commit_model_weights, prove_committed_shard_dag_with_ranges,
    verify_committed_shard_dag_with_ranges, Op, Store,
};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn load_i32_raw(path: &str) -> Vec<i32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw".to_string());
    let m: usize = std::env::var("ZKIE_M").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let n_layers: usize = std::env::var("ZKIE_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let routed = std::env::var("ZKIE_NO_ROUTED").is_err();

    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();
    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let rsqrt_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let sqrt_sp = store.push(load_i32("/data/zkIE/models/sqrt_softplus_table_i32.bin"));
    let silu = store.push(load_i32("/data/zkIE/models/silu_clipped_table_i32.bin"));
    let clip = store.push(load_i32("/data/zkIE/models/clip_m10_p10_table_i32.bin"));
    let sigmoid = store.push(load_i32("/data/zkIE/models/sigmoid_table_i32.bin"));
    let gate_table = store.push(load_i32("/data/zkIE/models/engram_gate_table_i32.bin"));
    let hash_ids = load_i32_raw("/data/zkIE/models/deepseek-v4.1-flash/cache/engram/hash_ids_i32.bin");
    let hash_ids: Vec<u32> = hash_ids.into_iter().map(|v| v as u32).collect();
    let x = if std::env::var("ZKIE_REAL").is_ok() {
        push_embedding(
            &mut store,
            "/data/zkIE/models/deepseek-v4.1-flash/cache/engram/embed_prompt_i32.bin",
            m,
        )
    } else {
        store.push(vec![from_i64(1 << 16); m * DIM_PAD])
    };

    let (x4, pre_mix, boundaries) = build_v41_forward(
        &mut store,
        &mut ops,
        &dir,
        m,
        16,
        routed,
        n_layers,
        x,
        exp_table,
        rsqrt_table,
        limb_table,
        sqrt_sp,
        silu,
        clip,
        gate_table,
        &hash_ids,
        sigmoid,
    );

    let mut argmax_t = store.push_idx(vec![]);
    if n_layers >= 40 {
        let (_logits, argmax) = build_final_logits(
            &mut store,
            &mut ops,
            &dir,
            m,
            16,
            x4,
            pre_mix,
            rsqrt_table,
            limb_table,
        );
        argmax_t = argmax;
    }

    let tokens_t = store.push_idx(vec![]);
    let mut ranges: Vec<(usize, usize)> = boundaries
        .windows(2)
        .map(|w| (w[0], w[1]))
        .filter(|&(s, e)| s < e)
        .collect();
    if n_layers >= 40 && *boundaries.last().unwrap() < ops.len() {
        ranges.push((*boundaries.last().unwrap(), ops.len()));
    }

    let whir = Whir::new_testing(10);
    let mut rng = XorShift64::new(0xB10C);
    let cache_dir = format!("/data/jimmyshi/zkie-cache/deepseek-v41-flash-l{n_layers}-m{m}");

    let t0 = std::time::Instant::now();
    let cached = commit_model_weights(&store, &ops, &cache_dir);
    let commit_t = t0.elapsed();

    let t1 = std::time::Instant::now();
    let proof = prove_committed_shard_dag_with_ranges(
        &mut store,
        &ops,
        &ranges,
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
        &ranges,
        &whir,
        &proof,
    ));
    let verify_t = t2.elapsed();

    let am = &store.idx[argmax_t];
    println!(
        "v41 committed: layers={n_layers} m={m} routed={routed} shards={} ops={} commit={commit_t:?} prove={prove_t:?} verify={verify_t:?} rss={}kB proof_bytes={} argmax={:?}",
        ranges.len(),
        ops.len(),
        peak_rss_kb(),
        bytes.len(),
        &am[..m.min(am.len())],
    );
}
