//! Committed store-free roundtrip for the full layer-0 block (attention + HC +
//! shared-expert MoE, routed experts off for speed). Exercises
//! `commit_model_weights`, `prove_committed_shard_dag_with_ranges`, and the
//! root-only verifier on a real layer-0 computation.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;
use zkie_models_deepseek_v41_flash::{
    build_block, build_identity_pre_mix, expand_to_hc, open_shard, DIM_PAD, HC_MULT,
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
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string()
    });
    let m: usize = std::env::var("ZKIE_M").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let file = open_shard(&path);
    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();

    let rsqrt_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let sqrt_sp = store.push(load_i32("/data/zkIE/models/sqrt_softplus_table_i32.bin"));
    let silu = store.push(load_i32("/data/zkIE/models/silu_clipped_table_i32.bin"));
    let clip = store.push(load_i32("/data/zkIE/models/clip_m10_p10_table_i32.bin"));
    let x = store.push(vec![from_i64(1 << 16); m * DIM_PAD]);
    let x4 = expand_to_hc(&mut store, &mut ops, x, m, HC_MULT, DIM_PAD);
    let pre_mix = build_identity_pre_mix(&mut store, m);
    let (_x4o, _pre_mixo, _x4_attn, _ffn_out) = build_block(
        &mut store,
        &mut ops,
        file,
        0,
        m,
        16,
        false,
        x4,
        pre_mix,
        rsqrt_table,
        limb_table,
        exp_table,
        sqrt_sp,
        silu,
        clip,
    );

    let tokens_t = store.push_idx(vec![]);
    let argmax_t = store.push_idx(vec![]);
    let whir = Whir::new_testing(10);
    let mut rng = XorShift64::new(0xB10C);
    let ranges = vec![(0usize, ops.len())];

    let t0 = std::time::Instant::now();
    let cached = commit_model_weights(
        &store,
        &ops,
        "/data/jimmyshi/zkie-cache/deepseek-v41-flash-layer0-block",
    );
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

    println!(
        "layer0 committed: ops={} commit={commit_t:?} prove={prove_t:?} verify={verify_t:?} rss={}kB proof_bytes={}",
        ops.len(),
        peak_rss_kb(),
        bytes.len()
    );
}
