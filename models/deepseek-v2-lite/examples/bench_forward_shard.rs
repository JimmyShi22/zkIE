//! Does `forward_shard` actually remove the global-materialisation floor?
//!
//! Context: on Deucalion, DeepSeek-V2-Lite PROVE at seq=512 peaks at ~315 GiB, and
//! that peak barely moves with shard count (318 / 315 / 291 GiB at 14 / 28 / 56
//! shards) because `prove_shard_dag` calls `forward_ops` first and materialises
//! the whole graph before any shard is proven. `forward_shard` exists to collapse
//! that part.
//!
//! RESULT (this benchmark, one node, ref e7eab84):
//!
//!   seq=512   whole-graph forward 223.2 GiB  ->  worst-case shard 121.7 GiB  (-45 %)
//!   seq=16    whole-graph forward  51.4 GiB  ->  worst-case shard  50.3 GiB  (-2 %)
//!
//! Two things that are easy to get wrong when reading this:
//!
//!   1. Forward is 71 % of the PROVE peak at seq=512, not all of it. The remaining
//!      ~92 GiB is proving work for all shards running concurrently on one shared
//!      store. This benchmark does not measure that; `bench_prove_shard` does.
//!   2. The saving is sequence-dependent and nearly vanishes at seq=16, where
//!      resident weights dominate and liveness has nothing to release. A small
//!      fixture will therefore suggest this change does nothing.
//!
//! ONE MODE PER PROCESS, deliberately: VmHWM is a high-water mark that never
//! decreases, so running both in one process would report the larger of the two
//! for both and hide the entire effect.
//!
//!   MODE=full  M=512            -> what the current path holds
//!   MODE=shard M=512 SHARD_IDX=n -> what one node would hold
//!
//! SHARD_IDX defaults to the LAST shard: the honest worst case, since that node
//! must run the whole forward and its saving comes purely from releasing.
use std::fs;
use zkie_core::common::field::Goldilocks;
use zkie_core::common::fixed_point::from_i32;
use zkie_models_deepseek_v2_lite::build_deepseek;
use zkie_ops::compose::{forward_shard, shard_ranges, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes.chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}
fn peak_rss_kb() -> u64 {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            return v.trim().trim_end_matches(" kB").parse().unwrap_or(0);
        }
    }
    0
}

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let ops_per_shard: usize =
        std::env::var("SHARD").ok().and_then(|v| v.parse().ok()).unwrap_or(718);
    let mode = std::env::var("MODE").unwrap_or_else(|_| "full".into());
    let dir = "models/deepseek-v2-lite/weights";

    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, _logits) = build_deepseek(&mut store, &mut ops, dir, m, 16);
    let ranges = shard_ranges(ops.len(), ops_per_shard);
    let idx: usize = std::env::var("SHARD_IDX").ok().and_then(|v| v.parse().ok())
        .unwrap_or(ranges.len() - 1);
    let after_build = peak_rss_kb();

    let t0 = std::time::Instant::now();
    match mode.as_str() {
        // The baseline is forward_shard over the WHOLE graph: its keep-set is then
        // every tensor any op touches, so it releases nothing and behaves exactly
        // like forward_ops. Using it avoids making forward_ops public purely to
        // benchmark it.
        "full" => forward_shard(&mut store, &ops, 0..ops.len()),
        "shard" => {
            let (s, e) = ranges[idx];
            forward_shard(&mut store, &ops, s..e);
        }
        other => panic!("MODE must be full|shard, got {other}"),
    }
    let dt = t0.elapsed();

    // Count OWNED elements only. store.get() panics on mmap-backed tensors, and
    // counting them would be wrong anyway: they are file-backed and already cost
    // almost nothing resident, which is the quantity this is measuring.
    let live: usize = (0..store.v.len())
        .map(|t| match &store.v[t] {
            zkie_ops::compose::TensorData::Owned(v) => v.len(),
            _ => 0,
        })
        .sum();
    let (s, e) = ranges[idx];
    println!(
        "mode={mode} M={m} ops={} shards={} shard_idx={idx} range={s}..{e} \
         forward={dt:?} peak_rss_kb={} after_build_kb={after_build} live_elements={live}",
        ops.len(), ranges.len(), peak_rss_kb()
    );
}
