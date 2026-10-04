//! Does `forward_shard` actually remove the global-materialisation floor?
//!
//! On Deucalion, DeepSeek-V2-Lite at seq=512 peaks at ~330 GiB and that floor is
//! ~90 % of total memory - measured over three granularities, and it barely moves
//! with shard count because `forward_ops` materialises the whole graph before any
//! shard is proven. `forward_shard` is supposed to collapse exactly that.
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

    let live: usize = (0..store.v.len()).map(|t| store.get(t).len()).sum();
    let (s, e) = ranges[idx];
    println!(
        "mode={mode} M={m} ops={} shards={} shard_idx={idx} range={s}..{e} \
         forward={dt:?} peak_rss_kb={} after_build_kb={after_build} live_elements={live}",
        ops.len(), ranges.len(), peak_rss_kb()
    );
}
