//! Does ONE shard, proven on its own node, fit in a 242 GiB CPU node?
//!
//! `bench_forward_shard` answered only half the question. It measured the FORWARD
//! part: 223.2 GiB whole-model vs 121.7 GiB for the worst-case shard at seq=512, a
//! 45 % cut. But proving was never in that measurement, and prove peaked at
//! 314.8 GiB - so the 91.6 GiB sitting on top of forward was still unaccounted for.
//!
//! The catch is that the 91.6 GiB was 28 shards proving CONCURRENTLY on one shared
//! store (`prove_shard_dag` does `forward_ops` then `ranges.par_iter()`). Dividing
//! it by 28 would assume the per-shard working set is independent and additive,
//! which is exactly the kind of guess this exists to replace.
//!
//! So: run the real distributed unit of work. One shard, alone, in one process -
//! `forward_shard` for its range, then `prove_shard_precomputed` over just that
//! range - and report peak RSS for the whole thing.
//!
//! ONE SHARD PER PROCESS, deliberately: VmHWM is a high-water mark that never
//! decreases, so measuring several in one process reports the largest for all of
//! them and hides the per-node figure completely.
//!
//!   SHARD_IDX=n M=512  -> what node n actually holds, forward AND prove
//!
//! SHARD_IDX defaults to the LAST shard, the honest worst case: that node runs the
//! whole forward prefix, so its saving comes purely from releasing.
//!
//! VERIFY=1 also verifies the shard proof. Worth doing at least once per
//! configuration: a memory number from a prover that silently produced a wrong
//! proof is not a measurement of anything. It costs roughly as much as proving.
use std::fs;
use std::io::Write;
use zkie_core::common::field::XorShift64;
use zkie_models_deepseek_v2_lite::build_deepseek;
use zkie_ops::compose::{
    forward_shard, prove_shard_precomputed, shard_ranges, verify_shard_precomputed, Store,
};

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
    let verify = std::env::var("VERIFY").map(|v| v == "1").unwrap_or(false);
    let dir = "models/deepseek-v2-lite/weights";

    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, _logits) = build_deepseek(&mut store, &mut ops, dir, m, 16);
    let ranges = shard_ranges(ops.len(), ops_per_shard);
    let idx: usize = std::env::var("SHARD_IDX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(ranges.len() - 1);
    let (s, e) = ranges[idx];
    let after_build = peak_rss_kb();

    // 1. Forward, releasing as it goes. This is the node's own prefix recompute.
    let t0 = std::time::Instant::now();
    forward_shard(&mut store, &ops, s..e);
    let fwd = t0.elapsed();
    let after_fwd = peak_rss_kb();

    // 2. Prove ONLY this shard, against tensors this node computed itself. No
    //    value crosses a node boundary, so there is nothing to take on trust.
    let mut rng = XorShift64::new(0xD5EED ^ (idx as u64));
    let t1 = std::time::Instant::now();
    let proof = prove_shard_precomputed(&store, &ops[s..e], &[], &mut rng);
    let prove = t1.elapsed();
    let after_prove = peak_rss_kb();

    // Owned elements only: store.get() panics on mmap-backed tensors, and those are
    // file-backed and already near-free resident, which is what is being measured.
    let live: usize = (0..store.v.len())
        .map(|t| match &store.v[t] {
            zkie_ops::compose::TensorData::Owned(v) => v.len(),
            _ => 0,
        })
        .sum();

    println!(
        "shard_idx={idx} M={m} ops={} shards={} range={s}..{e} claims={} \
         forward={fwd:?} prove={prove:?} \
         after_build_kb={after_build} after_fwd_kb={after_fwd} peak_rss_kb={after_prove} \
         live_elements={live}",
        ops.len(),
        ranges.len(),
        proof.claims.len(),
    );
    let _ = std::io::stdout().flush();

    if verify {
        let t2 = std::time::Instant::now();
        let ok = verify_shard_precomputed(&store, &ops[s..e], &proof);
        println!("verify shard_idx={idx} ok={ok} in {:?}", t2.elapsed());
        assert!(ok, "shard {idx} proof FAILED to verify - the memory number above is meaningless");
    }
}
