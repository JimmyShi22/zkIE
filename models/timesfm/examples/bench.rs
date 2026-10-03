//! Sweep shard granularity (ops_per_shard) for TimesFM 200M and report
//! prove/verify wall time + peak RSS, to pick the optimal multi-shard config.

use zkie_core::common::field::{PrimeCharacteristicRing, XorShift64};
use zkie_models_timesfm::build_timesfm;
use zkie_ops::compose::{prove_shard_dag, verify_shard_dag, Store};

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
    build_timesfm(&mut store, &mut ops, dir, 16);
    let total = ops.len();
    println!("total ops: {total}");

    let mut best_time = f64::MAX;
    let mut best_pps = 0usize;
    for &pps in &[108usize, 216, 432, 1080, 2160, total] {
        let mut s = Store { v: store.v.clone(), idx: store.idx.clone() };
        let mut rng = XorShift64::new(0xBEEF);
        let t0 = std::time::Instant::now();
        let proof = prove_shard_dag(&mut s, &ops, pps, &mut rng);
        let prove_t = t0.elapsed();
        let t1 = std::time::Instant::now();
        assert!(verify_shard_dag(&mut s, &ops, pps, &proof));
        let verify_t = t1.elapsed();
        let total_t = (prove_t + verify_t).as_secs_f64();
        println!("pps={pps:5} prove={prove_t:?} verify={verify_t:?} total={total_t:.2}s rss={}kB",
            peak_rss_kb());
        if total_t < best_time { best_time = total_t; best_pps = pps; }
    }
    println!("best ops_per_shard: {best_pps} ({:.2}s total)", best_time);
}
