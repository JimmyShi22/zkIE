//! Microbenchmark: N separate `open()` vs one `open_batch_multi()` at the same point.
use std::time::Instant;
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::pcs::whir::Whir;

fn main() {
    let log2 = 16usize; // 2^16 elements per MLE
    let n = 64usize; // number of tensors opened at the same point
    let mut rng = XorShift64::new(42);
    let tensors: Vec<Vec<Goldilocks>> = (0..n)
        .map(|_| (0..(1usize << log2)).map(|_| rng.field()).collect())
        .collect();
    let point: Vec<Goldilocks> = (0..log2).map(|_| rng.field()).collect();

    let whir = Whir::new_testing(log2);

    // Separate: commit + open each (N commits, N opens).
    let t0 = Instant::now();
    for t in &tensors {
        let (_, pd, proto) = whir.commit(t);
        let _ = whir.open(pd, &proto, &point);
    }
    let separate = t0.elapsed();

    // Batch: commit_batch + one open_batch_multi (1 commit, 1 open).
    let t1 = Instant::now();
    let refs: Vec<&[Goldilocks]> = tensors.iter().map(|t| t.as_slice()).collect();
    let (_, pd, proto, bw) = whir.commit_batch(&refs);
    let _ = bw.open_batch_multi(pd, &proto, n, &point);
    let batch = t1.elapsed();

    println!("separate: {n} commit+open = {separate:?}");
    println!("batch:    1 commit_batch + 1 open_batch_multi = {batch:?}");
    println!("speedup: {:.1}x", separate.as_secs_f64() / batch.as_secs_f64());
}