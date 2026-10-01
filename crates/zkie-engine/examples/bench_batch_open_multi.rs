//! Microbenchmark: N separate opens at DIFFERENT points vs one multi-point batch_open.
use std::time::Instant;
use zkie_core::pcs::batch_open::batch_open;
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::mle;
use zkie_core::pcs::whir::Whir;

fn main() {
    let log2 = 16usize;
    let n = 64usize;
    let mut rng = XorShift64::new(7);
    let tensors: Vec<Vec<Goldilocks>> =
        (0..n).map(|_| (0..(1usize << log2)).map(|_| rng.field()).collect()).collect();
    let points: Vec<Vec<Goldilocks>> =
        (0..n).map(|_| (0..log2).map(|_| rng.field()).collect()).collect();
    let claimed: Vec<Goldilocks> =
        (0..n).map(|i| mle::eval(&tensors[i], &points[i])).collect();

    // Separate: commit + open each table at its own point.
    let whir = Whir::new_testing(log2);
    let t0 = Instant::now();
    for i in 0..n {
        let (_, pd, proto) = whir.commit(&tensors[i]);
        let _ = whir.open(pd, &proto, &points[i]);
    }
    let separate = t0.elapsed();

    // batch_open: multi-point-to-one-point reduction (1 sumcheck + 1 FRI open).
    let t1 = Instant::now();
    let mut r2 = XorShift64::new(99);
    let ok = batch_open(&tensors, &points, &claimed, &mut r2);
    let batch = t1.elapsed();

    println!("separate:   {n} commit+open (different points) = {separate:?}");
    println!("batch_open: multi-point reduction            = {batch:?}  (ok={ok})");
    println!("speedup: {:.1}x", separate.as_secs_f64() / batch.as_secs_f64());
}