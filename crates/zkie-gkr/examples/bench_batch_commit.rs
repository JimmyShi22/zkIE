//! Measure single vs batch WHIR commit time (CPU and GPU paths).

use std::time::Instant;

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::whir::Whir;

fn run(rng: &mut XorShift64, whir: &Whir, size: usize, n: usize, label: &str) {
    let len = 1usize << size;
    let tensors: Vec<Vec<Goldilocks>> = (0..n)
        .map(|_| (0..len).map(|_| rng.field()).collect())
        .collect();

    let t0 = Instant::now();
    for t in &tensors {
        let _ = whir.commit(t);
    }
    let single = t0.elapsed().as_secs_f64();

    let refs: Vec<&[Goldilocks]> = tensors.iter().map(|t| t.as_slice()).collect();
    let t0 = Instant::now();
    let _ = whir.commit_batch(&refs);
    let batch = t0.elapsed().as_secs_f64();

    println!(
        "{label}: size=2^{size} n={n} | single={single:.3}s batch={batch:.3}s | speedup={:.2}x",
        single / batch
    );
}

fn main() {
    let mut rng = XorShift64::new(0xabc);
    let whir12 = Whir::new_testing(12);
    let whir22 = Whir::new_testing(22);

    run(&mut rng, &whir12, 12, 1000, "small");
    run(&mut rng, &whir22, 22, 100, "large");
}
