//! Benchmark a single WHIR commit/open/verify cycle at 2^15 and 2^22, the two
//! dominant commitment sizes in the 200M proof (activation vs weight commits).
//!
//! Mirrors the proof's configuration exactly (`Whir::new_testing`, the same
//! seeded permutation), so the per-commit numbers here are the per-commit cost
//! of `prove_full_stack_200m`. Run with `RAYON_NUM_THREADS` to control the
//! thread pool used by the p3 parallel features.

use std::time::Instant;

use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::whir::Whir;

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "auto".into());
    println!("rayon threads: {threads}");

    for &d in &[15usize, 22] {
        let whir = Whir::new_testing(d);
        let mut rng = XorShift64::new(0x600);
        let evals: Vec<Goldilocks> = (0..(1 << d)).map(|_| rng.field()).collect();
        let point: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();

        // Warm-up (allocates twiddle tables, GPU buffers etc. when enabled).
        let (commitment, prover_data, protocol) = whir.commit(&evals);

        let t0 = Instant::now();
        let _ = whir.commit(&evals);
        let commit_secs = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let (proof, opened) = whir.open(prover_data, &protocol, &point);
        let open_secs = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let verified = whir
            .verify(&commitment, &proof, &protocol, &point)
            .expect("verify");
        let verify_secs = t0.elapsed().as_secs_f64();

        assert_eq!(opened, verified);
        println!(
            "d={d:2} ({:9} elems): commit {:8.2}s | open {:6.3}s | verify {:6.3}s",
            1 << d,
            commit_secs,
            open_secs,
            verify_secs
        );
    }
}
