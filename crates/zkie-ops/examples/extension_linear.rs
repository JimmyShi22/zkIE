//! Tiny runnable demo of the claim-driven linear-op reducer (stage B1).
//!
//! Run: cargo run --release -p zkie-ops --example extension_linear
//!
//! Scale (exact, shift 0), Add, AddConst, and Transpose: full-EF output
//! point/value mapped to input claims, WHIR-authenticated against statement
//! roots. Verifier = statement + proof only (no forward recomputation, no
//! opens). WHIR: PCS target 90 bits, PoW budget 0 — correctness prototype,
//! not a full-model or end-to-end soundness claim.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_ops::extension_linear::{prove, verify, LinearOpKind};

fn main() {
    let mut rng = XorShift64::new(0xB1C0);
    let (m, k) = (4usize, 8usize);
    let arity = (m * k).trailing_zeros() as usize;
    let whir = zkie_core::pcs::whir::Whir::new_target(arity, 90, 0).expect("valid arity");

    let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
    let b: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
    let out: Vec<Goldilocks> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();

    let t0 = std::time::Instant::now();
    let (stmt, proof) =
        prove(&whir, LinearOpKind::Add, m, k, &a, Some(&b), &out).expect("honest prove");
    let prove_time = t0.elapsed();
    let t1 = std::time::Instant::now();
    let ok = verify(&whir, &stmt, &proof);
    let verify_time = t1.elapsed();

    println!("extension_linear Add: dims m={m} k={k}, PCS target 90 bits, PoW budget 0");
    println!("prove {:>10.3} ms | verify {:>10.3} ms | ok={ok}",
        prove_time.as_secs_f64() * 1e3, verify_time.as_secs_f64() * 1e3);
    println!("open_stats after prove+verify: {:?} | verify_stats: {:?}",
        whir.open_stats(), whir.verify_stats());
    println!("(correctness prototype: no full-model or end-to-end soundness claim)");
    assert!(ok);
}
