//! Tiny runnable two-shard extension-field GKR chain demo.
//!
//! Run: cargo run --release -p zkie-ops --example extension_chain
//!
//! Linear matmuls only (Y = H @ W2, H = X @ W1), non-square power-of-two
//! matrices, all challenges/claims in the full quadratic extension field,
//! base-Goldilocks tensor storage. WHIR instances: PCS target 90 bits, PoW
//! budget 0 — CORRECTNESS PROTOTYPE; this is not a claim of end-to-end
//! 90-bit soundness (sumcheck round soundness and the Fiat–Shamir reduction
//! are not analyzed here).

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_ops::extension_chain::{prove, verify, ChainWhir};

fn matmul(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for kk in 0..k {
                acc = acc + a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn main() {
    let (m, d, k, n) = (4usize, 8usize, 8usize, 16usize);
    let mut rng = XorShift64::new(0xEC1);
    let x: Vec<Goldilocks> = (0..m * d).map(|_| rng.field()).collect();
    let w1: Vec<Goldilocks> = (0..d * k).map(|_| rng.field()).collect();
    let w2: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
    let h = matmul(&x, &w1, m, d, k);
    let y = matmul(&h, &w2, m, k, n);

    let chain = ChainWhir::new(m, d, k, n, 90, 0).expect("valid dims");
    let t0 = std::time::Instant::now();
    let (stmt, proof) = prove(&chain, &x, &w1, &h, &w2, &y, m, d, k, n).expect("honest prove");
    let prove_time = t0.elapsed();
    let t1 = std::time::Instant::now();
    let ok = verify(&chain, &stmt, &proof);
    let verify_time = t1.elapsed();

    println!("extension_chain: dims m={m} d={d} k={k} n={n}, PCS target 90 bits, PoW budget 0");
    println!("prove {:>10.3} ms | verify {:>10.3} ms | ok={ok}", prove_time.as_secs_f64() * 1e3, verify_time.as_secs_f64() * 1e3);
    println!("rounds: shard2={} shard1={}", proof.rounds2.len(), proof.rounds1.len());
    let opens = (
        chain.x.open_stats(),
        chain.w1.open_stats(),
        chain.h.open_stats(),
        chain.w2.open_stats(),
        chain.y.open_stats(),
    );
    println!("open_stats after prove+verify (x,w1,h,w2,y): {:?}", opens);
    let verifies = (
        chain.x.verify_stats(),
        chain.w1.verify_stats(),
        chain.h.verify_stats(),
        chain.w2.verify_stats(),
        chain.y.verify_stats(),
    );
    println!("verify_stats (x,w1,h,w2,y): {:?}", verifies);
    println!("(correctness prototype: no end-to-end 90-bit soundness claim)");
    assert!(ok);
}
