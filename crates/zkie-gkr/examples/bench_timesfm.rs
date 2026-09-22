//! Benchmark the GKR prover on TimesFM-shaped matmuls (power-of-2 padded).
//!
//! TimesFM 8M: hidden=264 (pad to 512), intermediate=1024, 16 patches/context,
//! 7 layers. The dominant work is the FFN (264x1024 -> 512x1024) and QKV
//! (264x264 -> 512x512) matmuls. This reports the GKR sum-check prover cost and
//! the WHIR commitment cost for one FFN weight matrix, then extrapolates.

use std::time::Instant;

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::whir::Whir;
use zkie_gkr::{matmul, mle};

fn dense(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for w in 0..k {
                acc = acc + a[i * k + w] * b[w * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut at = vec![Goldilocks::ZERO; k * m];
    for i in 0..m {
        for w in 0..k {
            at[w * m + i] = a[i * k + w];
        }
    }
    at
}

fn bench_matmul(m: usize, k: usize, n: usize, rng: &mut XorShift64) -> f64 {
    let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
    let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
    let c = dense(&a, &b, m, k, n);
    let at = transpose(&a, m, k);
    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();

    let t0 = Instant::now();
    let proof = matmul::prove(&at, &b, &c, m, k, n, &u, &v, &ch);
    let secs = t0.elapsed().as_secs_f64();
    assert!(matmul::verify(&proof, &ch, mle::eval(&at, &{ let mut p = u; p.extend_from_slice(&ch); p }), mle::eval(&b, &{ let mut p = v; p.extend_from_slice(&ch); p })));
    secs
}

fn main() {
    let mut rng = XorShift64::new(0x71f);

    // TimesFM 8M shapes, power-of-2 padded.
    let (seq, hidden_pad, intermediate) = (16usize, 512usize, 1024usize);

    // FFN up-projection: (seq x hidden) @ (hidden x intermediate).
    let ffn_up = bench_matmul(seq, hidden_pad, intermediate, &mut rng);
    // FFN down-projection: (seq x intermediate) @ (intermediate x hidden).
    let ffn_down = bench_matmul(seq, intermediate, hidden_pad, &mut rng);
    // QKV as three separate projections (hidden -> hidden each); the fused
    // 3*hidden output is not a power of two, so it must be split.
    let qkv = bench_matmul(seq, hidden_pad, hidden_pad, &mut rng) * 3.0;

    println!("padded matmul sum-check prover (one layer, 16 patches):");
    println!("  ffn_up   {ffn_up:.4}s");
    println!("  ffn_down {ffn_down:.4}s");
    println!("  qkv      {qkv:.4}s");
    let attention = bench_matmul(seq, hidden_pad, seq, &mut rng) * 2.0;
    let per_layer = ffn_up + ffn_down + qkv + attention;
    println!("  per-layer sumcheck ~{per_layer:.4}s, 7 layers ~{:.2}s", per_layer * 7.0);

    // WHIR commitment cost for one FFN weight matrix (512 x 1024 = 524288 elems).
    let whir = Whir::new_testing(19); // 2^19 = 524288
    let w: Vec<Goldilocks> = (0..hidden_pad * intermediate).map(|_| rng.field()).collect();
    let t0 = Instant::now();
    let (_commitment, _pd, _proto) = whir.commit(&w);
    let commit_secs = t0.elapsed().as_secs_f64();
    println!("whir commit 512x1024 (524288 elems, testing security): {commit_secs:.3}s");
    // TimesFM 8M has ~8M weights + activations; each element is hashed once.
    let total_elems = 8_000_000u64;
    println!(
        "  extrapolated commit for 8M elems ~{:.1}s (testing security; 90-bit PoW adds per-proof overhead)",
        commit_secs * total_elems as f64 / 524288.0
    );
}
