//! Prove the real TimesFM FFN with int32 fixed-point (real input + weights).
//!
//! Loads the real FFN input activation and gate/down weights (all quantized at
//! scale 2^12 into int32, padded), runs FFN1 -> ReLU -> FFN2, and verifies both
//! GKR matmuls. Demonstrates the int32 fixed-point path the real model needs.

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64, XorShift64};
use zkie_gkr::fixed_point::from_i32;
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

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run the extraction script first");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn relu(v: Goldilocks) -> Goldilocks {
    if v.as_canonical_u64() < zkie_gkr::field::P / 2 {
        v
    } else {
        Goldilocks::ZERO
    }
}

fn prove_matmul(
    at: &[Goldilocks],
    b: &[Goldilocks],
    c: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    rng: &mut XorShift64,
) {
    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let proof = matmul::prove(at, b, c, m, k, n, &u, &v, &ch);
    let mut fp = u.clone();
    fp.extend_from_slice(&ch);
    let f = mle::eval(at, &fp);
    let mut hp = v.clone();
    hp.extend_from_slice(&ch);
    let h = mle::eval(b, &hp);
    assert!(matmul::verify(&proof, &ch, f, h));
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let act = load_i32(&format!("{base}ffn_in_i32_pad.bin"));
    let gate = load_i32(&format!("{base}gate_l0_i32.bin"));
    let down = load_i32(&format!("{base}down_l0_i32.bin"));
    let (m, hidden, inter) = (1usize, 512usize, 1024usize);
    assert_eq!(act.len(), m * hidden);
    assert_eq!(gate.len(), hidden * inter);
    assert_eq!(down.len(), inter * hidden);

    let ffn1 = dense(&act, &gate, m, hidden, inter);
    let g: Vec<Goldilocks> = ffn1.iter().map(|&v| relu(v)).collect();
    let ffn2 = dense(&g, &down, m, inter, hidden);

    let mut rng = XorShift64::new(0x7ee);
    prove_matmul(&transpose(&act, m, hidden), &gate, &ffn1, m, hidden, inter, &mut rng);
    prove_matmul(&transpose(&g, m, inter), &down, &ffn2, m, inter, hidden, &mut rng);
    println!("real FFN (int32 fixed-point) GKR proofs verified");
}
