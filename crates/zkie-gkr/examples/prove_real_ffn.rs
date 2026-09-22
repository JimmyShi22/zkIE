//! Prove a real TimesFM FFN (matmul -> ReLU -> matmul) with real weights.
//!
//! Loads layer 0's `gate_proj` and `down_proj` weights (Q8.8, padded to
//! 512x1024 and 1024x512), runs one FFN on a random activation, and proves both
//! GKR sum-check matmuls. ReLU is applied host-side here; the lookup binding is
//! exercised separately in `tests/lookup_bound.rs`.

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64, XorShift64};
use zkie_gkr::fixed_point::from_i16;
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

fn load_i16(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run the extraction script first");
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(from_i16(i16::from_le_bytes([chunk[0], chunk[1]])));
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
    let gate = load_i16(&format!("{base}gate_l0_pad_i16.bin"));
    let down = load_i16(&format!("{base}down_l0_pad_i16.bin"));
    assert_eq!(gate.len(), 512 * 1024);
    assert_eq!(down.len(), 1024 * 512);

    let (m, hidden, inter) = (16usize, 512usize, 1024usize);
    let mut rng = XorShift64::new(0x7ee);
    let act: Vec<Goldilocks> = (0..m * hidden)
        .map(|_| from_i16((rng.next_u64() % 65536) as i16))
        .collect();

    // FFN1 = act @ gate
    let ffn1 = dense(&act, &gate, m, hidden, inter);
    // ReLU (host-side; lookup binding is in lookup_bound.rs)
    let g: Vec<Goldilocks> = ffn1.iter().map(|&v| relu(v)).collect();
    // FFN2 = g @ down
    let ffn2 = dense(&g, &down, m, inter, hidden);

    prove_matmul(&transpose(&act, m, hidden), &gate, &ffn1, m, hidden, inter, &mut rng);
    prove_matmul(&transpose(&g, m, inter), &down, &ffn2, m, inter, hidden, &mut rng);
    println!("real FFN (gate_proj -> ReLU -> down_proj) GKR proofs verified");
}
