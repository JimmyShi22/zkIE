//! Prove a GKR matmul over a *real* TimesFM weight matrix.
//!
//! Loads `models/qkv_l0_pad_i16.bin` (layer 0's QKV projection weight,
//! quantized Q8.8 and zero-padded from 264x792 to 512x1024), maps it into
//! Goldilocks via int16, and proves one matmul against a random activation with
//! the GKR sum-check. This closes the loop from "synthetic data works" to "real
//! model weights work".

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
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

fn main() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/qkv_l0_pad_i16.bin"
    );
    let bytes = std::fs::read(path).expect("run the extraction script first");
    let mut i16s = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        i16s.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    let (k, n) = (512usize, 1024usize);
    assert_eq!(i16s.len(), k * n, "padded weight must be 512x1024");
    let weight: Vec<Goldilocks> = i16s.iter().map(|&x| from_i16(x)).collect();

    let m = 16usize;
    let mut rng = XorShift64::new(0x7ee);
    // Random int16 activation (16 x 512), mapped the same way.
    let act: Vec<Goldilocks> = (0..m * k)
        .map(|_| from_i16((rng.next_u64() % 65536) as i16))
        .collect();
    let c = dense(&act, &weight, m, k, n);
    let at = transpose(&act, m, k);

    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let proof = matmul::prove(&at, &weight, &c, m, k, n, &u, &v, &ch);
    let mut fp = u.clone();
    fp.extend_from_slice(&ch);
    let f = mle::eval(&at, &fp);
    let mut hp = v.clone();
    hp.extend_from_slice(&ch);
    let h = mle::eval(&weight, &hp);
    assert!(matmul::verify(&proof, &ch, f, h));
    println!("real QKV weight matmul (16x512 @ 512x1024) GKR proof verified");
}
