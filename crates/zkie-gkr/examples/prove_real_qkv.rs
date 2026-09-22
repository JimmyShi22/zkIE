//! Prove the layer-0 QKV projection with a *real* activation from the forward
//! pass (not a random one), against the real QKV weight.

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

fn load_i16(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run the extraction script first");
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(from_i16(i16::from_le_bytes([chunk[0], chunk[1]])));
    }
    out
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let act = load_i16(&format!("{base}qkv_l0_act_i16.bin"));
    let weight = load_i16(&format!("{base}qkv_l0_pad_i16.bin"));
    let (m, k, n) = (1usize, 512usize, 1024usize);
    assert_eq!(act.len(), m * k);
    assert_eq!(weight.len(), k * n);

    let c = dense(&act, &weight, m, k, n);
    let at = transpose(&act, m, k);

    let mut rng = XorShift64::new(0x7ee);
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
    println!("real activation QKV matmul GKR proof verified");
}
