//! Prove the real FFN gate matmul with WHIR commitments (not just the sumcheck).
//!
//! Loads the real FFN input and gate weight (int32, padded), commits both via
//! WHIR, computes FFN1 = input @ gate, commits it, and verifies the GKR matmul
//! from prescribed-point openings — the verifier trusts commitments + openings,
//! not raw evaluations.

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::from_i32;
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

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run the extraction script first");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let input = load_i32(&format!("{base}ffn_in_i32_pad.bin"));
    let gate = load_i32(&format!("{base}gate_l0_i32.bin"));
    let (m, k, n) = (1usize, 512usize, 1024usize);
    assert_eq!(input.len(), m * k);
    assert_eq!(gate.len(), k * n);

    let ffn1 = dense(&input, &gate, m, k, n);

    // Commit input (2^9), gate (2^19), and ffn1 (2^10) once each.
    let whir_in = Whir::new_testing(9);
    let whir_gate = Whir::new_testing(19);
    let whir_out = Whir::new_testing(10);
    let (in_c, in_pd, in_proto) = whir_in.commit(&input);
    let (gate_c, gate_pd, gate_proto) = whir_gate.commit(&gate);
    let (out_c, out_pd, out_proto) = whir_out.commit(&ffn1);

    let mut rng = XorShift64::new(0x7ee);
    // m=1 -> u has 0 bits; A is a vector, so no transpose point-swap needed.
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let proof = matmul::prove(&input, &gate, &ffn1, m, k, n, &[], &v, &ch);

    let (a_open, f) = whir_in.open(in_pd, &in_proto, &ch);
    let mut bp = v.clone();
    bp.extend_from_slice(&ch);
    let (b_open, h) = whir_gate.open(gate_pd, &gate_proto, &bp);
    let (c_open, claimed) = whir_out.open(out_pd, &out_proto, &v);

    assert_eq!(whir_in.verify(&in_c, &a_open, &in_proto, &ch).unwrap(), f);
    assert_eq!(whir_gate.verify(&gate_c, &b_open, &gate_proto, &bp).unwrap(), h);
    assert_eq!(whir_out.verify(&out_c, &c_open, &out_proto, &v).unwrap(), claimed);
    assert_eq!(f, mle::eval(&input, &ch));
    assert_eq!(h, mle::eval(&gate, &bp));
    assert_eq!(claimed, mle::eval(&ffn1, &v));
    assert!(matmul::verify(&proof, &ch, f, h));
    println!("real FFN gate matmul verified from WHIR commitments + openings");
}
