//! Prove the full real FFN subgraph (gate matmul -> ReLU -> down matmul) from
//! WHIR commitments. ReLU is applied host-side here; the lookup binding is
//! exercised in tests/lookup_bound.rs.

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64, XorShift64};
use zkie_gkr::fixed_point::from_i32;
use zkie_gkr::whir::{Commitment, OpeningProtocol, ProverData, Whir};
use zkie_gkr::matmul;

struct Ct {
    commitment: Commitment,
    pd: ProverData,
    proto: OpeningProtocol,
}

fn commit(whir: &Whir, m: &[Goldilocks]) -> Ct {
    let (commitment, pd, proto) = whir.commit(m);
    Ct { commitment, pd, proto }
}

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

fn relu(v: Goldilocks) -> Goldilocks {
    if v.as_canonical_u64() < zkie_gkr::field::P / 2 {
        v
    } else {
        Goldilocks::ZERO
    }
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let input = load_i32(&format!("{base}ffn_in_i32_pad.bin"));
    let gate = load_i32(&format!("{base}gate_l0_i32.bin"));
    let down = load_i32(&format!("{base}down_l0_i32.bin"));
    let (hidden, inter) = (512usize, 1024usize);

    let ffn1 = dense(&input, &gate, 1, hidden, inter);
    let g: Vec<Goldilocks> = ffn1.iter().map(|&v| relu(v)).collect();
    let ffn2 = dense(&g, &down, 1, inter, hidden);

    let mut rng = XorShift64::new(0x7ee);
    // Commit once each, then prove the two matmuls with openings.
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir19 = Whir::new_testing(19);
    let c_in = commit(&whir9, &input);
    let c_ffn1 = commit(&whir10, &ffn1);
    let c_g = commit(&whir10, &g);
    let c_ffn2 = commit(&whir9, &ffn2);
    let c_gate = commit(&whir19, &gate);
    let c_down = commit(&whir19, &down);

    // gate: input @ gate = ffn1
    let ch: Vec<Goldilocks> = (0..hidden.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v1: Vec<Goldilocks> = (0..inter.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let p1 = matmul::prove(&input, &gate, &ffn1, 1, hidden, inter, &[], &v1, &ch);
    let (o_a, f) = whir9.open(c_in.pd.clone(), &c_in.proto, &ch);
    let mut bp1 = v1.clone();
    bp1.extend_from_slice(&ch);
    let (o_b, h) = whir19.open(c_gate.pd.clone(), &c_gate.proto, &bp1);
    let (o_c, claimed) = whir10.open(c_ffn1.pd.clone(), &c_ffn1.proto, &v1);
    assert_eq!(whir9.verify(&c_in.commitment, &o_a, &c_in.proto, &ch).unwrap(), f);
    assert_eq!(whir19.verify(&c_gate.commitment, &o_b, &c_gate.proto, &bp1).unwrap(), h);
    assert_eq!(whir10.verify(&c_ffn1.commitment, &o_c, &c_ffn1.proto, &v1).unwrap(), claimed);
    assert!(matmul::verify(&p1, &ch, f, h));

    // down: g @ down = ffn2
    let ch2: Vec<Goldilocks> = (0..inter.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v2: Vec<Goldilocks> = (0..hidden.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let p2 = matmul::prove(&g, &down, &ffn2, 1, inter, hidden, &[], &v2, &ch2);
    let (o_a2, f2) = whir10.open(c_g.pd.clone(), &c_g.proto, &ch2);
    let mut bp2 = v2.clone();
    bp2.extend_from_slice(&ch2);
    let (o_b2, h2) = whir19.open(c_down.pd.clone(), &c_down.proto, &bp2);
    let (o_c2, claimed2) = whir9.open(c_ffn2.pd.clone(), &c_ffn2.proto, &v2);
    assert_eq!(whir10.verify(&c_g.commitment, &o_a2, &c_g.proto, &ch2).unwrap(), f2);
    assert_eq!(whir19.verify(&c_down.commitment, &o_b2, &c_down.proto, &bp2).unwrap(), h2);
    assert_eq!(whir9.verify(&c_ffn2.commitment, &o_c2, &c_ffn2.proto, &v2).unwrap(), claimed2);
    assert!(matmul::verify(&p2, &ch2, f2, h2));

    println!("real FFN (gate -> ReLU -> down) verified from WHIR commitments + openings");
}
