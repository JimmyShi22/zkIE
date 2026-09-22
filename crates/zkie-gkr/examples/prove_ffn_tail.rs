//! Prove the complete FFN tail (LayerNorm -> gate -> ReLU -> down) from WHIR
//! commitments: the norm's raw product plus the two matmuls, using real data.

use zkie_gkr::committed::{commit, layer_norm_raw, prove_layer_norm, prove_matmul};
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::from_i32;
use zkie_gkr::whir::Whir;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run the extraction scripts first");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn dense(a: &[Goldilocks], b: &[Goldilocks], k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; n];
    for j in 0..n {
        let mut acc = Goldilocks::ZERO;
        for w in 0..k {
            acc = acc + a[w] * b[w * n + j];
        }
        c[j] = acc;
    }
    c
}

fn main() {
    let m = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let a = format!("{m}activations/");
    let w = format!("{m}weights/");

    let add5 = load_i32(&format!("{m}norm_in.bin"));
    let ln_w = load_i32(&format!("{m}norm_w.bin"));
    let rsqrt_table = load_i32(&format!("{m}rsqrt_table_i32.bin"));
    const N_REAL: usize = 264;
    let lnout = load_i32(&format!("{a}layer_norm_512_i32.bin"));
    let gate = load_i32(&format!("{w}val_126_512x1024_i32.bin"));
    let down = load_i32(&format!("{w}val_128_1024x512_i32.bin"));
    let v127 = load_i32(&format!("{a}val_127_1024_i32.bin"));
    let relu = load_i32(&format!("{a}relu_1024_i32.bin"));

    let mut rng = XorShift64::new(0x7ee);
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir19 = Whir::new_testing(19);

    // Norm raw product: raw[i] = (add5[i] - mean) * rstd * ln_w[i]  (scale 2^48),
    // with mean derived from x and rstd bound by an rsqrt lookup.
    let (raw, _mean, _rstd, _s_index) = layer_norm_raw(&add5, &ln_w, N_REAL, &rsqrt_table);
    let c_add5 = commit(&whir9, &add5);
    let c_raw = commit(&whir9, &raw);
    let alpha = rng.field();
    let beta = rng.field();
    assert!(prove_layer_norm(
        &whir9,
        &c_add5,
        &add5,
        &whir9,
        &c_raw,
        &raw,
        &ln_w,
        N_REAL,
        &rsqrt_table,
        alpha,
        beta,
        &mut rng,
    ));

    // gate: layer_norm @ gate = val_127 (raw 2^32).
    let v127_raw = dense(&lnout, &gate, 512, 1024);
    let c_ln = commit(&whir9, &lnout);
    let c_gate = commit(&whir19, &gate);
    let c_v127 = commit(&whir10, &v127_raw);
    assert!(prove_matmul(&whir9, &c_ln, &whir19, &c_gate, &whir10, &c_v127, &lnout, &gate, &v127_raw, 1, 512, 1024, &mut rng));

    // down: relu @ down = val_129 (raw 2^32).
    let v129_raw = dense(&relu, &down, 1024, 512);
    let c_relu = commit(&whir10, &relu);
    let c_down = commit(&whir19, &down);
    let c_v129 = commit(&whir9, &v129_raw);
    assert!(prove_matmul(&whir10, &c_relu, &whir19, &c_down, &whir9, &c_v129, &relu, &down, &v129_raw, 1, 1024, 512, &mut rng));

    let _ = v127;
    println!("FFN tail (LayerNorm -> gate -> ReLU -> down) verified from WHIR commitments");
}
