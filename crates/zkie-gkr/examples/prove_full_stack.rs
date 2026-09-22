//! Prove the whole TimesFM 8M transformer stack (7 layers) as one *chained*
//! circuit. Each layer is `RMSNorm -> V -> o_proj -> residual -> LayerNorm ->
//! gate -> ReLU -> down -> residual`, and the residual stream is chained across
//! layers, so every intermediate tensor's output commitment is the next stage's
//! input commitment. The fixed-point computation matches the ONNX float
//! reference within 8.4e-4 (see `simulate_full_stack.py`).

use zkie_gkr::committed::{
    affine_raw, commit, layer_norm_raw, prove_add, prove_affine, prove_layer_norm, prove_matmul,
    prove_relu, prove_rms_norm, rms_norm_raw,
};
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::from_i32;
use zkie_gkr::whir::Whir;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run extract_full_stack.py first");
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

fn add_vec(a: &[Goldilocks], b: &[Goldilocks]) -> Vec<Goldilocks> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let stack = format!("{base}full_stack/");
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    const N_REAL: usize = 264;
    let zero_bias = vec![from_i32(0); 512];

    let mut rng = XorShift64::new(0x7ee);
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir18 = Whir::new_testing(18);
    let whir19 = Whir::new_testing(19);

    let mut x = load_i32(&format!("{stack}x_i32.bin"));

    for li in 0..7 {
        let lnw = load_i32(&format!("{stack}L{li}_lnw_i32.bin"));
        let v_w = load_i32(&format!("{stack}L{li}_v_w_i32.bin"));
        let v_b = load_i32(&format!("{stack}L{li}_v_b_i32.bin"));
        let op_w = load_i32(&format!("{stack}L{li}_op_w_i32.bin"));
        let op_b = load_i32(&format!("{stack}L{li}_op_b_i32.bin"));
        let mlp_w = load_i32(&format!("{stack}L{li}_mlp_w_i32.bin"));
        let mlp_b = load_i32(&format!("{stack}L{li}_mlp_b_i32.bin"));
        let gate_w = load_i32(&format!("{stack}L{li}_gate_w_i32.bin"));
        let gate_b = load_i32(&format!("{stack}L{li}_gate_b_i32.bin"));
        let down_w = load_i32(&format!("{stack}L{li}_down_w_i32.bin"));
        let down_b = load_i32(&format!("{stack}L{li}_down_b_i32.bin"));

        // Attention half: RMSNorm -> V -> o_proj -> residual.
        let (rms_raw, _, _) = rms_norm_raw(&x, &lnw, N_REAL, &rsqrt_table);
        let mul9 = affine_raw(&rms_raw, &zero_bias, 32, false);
        let v_raw = dense(&mul9, &v_w, 512, 512);
        let lin3v = affine_raw(&v_raw, &v_b, 16, false);
        let op_raw = dense(&lin3v, &op_w, 512, 512);
        let lin4 = affine_raw(&op_raw, &op_b, 16, false);
        let add5 = add_vec(&x, &lin4);

        let c_x = commit(&whir9, &x);
        let c_rms = commit(&whir9, &rms_raw);
        assert!(prove_rms_norm(&whir9, &c_x, &whir9, &c_rms, &lnw, N_REAL, &rsqrt_table, rng.field(), rng.field()));
        let c_mul9 = commit(&whir9, &mul9);
        assert!(prove_affine(&whir9, &c_rms, &whir9, &c_mul9, &zero_bias, 32, false));
        let c_vw = commit(&whir18, &v_w);
        let c_vraw = commit(&whir9, &v_raw);
        assert!(prove_matmul(&whir9, &c_mul9, &whir18, &c_vw, &whir9, &c_vraw, &mul9, &v_w, &v_raw, 512, 512, &mut rng));
        let c_lin3v = commit(&whir9, &lin3v);
        assert!(prove_affine(&whir9, &c_vraw, &whir9, &c_lin3v, &v_b, 16, false));
        let c_opw = commit(&whir18, &op_w);
        let c_opraw = commit(&whir9, &op_raw);
        assert!(prove_matmul(&whir9, &c_lin3v, &whir18, &c_opw, &whir9, &c_opraw, &lin3v, &op_w, &op_raw, 512, 512, &mut rng));
        let c_lin4 = commit(&whir9, &lin4);
        assert!(prove_affine(&whir9, &c_opraw, &whir9, &c_lin4, &op_b, 16, false));
        let c_add5 = commit(&whir9, &add5);
        assert!(prove_add(&whir9, &c_x, &whir9, &c_lin4, &whir9, &c_add5, 512));

        // FFN half: LayerNorm -> gate -> ReLU -> down -> residual.
        let (ln_raw, _, _, _) = layer_norm_raw(&add5, &mlp_w, N_REAL, &rsqrt_table);
        let ln_out = affine_raw(&ln_raw, &mlp_b, 32, false);
        let gate_raw = dense(&ln_out, &gate_w, 512, 1024);
        let relu = affine_raw(&gate_raw, &gate_b, 16, true);
        let down_raw = dense(&relu, &down_w, 1024, 512);
        let ffn_out = affine_raw(&down_raw, &down_b, 16, false);
        let next_x = add_vec(&add5, &ffn_out);

        let c_lnraw = commit(&whir9, &ln_raw);
        assert!(prove_layer_norm(&whir9, &c_add5, &whir9, &c_lnraw, &mlp_w, N_REAL, &rsqrt_table, rng.field(), rng.field()));
        let c_lnout = commit(&whir9, &ln_out);
        assert!(prove_affine(&whir9, &c_lnraw, &whir9, &c_lnout, &mlp_b, 32, false));
        let c_gatew = commit(&whir19, &gate_w);
        let c_gateraw = commit(&whir10, &gate_raw);
        assert!(prove_matmul(&whir9, &c_lnout, &whir19, &c_gatew, &whir10, &c_gateraw, &ln_out, &gate_w, &gate_raw, 512, 1024, &mut rng));
        let c_relu = commit(&whir10, &relu);
        assert!(prove_relu(&whir10, &c_gateraw, &whir10, &c_relu, &gate_b));
        let c_downw = commit(&whir19, &down_w);
        let c_downraw = commit(&whir9, &down_raw);
        assert!(prove_matmul(&whir10, &c_relu, &whir19, &c_downw, &whir9, &c_downraw, &relu, &down_w, &down_raw, 1024, 512, &mut rng));
        let c_ffnout = commit(&whir9, &ffn_out);
        assert!(prove_affine(&whir9, &c_downraw, &whir9, &c_ffnout, &down_b, 16, false));
        let c_next = commit(&whir9, &next_x);
        assert!(prove_add(&whir9, &c_add5, &whir9, &c_ffnout, &whir9, &c_next, 512));

        x = next_x;
        println!("layer {li} block verified");
    }

    println!("TimesFM 8M full stack: 7 layers x (attention + FFN) verified from WHIR commitments");
}
