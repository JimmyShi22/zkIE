//! Prove the whole TimesFM 8M transformer stack (7 layers) as one *chained*
//! circuit. Each layer is `RMSNorm -> V -> o_proj -> residual -> LayerNorm ->
//! gate -> ReLU -> down -> residual`, and the residual stream is chained across
//! layers, so every intermediate tensor's output commitment is the next stage's
//! input commitment. The fixed-point computation matches the ONNX float
//! reference within 8.4e-4 (see `simulate_full_stack.py`).

use zkie_gkr::committed::{
    affine_raw, commit, layer_norm_raw, prove_add, prove_affine, prove_layer_norm, prove_matmul,
    prove_lookup, prove_relu, prove_rms_norm, prove_scale, rms_norm_raw, scale_raw, silu_raw,
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
    let whir5 = Whir::new_testing(5);
    let whir6 = Whir::new_testing(6);
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir11 = Whir::new_testing(11);
    let whir15 = Whir::new_testing(15);
    let whir16 = Whir::new_testing(16);
    let whir18 = Whir::new_testing(18);
    let whir19 = Whir::new_testing(19);
    let whir20 = Whir::new_testing(20);
    let whir21 = Whir::new_testing(21);

    let silu_table = load_i32(&format!("{base}silu_table_i32.bin"));
    const SILU_OFFSET: u32 = 1 << 19;

    // Input embedding: cat = [LayerNorm(input_ts) (32), padding mask (32 zeros)].
    let input_ts = load_i32(&format!("{stack}input_ts_i32.bin"));
    let input_pad = load_i32(&format!("{stack}input_pad_i32.bin"));
    let w_identity = vec![from_i32(65536); 32];
    let (raw_norm, _m, _r, _s) = layer_norm_raw(&input_ts, &w_identity, 32, &rsqrt_table);
    let zero32 = vec![from_i32(0); 32];
    let normalized = affine_raw(&raw_norm, &zero32, 32, false);
    let mut cat = normalized.clone();
    cat.extend_from_slice(&input_pad);

    let c_ts = commit(&whir5, &input_ts);
    let c_raw_norm = commit(&whir5, &raw_norm);
    assert!(prove_layer_norm(
        &whir5, &c_ts, &input_ts, &whir5, &c_raw_norm, &raw_norm, &w_identity, 32, &rsqrt_table,
        rng.field(), rng.field(), &mut rng,
    ));
    let c_norm = commit(&whir5, &normalized);
    assert!(prove_affine(
        &whir5, &c_raw_norm, &raw_norm, &whir5, &c_norm, &normalized, &zero32, 32, &mut rng,
    ));

    // Prologue: cat -> SiLU FFN -> add, then +gather +embedding (freq) -> add_2.
    let pro_hid_w = load_i32(&format!("{stack}pro_hid_w_i32.bin"));
    let pro_hid_b = load_i32(&format!("{stack}pro_hid_b_i32.bin"));
    let pro_out_w = load_i32(&format!("{stack}pro_out_w_i32.bin"));
    let pro_out_b = load_i32(&format!("{stack}pro_out_b_i32.bin"));
    let pro_res_w = load_i32(&format!("{stack}pro_res_w_i32.bin"));
    let pro_res_b = load_i32(&format!("{stack}pro_res_b_i32.bin"));
    let gather = load_i32(&format!("{stack}gather_i32.bin"));
    let embedding = load_i32(&format!("{stack}embedding_i32.bin"));

    let hid_raw = dense(&cat, &pro_hid_w, 64, 1024);
    let linear = affine_raw(&hid_raw, &pro_hid_b, 16, false);
    let (silu_idx, silu) = silu_raw(&linear, &silu_table, SILU_OFFSET);
    let out_raw = dense(&silu, &pro_out_w, 1024, 512);
    let linear_1 = affine_raw(&out_raw, &pro_out_b, 16, false);
    let res_raw = dense(&cat, &pro_res_w, 64, 512);
    let linear_2 = affine_raw(&res_raw, &pro_res_b, 16, false);
    let add = add_vec(&linear_1, &linear_2);
    let add_1 = add_vec(&add, &gather);
    let add_2 = add_vec(&add_1, &embedding);

    let c_cat = commit(&whir6, &cat);
    let c_hidw = commit(&whir16, &pro_hid_w);
    let c_hidraw = commit(&whir10, &hid_raw);
    assert!(prove_matmul(&whir6, &c_cat, &whir16, &c_hidw, &whir10, &c_hidraw, &cat, &pro_hid_w, &hid_raw, 1, 64, 1024, &mut rng));
    let c_linear = commit(&whir10, &linear);
    assert!(prove_affine(&whir10, &c_hidraw, &hid_raw, &whir10, &c_linear, &linear, &pro_hid_b, 16, &mut rng));
    let idx_field: Vec<Goldilocks> = silu_idx.iter().map(|&i| from_i32(i as i32)).collect();
    let c_idx = commit(&whir10, &idx_field);
    let c_silu = commit(&whir10, &silu);
    assert!(prove_lookup(&whir10, &c_idx, &idx_field, &whir10, &c_silu, &silu, &silu_idx, &silu_table, rng.field(), rng.field(), &mut rng));
    let c_outw = commit(&whir19, &pro_out_w);
    let c_outraw = commit(&whir9, &out_raw);
    assert!(prove_matmul(&whir10, &c_silu, &whir19, &c_outw, &whir9, &c_outraw, &silu, &pro_out_w, &out_raw, 1, 1024, 512, &mut rng));
    let c_lin1 = commit(&whir9, &linear_1);
    assert!(prove_affine(&whir9, &c_outraw, &out_raw, &whir9, &c_lin1, &linear_1, &pro_out_b, 16, &mut rng));
    let c_resw = commit(&whir15, &pro_res_w);
    let c_resraw = commit(&whir9, &res_raw);
    assert!(prove_matmul(&whir6, &c_cat, &whir15, &c_resw, &whir9, &c_resraw, &cat, &pro_res_w, &res_raw, 1, 64, 512, &mut rng));
    let c_lin2 = commit(&whir9, &linear_2);
    assert!(prove_affine(&whir9, &c_resraw, &res_raw, &whir9, &c_lin2, &linear_2, &pro_res_b, 16, &mut rng));
    let c_add = commit(&whir9, &add);
    assert!(prove_add(&whir9, &c_lin1, &whir9, &c_lin2, &whir9, &c_add, 512, &mut rng));
    let c_gather = commit(&whir9, &gather);
    let c_add1 = commit(&whir9, &add_1);
    assert!(prove_add(&whir9, &c_add, &whir9, &c_gather, &whir9, &c_add1, 512, &mut rng));
    let c_emb = commit(&whir9, &embedding);
    let c_add2 = commit(&whir9, &add_2);
    assert!(prove_add(&whir9, &c_add1, &whir9, &c_emb, &whir9, &c_add2, 512, &mut rng));

    let mut x = add_2;

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
        assert!(prove_rms_norm(&whir9, &c_x, &x, &whir9, &c_rms, &rms_raw, &lnw, N_REAL, &rsqrt_table, rng.field(), rng.field(), &mut rng));
        let c_mul9 = commit(&whir9, &mul9);
        assert!(prove_affine(&whir9, &c_rms, &rms_raw, &whir9, &c_mul9, &mul9, &zero_bias, 32, &mut rng));
        let c_vw = commit(&whir18, &v_w);
        let c_vraw = commit(&whir9, &v_raw);
        assert!(prove_matmul(&whir9, &c_mul9, &whir18, &c_vw, &whir9, &c_vraw, &mul9, &v_w, &v_raw, 1, 512, 512, &mut rng));
        let c_lin3v = commit(&whir9, &lin3v);
        assert!(prove_affine(&whir9, &c_vraw, &v_raw, &whir9, &c_lin3v, &lin3v, &v_b, 16, &mut rng));
        let c_opw = commit(&whir18, &op_w);
        let c_opraw = commit(&whir9, &op_raw);
        assert!(prove_matmul(&whir9, &c_lin3v, &whir18, &c_opw, &whir9, &c_opraw, &lin3v, &op_w, &op_raw, 1, 512, 512, &mut rng));
        let c_lin4 = commit(&whir9, &lin4);
        assert!(prove_affine(&whir9, &c_opraw, &op_raw, &whir9, &c_lin4, &lin4, &op_b, 16, &mut rng));
        let c_add5 = commit(&whir9, &add5);
        assert!(prove_add(&whir9, &c_x, &whir9, &c_lin4, &whir9, &c_add5, 512, &mut rng));

        // FFN half: LayerNorm -> gate -> ReLU -> down -> residual.
        let (ln_raw, _, _, _) = layer_norm_raw(&add5, &mlp_w, N_REAL, &rsqrt_table);
        let ln_out = affine_raw(&ln_raw, &mlp_b, 32, false);
        let gate_raw = dense(&ln_out, &gate_w, 512, 1024);
        let relu = affine_raw(&gate_raw, &gate_b, 16, true);
        let down_raw = dense(&relu, &down_w, 1024, 512);
        let ffn_out = affine_raw(&down_raw, &down_b, 16, false);
        let next_x = add_vec(&add5, &ffn_out);

        let c_lnraw = commit(&whir9, &ln_raw);
        assert!(prove_layer_norm(&whir9, &c_add5, &add5, &whir9, &c_lnraw, &ln_raw, &mlp_w, N_REAL, &rsqrt_table, rng.field(), rng.field(), &mut rng));
        let c_lnout = commit(&whir9, &ln_out);
        assert!(prove_affine(&whir9, &c_lnraw, &ln_raw, &whir9, &c_lnout, &ln_out, &mlp_b, 32, &mut rng));
        let c_gatew = commit(&whir19, &gate_w);
        let c_gateraw = commit(&whir10, &gate_raw);
        assert!(prove_matmul(&whir9, &c_lnout, &whir19, &c_gatew, &whir10, &c_gateraw, &ln_out, &gate_w, &gate_raw, 1, 512, 1024, &mut rng));
        let c_relu = commit(&whir10, &relu);
        assert!(prove_relu(&whir10, &c_gateraw, &gate_raw, &whir10, &c_relu, &relu, &gate_b, &mut rng));
        let c_downw = commit(&whir19, &down_w);
        let c_downraw = commit(&whir9, &down_raw);
        assert!(prove_matmul(&whir10, &c_relu, &whir19, &c_downw, &whir9, &c_downraw, &relu, &down_w, &down_raw, 1, 1024, 512, &mut rng));
        let c_ffnout = commit(&whir9, &ffn_out);
        assert!(prove_affine(&whir9, &c_downraw, &down_raw, &whir9, &c_ffnout, &ffn_out, &down_b, 16, &mut rng));
        let c_next = commit(&whir9, &next_x);
        assert!(prove_add(&whir9, &c_add5, &whir9, &c_ffnout, &whir9, &c_next, 512, &mut rng));

        x = next_x;
        println!("layer {li} block verified");
    }

    // Epilogue (horizon FFN output head): hidden -> SiLU -> output + residual ->
    // rescale. `x` is now add_30 (layer-6 residual).
    let hid_w = load_i32(&format!("{stack}head_hid_w_i32.bin"));
    let hid_b = load_i32(&format!("{stack}head_hid_b_i32.bin"));
    let out_w = load_i32(&format!("{stack}head_out_w_i32.bin"));
    let out_b = load_i32(&format!("{stack}head_out_b_i32.bin"));
    let res_w = load_i32(&format!("{stack}head_res_w_i32.bin"));
    let res_b = load_i32(&format!("{stack}head_res_b_i32.bin"));
    let scale_bytes = std::fs::read(format!("{stack}scale_i64.bin")).expect("scale file");
    let scale_q = i64::from_le_bytes(scale_bytes[0..8].try_into().unwrap());
    let bias_q = i64::from_le_bytes(scale_bytes[8..16].try_into().unwrap());

    let hid_raw = dense(&x, &hid_w, 512, 1024);
    let lin31 = affine_raw(&hid_raw, &hid_b, 16, false);
    let (silu_idx, silu1) = silu_raw(&lin31, &silu_table, SILU_OFFSET);
    let out_raw = dense(&silu1, &out_w, 1024, 2048);
    let lin32 = affine_raw(&out_raw, &out_b, 16, false);
    let res_raw = dense(&x, &res_w, 512, 2048);
    let lin33 = affine_raw(&res_raw, &res_b, 16, false);
    let add31 = add_vec(&lin32, &lin33);
    let bias_bcast = vec![from_i32(bias_q as i32); 2048];
    let output_ts = scale_raw(&add31, scale_q, &bias_bcast);

    let c_x = commit(&whir9, &x);
    let c_hidw = commit(&whir19, &hid_w);
    let c_hidraw = commit(&whir10, &hid_raw);
    assert!(prove_matmul(&whir9, &c_x, &whir19, &c_hidw, &whir10, &c_hidraw, &x, &hid_w, &hid_raw, 1, 512, 1024, &mut rng));
    let c_lin31 = commit(&whir10, &lin31);
    assert!(prove_affine(&whir10, &c_hidraw, &hid_raw, &whir10, &c_lin31, &lin31, &hid_b, 16, &mut rng));
    let idx_field: Vec<Goldilocks> = silu_idx.iter().map(|&i| from_i32(i as i32)).collect();
    let c_idx = commit(&whir10, &idx_field);
    let c_silu = commit(&whir10, &silu1);
    assert!(prove_lookup(&whir10, &c_idx, &idx_field, &whir10, &c_silu, &silu1, &silu_idx, &silu_table, rng.field(), rng.field(), &mut rng));
    let c_outw = commit(&whir21, &out_w);
    let c_outraw = commit(&whir11, &out_raw);
    assert!(prove_matmul(&whir10, &c_silu, &whir21, &c_outw, &whir11, &c_outraw, &silu1, &out_w, &out_raw, 1, 1024, 2048, &mut rng));
    let c_lin32 = commit(&whir11, &lin32);
    assert!(prove_affine(&whir11, &c_outraw, &out_raw, &whir11, &c_lin32, &lin32, &out_b, 16, &mut rng));
    let c_resw = commit(&whir20, &res_w);
    let c_resraw = commit(&whir11, &res_raw);
    assert!(prove_matmul(&whir9, &c_x, &whir20, &c_resw, &whir11, &c_resraw, &x, &res_w, &res_raw, 1, 512, 2048, &mut rng));
    let c_lin33 = commit(&whir11, &lin33);
    assert!(prove_affine(&whir11, &c_resraw, &res_raw, &whir11, &c_lin33, &lin33, &res_b, 16, &mut rng));
    let c_add31 = commit(&whir11, &add31);
    assert!(prove_add(&whir11, &c_lin32, &whir11, &c_lin33, &whir11, &c_add31, 2048, &mut rng));
    let c_out = commit(&whir11, &output_ts);
    assert!(prove_scale(&whir11, &c_add31, &add31, &whir11, &c_out, &output_ts, scale_q, &bias_bcast, &mut rng));

    println!("TimesFM 8M: prologue + 7-layer stack + output head verified from WHIR commitments");
}
