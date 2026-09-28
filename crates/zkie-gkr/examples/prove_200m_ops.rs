//! TimesFM 200M end-to-end through the unified ops interface.
//! This is the reference program an AI-generated per-model implementation would
//! compile against: load weights, run forward pass, and let each op prove itself.

use zkie_gkr::committed::silu_raw;
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_gkr::fixed_point::from_i32;
use zkie_gkr::ops::{
    add, affine, layer_norm_rows, lookup, matmul, relu, rms_norm_rows, scale,
    softmax_rows, Ctx,
};

const H_PAD: usize = 2048;
const SEQ: usize = 16;
const HEADS: usize = 16;
const HDIM: usize = 80;
const HDIM_PAD: usize = 128;
const N_REAL: usize = 1280;
const N_LAYERS: usize = 20;
const SILU_OFFSET: u32 = 1 << 19;
const EXP_OFFSET: u32 = 1 << 21;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("missing extracted data");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn broadcast_bias(bias: &[Goldilocks], seq: usize) -> Vec<Goldilocks> {
    let h = bias.len();
    let mut out = Vec::with_capacity(seq * h);
    for _ in 0..seq {
        out.extend_from_slice(bias);
    }
    out
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let stack = format!("{base}full_stack_200m/");
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    let silu_table = load_i32(&format!("{base}silu_table_i32.bin"));
    let exp_table = load_i32(&format!("{base}exp_table_i32.bin"));

    let mut ctx = Ctx::new(0x200);

    // ---- Prologue: RevIN embedding is precomputed; proof starts from cat.
    let cat = ctx.commit(load_i32(&format!("{stack}cat_i32.bin")));
    let pro_hid_w = ctx.commit(load_i32(&format!("{stack}pro_hid_w_i32.bin")));
    let pro_hid_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}pro_hid_b_i32.bin")), SEQ));
    let pro_out_w = ctx.commit(load_i32(&format!("{stack}pro_out_w_i32.bin")));
    let pro_out_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}pro_out_b_i32.bin")), SEQ));
    let pro_res_w = ctx.commit(load_i32(&format!("{stack}pro_res_w_i32.bin")));
    let pro_res_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}pro_res_b_i32.bin")), SEQ));
    let gather = ctx.commit(load_i32(&format!("{stack}gather_i32.bin")));
    let embedding = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}embedding_i32.bin")), SEQ));

    let hid_raw = matmul(&mut ctx, &cat, &pro_hid_w, SEQ, 64, H_PAD);
    let linear = affine(&mut ctx, &hid_raw, &pro_hid_b.plain, 16, false);
    let (silu_idx, silu) = silu_raw(&linear.plain, &silu_table, SILU_OFFSET);
    let silu_t = lookup(&mut ctx, &silu_idx, &silu, &silu_table);
    let out_raw = matmul(&mut ctx, &silu_t, &pro_out_w, SEQ, H_PAD, H_PAD);
    let linear_1 = affine(&mut ctx, &out_raw, &pro_out_b.plain, 16, false);
    let res_raw = matmul(&mut ctx, &cat, &pro_res_w, SEQ, 64, H_PAD);
    let linear_2 = affine(&mut ctx, &res_raw, &pro_res_b.plain, 16, false);
    let add_a = add(&mut ctx, &linear_1, &linear_2);
    let add_1 = add(&mut ctx, &add_a, &gather);
    let mut x = add(&mut ctx, &add_1, &embedding);
    println!("TimesFM 200M (ops): prologue verified");

    let mask = ctx.commit(load_i32(&format!("{stack}mask_q_i32.bin")));
    let zero_hd = ctx.commit(vec![from_i32(0); SEQ * HDIM_PAD]);

    for li in 0..N_LAYERS {
        let lnw = ctx.commit(load_i32(&format!("{stack}L{li}_lnw_i32.bin")));
        let q_w = ctx.commit(load_i32(&format!("{stack}L{li}_q_w_i32.bin")));
        let k_w = ctx.commit(load_i32(&format!("{stack}L{li}_k_w_i32.bin")));
        let v_w = ctx.commit(load_i32(&format!("{stack}L{li}_v_w_i32.bin")));
        let q_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_q_b_i32.bin")), SEQ));
        let k_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_k_b_i32.bin")), SEQ));
        let v_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_v_b_i32.bin")), SEQ));
        let op_w = ctx.commit(load_i32(&format!("{stack}L{li}_o_proj_w_i32.bin")));
        let op_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_o_proj_b_i32.bin")), SEQ));
        let mlp_w = ctx.commit(load_i32(&format!("{stack}L{li}_mlp_w_i32.bin")));
        let mlp_b = ctx.commit(load_i32(&format!("{stack}L{li}_mlp_b_i32.bin")));
        let gate_w = ctx.commit(load_i32(&format!("{stack}L{li}_gate_w_i32.bin")));
        let gate_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_gate_b_i32.bin")), SEQ));
        let down_w = ctx.commit(load_i32(&format!("{stack}L{li}_down_w_i32.bin")));
        let down_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_down_b_i32.bin")), SEQ));

        // Attention pre-projection.
        let mul9 = rms_norm_rows(&mut ctx, &x, &lnw.plain, N_REAL, &rsqrt_table, SEQ);
        let q_raw = matmul(&mut ctx, &mul9, &q_w, SEQ, H_PAD, H_PAD);
        let q = affine(&mut ctx, &q_raw, &q_b.plain, 16, false);
        let k_raw = matmul(&mut ctx, &mul9, &k_w, SEQ, H_PAD, H_PAD);
        let k = affine(&mut ctx, &k_raw, &k_b.plain, 16, false);
        let v_raw = matmul(&mut ctx, &mul9, &v_w, SEQ, H_PAD, H_PAD);
        let v = affine(&mut ctx, &v_raw, &v_b.plain, 16, false);

        let qh = split_qv(&q.plain);
        let kh = split_k(&k.plain);
        let vh = split_qv(&v.plain);

        // Per-head QK^T and masked scores; collect whole scores for softmax.
        let mut scores_plain = Vec::with_capacity(HEADS * SEQ * SEQ);
        for h in 0..HEADS {
            let qh_h = ctx.commit(qh[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD].to_vec());
            let kh_h = ctx.commit(kh[h * HDIM_PAD * SEQ..(h + 1) * HDIM_PAD * SEQ].to_vec());
            let sr = matmul(&mut ctx, &qh_h, &kh_h, SEQ, HDIM_PAD, SEQ);
            let sc = affine(&mut ctx, &sr, &mask.plain, 16, false);
            scores_plain.extend_from_slice(&sc.plain);
        }
        let scores = ctx.commit(scores_plain);

        // Whole softmax over HEADS * SEQ rows.
        let sm = softmax_rows(&mut ctx, &scores, &exp_table, EXP_OFFSET, HEADS * SEQ, SEQ);

        // Per-head attention output PV.
        let mut attn_all = Vec::with_capacity(HEADS * SEQ * HDIM_PAD);
        for h in 0..HEADS {
            let sm_h = ctx.commit(sm.plain[h * SEQ * SEQ..(h + 1) * SEQ * SEQ].to_vec());
            let vh_h = ctx.commit(vh[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD].to_vec());
            let ar = matmul(&mut ctx, &sm_h, &vh_h, SEQ, SEQ, HDIM_PAD);
            let at = affine(&mut ctx, &ar, &zero_hd.plain, 16, false);
            attn_all.extend_from_slice(&at.plain);
        }
        let attn_full = ctx.commit(concat_attn(&attn_all));

        let op_raw = matmul(&mut ctx, &attn_full, &op_w, SEQ, H_PAD, H_PAD);
        let op = affine(&mut ctx, &op_raw, &op_b.plain, 16, false);
        let add5 = add(&mut ctx, &x, &op);

        // FFN.
        let ln_out = layer_norm_rows(&mut ctx, &add5, &mlp_w.plain, &mlp_b.plain, N_REAL, &rsqrt_table, SEQ);
        let gate_raw = matmul(&mut ctx, &ln_out, &gate_w, SEQ, H_PAD, H_PAD);
        let act = relu(&mut ctx, &gate_raw, &gate_b.plain);
        let down_raw = matmul(&mut ctx, &act, &down_w, SEQ, H_PAD, H_PAD);
        let ffn_out = affine(&mut ctx, &down_raw, &down_b.plain, 16, false);
        x = add(&mut ctx, &add5, &ffn_out);
        println!("layer {li} verified");
    }

    // ---- Epilogue: horizon FFN output head.
    let hid_w = ctx.commit(load_i32(&format!("{stack}head_hid_w_i32.bin")));
    let hid_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}head_hid_b_i32.bin")), SEQ));
    let out_w = ctx.commit(load_i32(&format!("{stack}head_out_w_i32.bin")));
    let out_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}head_out_b_i32.bin")), SEQ));
    let res_w = ctx.commit(load_i32(&format!("{stack}head_res_w_i32.bin")));
    let res_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}head_res_b_i32.bin")), SEQ));
    let scale_bytes = std::fs::read(format!("{stack}scale_i64.bin")).expect("scale file");
    let scale_q = i64::from_le_bytes(scale_bytes[0..8].try_into().unwrap());
    let bias_q = i64::from_le_bytes(scale_bytes[8..16].try_into().unwrap());

    let hid_raw = matmul(&mut ctx, &x, &hid_w, SEQ, H_PAD, H_PAD);
    let lin31 = affine(&mut ctx, &hid_raw, &hid_b.plain, 16, false);
    let (silu_idx, silu1) = silu_raw(&lin31.plain, &silu_table, SILU_OFFSET);
    let silu1_t = lookup(&mut ctx, &silu_idx, &silu1, &silu_table);
    let out_raw = matmul(&mut ctx, &silu1_t, &out_w, SEQ, H_PAD, H_PAD);
    let lin32 = affine(&mut ctx, &out_raw, &out_b.plain, 16, false);
    let res_raw = matmul(&mut ctx, &x, &res_w, SEQ, H_PAD, H_PAD);
    let lin33 = affine(&mut ctx, &res_raw, &res_b.plain, 16, false);
    let add31 = add(&mut ctx, &lin32, &lin33);
    let bias_bcast = ctx.commit(vec![from_i32(bias_q as i32); SEQ * H_PAD]);
    let _output_ts = scale(&mut ctx, &add31, scale_q, &bias_bcast.plain);

    println!("TimesFM 200M (ops): prologue + 20 layers + output head verified");
}

// Model-specific layout helpers (no proving, just reshapes).
fn split_qv(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; HEADS * SEQ * HDIM_PAD];
    for h in 0..HEADS {
        for s in 0..SEQ {
            for j in 0..HDIM {
                out[(h * SEQ + s) * HDIM_PAD + j] = x[s * H_PAD + h * HDIM + j];
            }
        }
    }
    out
}

fn split_k(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; HEADS * HDIM_PAD * SEQ];
    for h in 0..HEADS {
        for j in 0..HDIM {
            for s in 0..SEQ {
                out[(h * HDIM_PAD + j) * SEQ + s] = x[s * H_PAD + h * HDIM + j];
            }
        }
    }
    out
}

fn concat_attn(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; SEQ * H_PAD];
    for h in 0..HEADS {
        for s in 0..SEQ {
            for j in 0..HDIM {
                out[s * H_PAD + h * HDIM + j] = x[(h * SEQ + s) * HDIM_PAD + j];
            }
        }
    }
    out
}
