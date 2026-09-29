//! TimesFM 200M end-to-end through the IR two-phase executor.
//!
//! This is the batch-commit counterpart to `prove_200m_ops.rs`: the forward
//! pass declares ops (and collects every tensor into a BatchBuilder keyed by
//! size/group), then a single `prove()` pass commits tensors in batches and
//! walks the op list opening each tensor by its (size, group, index).

use zkie_gkr::committed::silu_raw;
use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::fixed_point::from_i32;
use zkie_gkr::ir::Exec;
use zkie_gkr::whir::Whir;

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

/// Row-wise RMSNorm decomposed into per-row single-row ops so the BatchBuilder
/// groups all rows (and all other size-`h` tensors) into one commitment.
fn rms_norm_rows_ir(
    ex: &mut Exec,
    x: usize,
    weight: &[Goldilocks],
    n_real: usize,
    n_rows: usize,
) -> usize {
    let x_len = ex.get(x).len();
    let h = x_len / n_rows;
    let zero = vec![from_i32(0); h];
    let mut norm = Vec::with_capacity(x_len);
    for s in 0..n_rows {
        let row = ex.get(x)[s * h..(s + 1) * h].to_vec();
        let row_id = ex.input(row, 0);
        let raw_id = ex.rms_norm(row_id, weight.to_vec(), n_real);
        let norm_id = ex.affine(raw_id, zero.clone(), 32);
        norm.extend_from_slice(ex.get(norm_id));
    }
    ex.input(norm, 0)
}

fn layer_norm_rows_ir(
    ex: &mut Exec,
    x: usize,
    weight: &[Goldilocks],
    bias: &[Goldilocks],
    n_real: usize,
    n_rows: usize,
) -> usize {
    let x_len = ex.get(x).len();
    let h = x_len / n_rows;
    let mut out = Vec::with_capacity(x_len);
    for s in 0..n_rows {
        let row = ex.get(x)[s * h..(s + 1) * h].to_vec();
        let row_id = ex.input(row, 0);
        let raw_id = ex.layer_norm(row_id, weight.to_vec(), n_real);
        let out_id = ex.affine(raw_id, bias.to_vec(), 32);
        out.extend_from_slice(ex.get(out_id));
    }
    ex.input(out, 0)
}

// Model-specific layout helpers (no proving, just reshapes).
fn split_qv(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![from_i32(0); HEADS * SEQ * HDIM_PAD];
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
    let mut out = vec![from_i32(0); HEADS * HDIM_PAD * SEQ];
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
    let mut out = vec![from_i32(0); SEQ * H_PAD];
    for h in 0..HEADS {
        for s in 0..SEQ {
            for j in 0..HDIM {
                out[s * H_PAD + h * HDIM + j] = x[(h * SEQ + s) * HDIM_PAD + j];
            }
        }
    }
    out
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let stack = format!("{base}full_stack_200m/");
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    let silu_table = load_i32(&format!("{base}silu_table_i32.bin"));
    let exp_table = load_i32(&format!("{base}exp_table_i32.bin"));
    let mask = load_i32(&format!("{stack}mask_q_i32.bin"));
    let zero_hd = vec![from_i32(0); SEQ * HDIM_PAD];
    let whir = Whir::new_testing(16);
    let mut rng = XorShift64::new(0x200);
    let n_layers: usize = std::env::var("ZKIE_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(N_LAYERS);

    // ---- Prologue: fresh executor, proven and dropped immediately.
    let mut x_plain: Vec<Goldilocks> = {
        let mut ex = Exec::new();
        ex.set_rsqrt(rsqrt_table.clone());
        let cat = ex.input(load_i32(&format!("{stack}cat_i32.bin")), 0);
        let pro_hid_w = ex.input(load_i32(&format!("{stack}pro_hid_w_i32.bin")), 0);
        let pro_out_w = ex.input(load_i32(&format!("{stack}pro_out_w_i32.bin")), 0);
        let pro_res_w = ex.input(load_i32(&format!("{stack}pro_res_w_i32.bin")), 0);
        let gather = ex.input(load_i32(&format!("{stack}gather_i32.bin")), 0);
        let embedding = ex.input(broadcast_bias(&load_i32(&format!("{stack}embedding_i32.bin")), SEQ), 0);
        let pro_hid_b = broadcast_bias(&load_i32(&format!("{stack}pro_hid_b_i32.bin")), SEQ);
        let pro_out_b = broadcast_bias(&load_i32(&format!("{stack}pro_out_b_i32.bin")), SEQ);
        let pro_res_b = broadcast_bias(&load_i32(&format!("{stack}pro_res_b_i32.bin")), SEQ);

        let hid_raw = ex.matmul(cat, pro_hid_w, SEQ, 64, H_PAD);
        let linear = ex.affine(hid_raw, pro_hid_b, 16);
        let silu_idx = silu_raw(ex.get(linear), &silu_table, SILU_OFFSET).0;
        let silu_t = ex.lookup(&silu_idx, &silu_table, &mut rng);
        let out_raw = ex.matmul(silu_t, pro_out_w, SEQ, H_PAD, H_PAD);
        let linear_1 = ex.affine(out_raw, pro_out_b, 16);
        let res_raw = ex.matmul(cat, pro_res_w, SEQ, 64, H_PAD);
        let linear_2 = ex.affine(res_raw, pro_res_b, 16);
        let add_a = ex.add(linear_1, linear_2);
        let add_1 = ex.add(add_a, gather);
        let x = ex.add(add_1, embedding);
        ex.prove(&whir, &mut rng);
        ex.get(x).to_vec()
    };
    println!("TimesFM 200M (IR): prologue verified");

    // ---- 20 layers, each in a fresh executor, carrying only the residual.
    for li in 0..n_layers {
        let mut ex = Exec::new();
        ex.set_rsqrt(rsqrt_table.clone());
        let x = ex.input(x_plain.clone(), 0);

        let lnw = load_i32(&format!("{stack}L{li}_lnw_i32.bin"));
        let q_w = ex.input(load_i32(&format!("{stack}L{li}_q_w_i32.bin")), 0);
        let k_w = ex.input(load_i32(&format!("{stack}L{li}_k_w_i32.bin")), 0);
        let v_w = ex.input(load_i32(&format!("{stack}L{li}_v_w_i32.bin")), 0);
        let op_w = ex.input(load_i32(&format!("{stack}L{li}_o_proj_w_i32.bin")), 0);
        let mlp_w = load_i32(&format!("{stack}L{li}_mlp_w_i32.bin"));
        let mlp_b = load_i32(&format!("{stack}L{li}_mlp_b_i32.bin"));
        let gate_w = ex.input(load_i32(&format!("{stack}L{li}_gate_w_i32.bin")), 0);
        let down_w = ex.input(load_i32(&format!("{stack}L{li}_down_w_i32.bin")), 0);
        let q_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_q_b_i32.bin")), SEQ);
        let k_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_k_b_i32.bin")), SEQ);
        let v_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_v_b_i32.bin")), SEQ);
        let op_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_o_proj_b_i32.bin")), SEQ);
        let gate_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_gate_b_i32.bin")), SEQ);
        let down_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_down_b_i32.bin")), SEQ);

        // Attention pre-projection.
        let mul9 = rms_norm_rows_ir(&mut ex, x, &lnw, N_REAL, SEQ);
        let q_raw = ex.matmul(mul9, q_w, SEQ, H_PAD, H_PAD);
        let q = ex.affine(q_raw, q_b, 16);
        let k_raw = ex.matmul(mul9, k_w, SEQ, H_PAD, H_PAD);
        let k = ex.affine(k_raw, k_b, 16);
        let v_raw = ex.matmul(mul9, v_w, SEQ, H_PAD, H_PAD);
        let v = ex.affine(v_raw, v_b, 16);

        let qh = split_qv(ex.get(q));
        let kh = split_k(ex.get(k));
        let vh = split_qv(ex.get(v));

        // Per-head QK^T and masked scores; collect whole scores for softmax.
        let mut scores_plain = Vec::with_capacity(HEADS * SEQ * SEQ);
        for h in 0..HEADS {
            let qh_h = ex.input(qh[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD].to_vec(), 0);
            let kh_h = ex.input(kh[h * HDIM_PAD * SEQ..(h + 1) * HDIM_PAD * SEQ].to_vec(), 0);
            let sr = ex.matmul(qh_h, kh_h, SEQ, HDIM_PAD, SEQ);
            let sc = ex.affine(sr, mask.clone(), 16);
            scores_plain.extend_from_slice(ex.get(sc));
        }
        let scores = ex.input(scores_plain, 0);

        // Whole softmax over HEADS * SEQ rows.
        let sm = ex.softmax(scores, &exp_table, EXP_OFFSET, HEADS * SEQ, SEQ, &mut rng);

        // Per-head attention output PV.
        let mut attn_all = Vec::with_capacity(HEADS * SEQ * HDIM_PAD);
        for h in 0..HEADS {
            let sm_h = ex.input(ex.get(sm)[h * SEQ * SEQ..(h + 1) * SEQ * SEQ].to_vec(), 0);
            let vh_h = ex.input(vh[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD].to_vec(), 0);
            let ar = ex.matmul(sm_h, vh_h, SEQ, SEQ, HDIM_PAD);
            let at = ex.affine(ar, zero_hd.clone(), 16);
            attn_all.extend_from_slice(ex.get(at));
        }
        let attn_full = ex.input(concat_attn(&attn_all), 0);

        let op_raw = ex.matmul(attn_full, op_w, SEQ, H_PAD, H_PAD);
        let op = ex.affine(op_raw, op_b, 16);
        let add5 = ex.add(x, op);

        // FFN.
        let ln_out = layer_norm_rows_ir(&mut ex, add5, &mlp_w, &mlp_b, N_REAL, SEQ);
        let gate_raw = ex.matmul(ln_out, gate_w, SEQ, H_PAD, H_PAD);
        let act = ex.relu(gate_raw, gate_b);
        let down_raw = ex.matmul(act, down_w, SEQ, H_PAD, H_PAD);
        let ffn_out = ex.affine(down_raw, down_b, 16);
        let x_new = ex.add(add5, ffn_out);
        ex.prove(&whir, &mut rng);
        x_plain = ex.get(x_new).to_vec();
        println!("layer {li} verified");
    }

    // ---- Epilogue: horizon FFN output head (fresh executor).
    {
        let mut ex = Exec::new();
        ex.set_rsqrt(rsqrt_table.clone());
        let x = ex.input(x_plain.clone(), 0);
        let hid_w = ex.input(load_i32(&format!("{stack}head_hid_w_i32.bin")), 0);
        let out_w = ex.input(load_i32(&format!("{stack}head_out_w_i32.bin")), 0);
        let res_w = ex.input(load_i32(&format!("{stack}head_res_w_i32.bin")), 0);
        let hid_b = broadcast_bias(&load_i32(&format!("{stack}head_hid_b_i32.bin")), SEQ);
        let out_b = broadcast_bias(&load_i32(&format!("{stack}head_out_b_i32.bin")), SEQ);
        let res_b = broadcast_bias(&load_i32(&format!("{stack}head_res_b_i32.bin")), SEQ);
        let scale_bytes = std::fs::read(format!("{stack}scale_i64.bin")).expect("scale file");
        let scale_q = i64::from_le_bytes(scale_bytes[0..8].try_into().unwrap());
        let bias_q = i64::from_le_bytes(scale_bytes[8..16].try_into().unwrap());

        let hid_raw = ex.matmul(x, hid_w, SEQ, H_PAD, H_PAD);
        let lin31 = ex.affine(hid_raw, hid_b, 16);
        let silu1_idx = silu_raw(ex.get(lin31), &silu_table, SILU_OFFSET).0;
        let silu1_t = ex.lookup(&silu1_idx, &silu_table, &mut rng);
        let out_raw = ex.matmul(silu1_t, out_w, SEQ, H_PAD, H_PAD);
        let lin32 = ex.affine(out_raw, out_b, 16);
        let res_raw = ex.matmul(x, res_w, SEQ, H_PAD, H_PAD);
        let lin33 = ex.affine(res_raw, res_b, 16);
        let add31 = ex.add(lin32, lin33);
        let _output_ts = ex.scale(add31, scale_q, vec![from_i32(bias_q as i32); SEQ * H_PAD]);
        ex.prove(&whir, &mut rng);
    }
    println!("TimesFM 200M (IR): prologue + 20 layers + output head verified");
}
