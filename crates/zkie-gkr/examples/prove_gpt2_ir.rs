//! GPT-2 124M through the IR two-phase executor (batch-commit + batch-open).

use zkie_gkr::committed::silu_raw;
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::{from_i32, to_i64};
use zkie_gkr::ir::Exec;
use zkie_gkr::whir::Whir;

const H_PAD: usize = 1024;
const FFN_PAD: usize = 4096;
const VOCAB_PAD: usize = 65536;
const VOCAB: usize = 50257;
const SEQ: usize = 16;
const HEADS: usize = 12;
const HEADS_PAD: usize = 16;
const HDIM: usize = 64;
const N_REAL: usize = 768;
const N_LAYERS: usize = 12;
const GELU_OFFSET: u32 = 1 << 23;
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

fn split_heads_q(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; HEADS_PAD * SEQ * HDIM];
    for h in 0..HEADS_PAD {
        for s in 0..SEQ {
            for j in 0..HDIM {
                out[(h * SEQ + s) * HDIM + j] = x[s * H_PAD + h * HDIM + j];
            }
        }
    }
    out
}

fn split_heads_k(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; HEADS_PAD * HDIM * SEQ];
    for h in 0..HEADS_PAD {
        for j in 0..HDIM {
            for s in 0..SEQ {
                out[(h * HDIM + j) * SEQ + s] = x[s * H_PAD + h * HDIM + j];
            }
        }
    }
    out
}

fn concat_heads(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; SEQ * H_PAD];
    for h in 0..HEADS {
        for s in 0..SEQ {
            for j in 0..HDIM {
                out[s * H_PAD + h * HDIM + j] = x[(h * SEQ + s) * HDIM + j];
            }
        }
    }
    out
}

fn layer_norm_rows_ir(
    ex: &mut Exec,
    x: usize,
    weight: &[Goldilocks],
    bias: &[Goldilocks],
    n_real: usize,
    n_rows: usize,
) -> usize {
    let h = ex.get(x).len() / n_rows;
    let mut out = Vec::with_capacity(ex.get(x).len());
    for s in 0..n_rows {
        let row = ex.get(x)[s * h..(s + 1) * h].to_vec();
        let row_id = ex.input(row, 0);
        let raw_id = ex.layer_norm(row_id, weight.to_vec(), n_real);
        let out_id = ex.affine(raw_id, bias.to_vec(), 32);
        out.extend_from_slice(ex.get(out_id));
    }
    ex.input(out, 0)
}

fn forward_layer(
    li: usize,
    x_plain: &[Goldilocks],
    rng: &mut XorShift64,
    rsqrt_table: &[Goldilocks],
    exp_table: &[Goldilocks],
    gelu_table: &[Goldilocks],
    mask: &[Goldilocks],
    zero_hd: &[Goldilocks],
    stack: &str,
) -> (Exec, Vec<Goldilocks>) {
    let mut ex = Exec::new();
    ex.set_rsqrt(rsqrt_table.to_vec());
    let x = ex.input(x_plain.to_vec(), 0);

    let ln1_w = load_i32(&format!("{stack}L{li}_ln1_w_i32.bin"));
    let ln1_b = load_i32(&format!("{stack}L{li}_ln1_b_i32.bin"));
    let q_w = ex.input(load_i32(&format!("{stack}L{li}_q_w_i32.bin")), 0);
    let k_w = ex.input(load_i32(&format!("{stack}L{li}_k_w_i32.bin")), 0);
    let v_w = ex.input(load_i32(&format!("{stack}L{li}_v_w_i32.bin")), 0);
    let o_w = ex.input(load_i32(&format!("{stack}L{li}_o_proj_w_i32.bin")), 0);
    let ln2_w = load_i32(&format!("{stack}L{li}_ln2_w_i32.bin"));
    let ln2_b = load_i32(&format!("{stack}L{li}_ln2_b_i32.bin"));
    let fc_w = ex.input(load_i32(&format!("{stack}L{li}_fc_w_i32.bin")), 0);
    let proj_w = ex.input(load_i32(&format!("{stack}L{li}_proj_w_i32.bin")), 0);

    let q_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_q_b_i32.bin")), SEQ);
    let k_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_k_b_i32.bin")), SEQ);
    let v_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_v_b_i32.bin")), SEQ);
    let o_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_o_proj_b_i32.bin")), SEQ);
    let fc_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_fc_b_i32.bin")), SEQ);
    let proj_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_proj_b_i32.bin")), SEQ);

    let ln1 = layer_norm_rows_ir(&mut ex, x, &ln1_w, &ln1_b, N_REAL, SEQ);
    let q_raw = ex.matmul(ln1, q_w, SEQ, H_PAD, H_PAD);
    let q = ex.affine(q_raw, q_b, 16);
    let k_raw = ex.matmul(ln1, k_w, SEQ, H_PAD, H_PAD);
    let k = ex.affine(k_raw, k_b, 16);
    let v_raw = ex.matmul(ln1, v_w, SEQ, H_PAD, H_PAD);
    let v = ex.affine(v_raw, v_b, 16);

    let qh = split_heads_q(ex.get(q));
    let kh = split_heads_k(ex.get(k));
    let vh = split_heads_q(ex.get(v));

    let mut scores_plain = Vec::with_capacity(HEADS_PAD * SEQ * SEQ);
    for h in 0..HEADS_PAD {
        let qh_h = ex.input(qh[h * SEQ * HDIM..(h + 1) * SEQ * HDIM].to_vec(), 0);
        let kh_h = ex.input(kh[h * HDIM * SEQ..(h + 1) * HDIM * SEQ].to_vec(), 0);
        let sr = ex.matmul(qh_h, kh_h, SEQ, HDIM, SEQ);
        let sc = ex.affine(sr, mask.to_vec(), 19);
        scores_plain.extend_from_slice(ex.get(sc));
    }
    let scores = ex.input(scores_plain, 0);
    let sm = ex.softmax(scores, exp_table, EXP_OFFSET, HEADS_PAD * SEQ, SEQ, rng);

    let mut attn_all = Vec::with_capacity(HEADS * SEQ * HDIM);
    for h in 0..HEADS {
        let sm_h = ex.input(ex.get(sm)[h * SEQ * SEQ..(h + 1) * SEQ * SEQ].to_vec(), 0);
        let vh_h = ex.input(vh[h * SEQ * HDIM..(h + 1) * SEQ * HDIM].to_vec(), 0);
        let ar = ex.matmul(sm_h, vh_h, SEQ, SEQ, HDIM);
        let at = ex.affine(ar, zero_hd.to_vec(), 16);
        attn_all.extend_from_slice(ex.get(at));
    }
    let attn_full = ex.input(concat_heads(&attn_all), 0);
    let op_raw = ex.matmul(attn_full, o_w, SEQ, H_PAD, H_PAD);
    let op = ex.affine(op_raw, o_b, 16);
    let add1 = ex.add(x, op);

    let ln2 = layer_norm_rows_ir(&mut ex, add1, &ln2_w, &ln2_b, N_REAL, SEQ);
    let fc_raw = ex.matmul(ln2, fc_w, SEQ, H_PAD, FFN_PAD);
    let fc = ex.affine(fc_raw, fc_b, 16);
    let gelu_idx = silu_raw(ex.get(fc), gelu_table, GELU_OFFSET).0;
    let act = ex.lookup(&gelu_idx, gelu_table, rng);
    let proj_raw = ex.matmul(act, proj_w, SEQ, FFN_PAD, H_PAD);
    let ffn_out = ex.affine(proj_raw, proj_b, 16);
    let x_new = ex.add(add1, ffn_out);

    let out = ex.get(x_new).to_vec();
    (ex, out)
}

fn forward_final(
    x_plain: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    stack: &str,
) -> (Exec, Vec<Goldilocks>) {
    let mut ex = Exec::new();
    ex.set_rsqrt(rsqrt_table.to_vec());
    let x = ex.input(x_plain.to_vec(), 0);
    let ln_f_w = load_i32(&format!("{stack}ln_f_w_i32.bin"));
    let ln_f_b = load_i32(&format!("{stack}ln_f_b_i32.bin"));
    let lm_head_w = ex.input(load_i32(&format!("{stack}lm_head_w_i32.bin")), 0);
    let ln_f = layer_norm_rows_ir(&mut ex, x, &ln_f_w, &ln_f_b, N_REAL, SEQ);
    let logits = ex.matmul(ln_f, lm_head_w, SEQ, H_PAD, VOCAB_PAD);
    let out = ex.get(logits).to_vec();
    (ex, out)
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let stack = format!("{base}gpt2_stack/");
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    let exp_table = load_i32(&format!("{base}exp_table_i32.bin"));
    let gelu_table = load_i32(&format!("{stack}gelu_table_i32.bin"));
    let mask = load_i32(&format!("{stack}mask_i32.bin"));

    let whir = Whir::new_testing(16);
    let mut rng = XorShift64::new(0x200);
    let zero_hd = vec![from_i32(0); SEQ * HDIM];

    let mut x = load_i32(&format!("{stack}embedding_i32.bin"));

    // Streaming: forward + prove each layer, dropping its executor to bound memory.
    for li in 0..N_LAYERS {
        let (ex_li, x_new) = forward_layer(
            li, &x, &mut rng, &rsqrt_table, &exp_table, &gelu_table, &mask, &zero_hd, &stack,
        );
        x = x_new;
        ex_li.prove(&whir, &mut rng);
        println!("layer {li} proved");
    }

    let (ex_f, logits) = forward_final(&x, &rsqrt_table, &stack);
    let argmax = argmax_vocab(&logits);
    println!("GPT-2 124M (IR): argmax {:?}", argmax);
    ex_f.prove(&whir, &mut rng);
    println!("GPT-2 124M (IR): verified; global_open_count = {}", zkie_gkr::whir::global_open_count());
}

fn argmax_vocab(logits: &[Goldilocks]) -> Vec<usize> {
    let mut out = Vec::with_capacity(SEQ);
    for s in 0..SEQ {
        let row = &logits[s * VOCAB_PAD..(s + 1) * VOCAB_PAD];
        let mut best = 0usize;
        let mut best_v = i64::MIN;
        for j in 0..VOCAB {
            let v = to_i64(row[j]);
            if v > best_v {
                best_v = v;
                best = j;
            }
        }
        out.push(best);
    }
    out
}