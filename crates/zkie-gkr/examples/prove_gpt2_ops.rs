//! GPT-2 124M end-to-end through the unified ops interface.
//! Mirrors prove_200m_ops.rs: load weights, run the forward pass, and let each
//! op prove itself. Differs from TimesFM in (a) GELU(gelu_new) instead of
//! ReLU/SiLU, (b) fused QKV split into 3 matmuls, (c) heads=12 padded to
//! HEADS_PAD=16 for the power-of-two softmax batch, and (d) the LM head
//! (tied embeddings, transposed to [hidden, vocab]).

use zkie_gkr::committed::silu_raw;
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_gkr::fixed_point::{from_i32, to_i64};
use zkie_gkr::ops::{
    add, affine, layer_norm_rows, lookup, matmul, softmax_rows, Ctx,
};

const H_PAD: usize = 1024;
const FFN_PAD: usize = 4096;
const VOCAB_PAD: usize = 65536;
const VOCAB: usize = 50257;
// Context length; must match GPT2_SEQ used in export/extract scripts.
const SEQ: usize = 512;
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

/// Broadcast a per-dimension bias [h] to [seq, h] for per-token affine.
fn broadcast_bias(bias: &[Goldilocks], seq: usize) -> Vec<Goldilocks> {
    let h = bias.len();
    let mut out = Vec::with_capacity(seq * h);
    for _ in 0..seq {
        out.extend_from_slice(bias);
    }
    out
}

/// Reorder Q/V [SEQ, H_PAD] into [HEADS_PAD, SEQ, HDIM]; heads 12..16 read the
/// zero padding columns (768..1024), so they are naturally zero.
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

/// Reorder K [SEQ, H_PAD] into [HEADS_PAD, HDIM, SEQ] (transposed for QK^T).
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

/// Reorder [HEADS, SEQ, HDIM] back into [SEQ, H_PAD] (real heads only).
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

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let stack = format!("{base}gpt2_stack/");
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    let exp_table = load_i32(&format!("{base}exp_table_i32.bin"));
    let gelu_table = load_i32(&format!("{stack}gelu_table_i32.bin"));

    let mut ctx = Ctx::new(0x200);

    // Prologue: token + positional embeddings precomputed into one hidden tensor.
    let mut x = ctx.commit(load_i32(&format!("{stack}embedding_i32.bin")));
    let mask = ctx.commit(load_i32(&format!("{stack}mask_i32.bin")));
    let zero_hd = ctx.commit(vec![from_i32(0); SEQ * HDIM]);

    for li in 0..N_LAYERS {
        let ln1_w = ctx.commit(load_i32(&format!("{stack}L{li}_ln1_w_i32.bin")));
        let ln1_b = ctx.commit(load_i32(&format!("{stack}L{li}_ln1_b_i32.bin")));
        let q_w = ctx.commit(load_i32(&format!("{stack}L{li}_q_w_i32.bin")));
        let k_w = ctx.commit(load_i32(&format!("{stack}L{li}_k_w_i32.bin")));
        let v_w = ctx.commit(load_i32(&format!("{stack}L{li}_v_w_i32.bin")));
        let q_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_q_b_i32.bin")), SEQ));
        let k_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_k_b_i32.bin")), SEQ));
        let v_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_v_b_i32.bin")), SEQ));
        let o_w = ctx.commit(load_i32(&format!("{stack}L{li}_o_proj_w_i32.bin")));
        let o_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_o_proj_b_i32.bin")), SEQ));
        let ln2_w = ctx.commit(load_i32(&format!("{stack}L{li}_ln2_w_i32.bin")));
        let ln2_b = ctx.commit(load_i32(&format!("{stack}L{li}_ln2_b_i32.bin")));
        let fc_w = ctx.commit(load_i32(&format!("{stack}L{li}_fc_w_i32.bin")));
        let fc_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_fc_b_i32.bin")), SEQ));
        let proj_w = ctx.commit(load_i32(&format!("{stack}L{li}_proj_w_i32.bin")));
        let proj_b = ctx.commit(broadcast_bias(&load_i32(&format!("{stack}L{li}_proj_b_i32.bin")), SEQ));

        // Attention: Q/K/V projections, per-head causal attention, o_proj, residual.
        let ln1 = layer_norm_rows(&mut ctx, &x, &ln1_w.plain, &ln1_b.plain, N_REAL, &rsqrt_table, SEQ);
        let q_raw = matmul(&mut ctx, &ln1, &q_w, SEQ, H_PAD, H_PAD);
        let q = affine(&mut ctx, &q_raw, &q_b.plain, 16, false);
        let k_raw = matmul(&mut ctx, &ln1, &k_w, SEQ, H_PAD, H_PAD);
        let k = affine(&mut ctx, &k_raw, &k_b.plain, 16, false);
        let v_raw = matmul(&mut ctx, &ln1, &v_w, SEQ, H_PAD, H_PAD);
        let v = affine(&mut ctx, &v_raw, &v_b.plain, 16, false);

        let qh = split_heads_q(&q.plain);
        let kh = split_heads_k(&k.plain);
        let vh = split_heads_q(&v.plain);

        let mut scores_plain = Vec::with_capacity(HEADS_PAD * SEQ * SEQ);
        for h in 0..HEADS_PAD {
            let qh_h = ctx.commit(qh[h * SEQ * HDIM..(h + 1) * SEQ * HDIM].to_vec());
            let kh_h = ctx.commit(kh[h * HDIM * SEQ..(h + 1) * HDIM * SEQ].to_vec());
            let sr = matmul(&mut ctx, &qh_h, &kh_h, SEQ, HDIM, SEQ);
            // shift 19 = rescale 2^32 -> 2^16 AND multiply by 1/sqrt(64) = 1/8.
            let sc = affine(&mut ctx, &sr, &mask.plain, 19, false);
            scores_plain.extend_from_slice(&sc.plain);
        }
        let scores = ctx.commit(scores_plain);
        let sm = softmax_rows(&mut ctx, &scores, &exp_table, EXP_OFFSET, HEADS_PAD * SEQ, SEQ);

        let mut attn_all = Vec::with_capacity(HEADS * SEQ * HDIM);
        for h in 0..HEADS {
            let sm_h = ctx.commit(sm.plain[h * SEQ * SEQ..(h + 1) * SEQ * SEQ].to_vec());
            let vh_h = ctx.commit(vh[h * SEQ * HDIM..(h + 1) * SEQ * HDIM].to_vec());
            let ar = matmul(&mut ctx, &sm_h, &vh_h, SEQ, SEQ, HDIM);
            let at = affine(&mut ctx, &ar, &zero_hd.plain, 16, false);
            attn_all.extend_from_slice(&at.plain);
        }
        let attn_full = ctx.commit(concat_heads(&attn_all));

        let op_raw = matmul(&mut ctx, &attn_full, &o_w, SEQ, H_PAD, H_PAD);
        let op = affine(&mut ctx, &op_raw, &o_b.plain, 16, false);
        let add1 = add(&mut ctx, &x, &op);

        // FFN with GELU (gelu_new) as a single lookup.
        let ln2 = layer_norm_rows(&mut ctx, &add1, &ln2_w.plain, &ln2_b.plain, N_REAL, &rsqrt_table, SEQ);
        let fc_raw = matmul(&mut ctx, &ln2, &fc_w, SEQ, H_PAD, FFN_PAD);
        let fc = affine(&mut ctx, &fc_raw, &fc_b.plain, 16, false);
        let (gelu_idx, gelu_out) = silu_raw(&fc.plain, &gelu_table, GELU_OFFSET);
        let act = lookup(&mut ctx, &gelu_idx, &gelu_out, &gelu_table);
        let proj_raw = matmul(&mut ctx, &act, &proj_w, SEQ, FFN_PAD, H_PAD);
        let ffn_out = affine(&mut ctx, &proj_raw, &proj_b.plain, 16, false);
        x = add(&mut ctx, &add1, &ffn_out);
        println!("layer {li} verified");
    }

    // Epilogue: final LayerNorm + LM head (tied embedding, transposed).
    let ln_f_w = ctx.commit(load_i32(&format!("{stack}ln_f_w_i32.bin")));
    let ln_f_b = ctx.commit(load_i32(&format!("{stack}ln_f_b_i32.bin")));
    let lm_head_w = ctx.commit(load_i32(&format!("{stack}lm_head_w_i32.bin")));
    let ln_f = layer_norm_rows(&mut ctx, &x, &ln_f_w.plain, &ln_f_b.plain, N_REAL, &rsqrt_table, SEQ);
    let logits = matmul(&mut ctx, &ln_f, &lm_head_w, SEQ, H_PAD, VOCAB_PAD);

    let argmax = argmax_vocab(&logits.plain);
    println!("GPT-2 124M (ops): 12 layers + LM head verified");
    println!("proof argmax: {:?}", argmax);
    let (cn, cs, on, os, vn, vs) = ctx.stats();
    println!("timing: commit {} ({:.1}s), open {} ({:.1}s), verify {} ({:.1}s)",
        cn, cs, on, os, vn, vs);
}

/// Decode the 2^32-scale logits and return the argmax over the real vocab.
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