//! DeepSeek-V2-Lite (27 layers, MLA attention + MoE FFN) op-graph builder.
//!
//! MLA decomposes into per-head nope/rope projections and two attention
//! matmuls. The MoE is assembled densely: all 64 routed experts are computed,
//! then scaled by a precomputed top-6 gate and summed (the routing proof is
//! deferred; the gate is currently a trusted constant).

use std::fs;

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::from_i32;
use zkie_ops::compose::{causal_mask, Op, Store};

pub const H: usize = 2048;
pub const VOCAB: usize = 102400;
pub const VOCAB_PAD: usize = 131072;
pub const LAYERS: usize = 27;
pub const HEADS: usize = 16;
pub const QK_NOPE: usize = 128;
pub const QK_ROPE: usize = 64;
pub const V_HEAD: usize = 128;
pub const KV_LORA: usize = 512;
pub const MOE_PAD: usize = 2048;
pub const MOE_REAL: usize = 1408;
pub const DENSE_PAD: usize = 16384;
pub const DENSE_REAL: usize = 10944;
pub const SHARED_PAD: usize = 4096;
pub const SHARED_REAL: usize = 2816;
pub const N_ROUTED: usize = 64;

pub fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn slice_cols(m: &[Goldilocks], cols: usize, c0: usize, c1: usize) -> Vec<Goldilocks> {
    let rows = m.len() / cols;
    (0..rows).flat_map(|i| (c0..c1).map(move |j| m[i * cols + j])).collect()
}

fn slice_rows(m: &[Goldilocks], cols: usize, r0: usize, r1: usize) -> Vec<Goldilocks> {
    (r0..r1).flat_map(|i| (0..cols).map(move |j| m[i * cols + j])).collect()
}

fn broadcast(b: &[Goldilocks], rows: usize) -> Vec<Goldilocks> {
    (0..rows).flat_map(|_| b.iter().copied()).collect()
}

fn broadcast_token(b: &[Goldilocks], cols: usize) -> Vec<Goldilocks> {
    (0..b.len()).flat_map(|t| (0..cols).map(move |_| b[t])).collect()
}

fn zeros(n: usize) -> Vec<Goldilocks> {
    vec![Goldilocks::ZERO; n]
}

/// Build MLA attention: nope/rope split, QK-norm-free, RoPE on the rope slice,
/// two-score attention, output projection. Returns the attention output tensor.
#[allow(clippy::too_many_arguments)]
fn build_attention(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    layer: usize,
    m: usize,
    shift: u32,
    exp_t: usize,
    rsqrt_t: usize,
    mask_t: usize,
    cos_t: usize,
    sin_t: usize,
) -> usize {
    let dir = "models/deepseek-v2-lite/weights";
    let qnope_w = load_i32(&format!("{dir}/L{layer}_qnope_w_i32.bin"));
    let qpe_w = load_i32(&format!("{dir}/L{layer}_qpe_w_i32.bin"));
    let kvlora_w = load_i32(&format!("{dir}/L{layer}_kvlora_w_i32.bin"));
    let kpe_w = load_i32(&format!("{dir}/L{layer}_kpe_w_i32.bin"));
    let kva_ln = load_i32(&format!("{dir}/L{layer}_kva_ln_i32.bin"));
    let knope_w = load_i32(&format!("{dir}/L{layer}_knope_w_i32.bin"));
    let v_w = load_i32(&format!("{dir}/L{layer}_v_w_i32.bin"));
    let o_w = load_i32(&format!("{dir}/L{layer}_o_w_i32.bin"));

    // shared kv latent + MQA rope key
    let kvlora_w_t = store.push(kvlora_w);
    let kpe_w_t = store.push(kpe_w);
    let kv_b = store.push(zeros(m * KV_LORA));
    let kpe_b = store.push(zeros(m * QK_ROPE));
    let kvlora = store.push(vec![]);
    let kvlora_rem = store.push(vec![]);
    let kpe = store.push(vec![]);
    let kpe_rem = store.push(vec![]);
    ops.push(Op::Projection { x, w: kvlora_w_t, bias: kv_b, out: kvlora, rem: kvlora_rem, m, k: H, n: KV_LORA, shift });
    ops.push(Op::Projection { x, w: kpe_w_t, bias: kpe_b, out: kpe, rem: kpe_rem, m, k: H, n: QK_ROPE, shift });
    let kva_ln_t = store.push(broadcast(&kva_ln, m));
    let kvlora_n = store.push(vec![]);
    ops.push(Op::RmsNorm { x: kvlora, w: kva_ln_t, out: kvlora_n, rsqrt_table: rsqrt_t, m, d: KV_LORA, n_real: KV_LORA });
    let kpe_rot = store.push(vec![]);
    ops.push(Op::RoPE { x: kpe, cos: cos_t, sin: sin_t, out: kpe_rot, m, d: QK_ROPE, shift });
    let kt = store.push(vec![]);
    ops.push(Op::Transpose { x: kpe_rot, out: kt, m, k: QK_ROPE });

    let mut head_outs = Vec::new();
    for head in 0..HEADS {
        let qn_w = store.push(slice_cols(&qnope_w, HEADS * QK_NOPE, head * QK_NOPE, (head + 1) * QK_NOPE));
        let qp_w = store.push(slice_cols(&qpe_w, HEADS * QK_ROPE, head * QK_ROPE, (head + 1) * QK_ROPE));
        let kn_w = store.push(slice_cols(&knope_w, HEADS * QK_NOPE, head * QK_NOPE, (head + 1) * QK_NOPE));
        let vh_w = store.push(slice_cols(&v_w, HEADS * V_HEAD, head * V_HEAD, (head + 1) * V_HEAD));
        let oh = store.push(slice_rows(&o_w, H, head * V_HEAD, (head + 1) * V_HEAD));
        let qn = store.push(vec![]);
        let qn_rem = store.push(vec![]);
        let qp = store.push(vec![]);
        let qp_rem = store.push(vec![]);
        let kn = store.push(vec![]);
        let kn_rem = store.push(vec![]);
        let vh = store.push(vec![]);
        let vh_rem = store.push(vec![]);
        ops.push(Op::Projection { x, w: qn_w, bias: store.push(zeros(m * QK_NOPE)), out: qn, rem: qn_rem, m, k: H, n: QK_NOPE, shift });
        ops.push(Op::Projection { x, w: qp_w, bias: store.push(zeros(m * QK_ROPE)), out: qp, rem: qp_rem, m, k: H, n: QK_ROPE, shift });
        ops.push(Op::Projection { x: kvlora_n, w: kn_w, bias: store.push(zeros(m * QK_NOPE)), out: kn, rem: kn_rem, m, k: KV_LORA, n: QK_NOPE, shift });
        ops.push(Op::Projection { x: kvlora_n, w: vh_w, bias: store.push(zeros(m * V_HEAD)), out: vh, rem: vh_rem, m, k: KV_LORA, n: V_HEAD, shift });
        let qp_rot = store.push(vec![]);
        ops.push(Op::RoPE { x: qp, cos: cos_t, sin: sin_t, out: qp_rot, m, d: QK_ROPE, shift });
        let knt = store.push(vec![]);
        ops.push(Op::Transpose { x: kn, out: knt, m, k: QK_NOPE });
        let s_nope = store.push(vec![]);
        let s_rope = store.push(vec![]);
        let s = store.push(vec![]);
        let s_16 = store.push(vec![]);
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let attn = store.push(vec![]);
        let attn_16 = store.push(vec![]);
        let out_h = store.push(vec![]);
        let out_rem = store.push(vec![]);
        ops.push(Op::MatMul { a: qn, b: knt, c: s_nope, m, k: QK_NOPE, n: m });
        ops.push(Op::MatMul { a: qp_rot, b: kt, c: s_rope, m, k: QK_ROPE, n: m });
        ops.push(Op::Add { a: s_nope, b: s_rope, c: s });
        ops.push(Op::Scale { x: s, out: s_16, factor: 7519, shift: 32 });
        ops.push(Op::StableSoftmaxIndex { x: s_16, mask: mask_t, out: idx, offset: 1 << 21, table_len: 1 << 21, m, n: m });
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_t, m, n: m });
        ops.push(Op::MatMul { a: probs, b: vh, c: attn, m, k: m, n: V_HEAD });
        ops.push(Op::Scale { x: attn, out: attn_16, factor: 1, shift: 16 });
        ops.push(Op::Projection { x: attn_16, w: oh, bias: store.push(zeros(m * H)), out: out_h, rem: out_rem, m, k: V_HEAD, n: H, shift });
        head_outs.push(out_h);
    }
    let mut acc = head_outs[0];
    for &ho in &head_outs[1..] {
        let s = store.push(vec![]);
        ops.push(Op::Add { a: acc, b: ho, c: s });
        acc = s;
    }
    acc
}

/// Build the MoE FFN (layers 1..26): all 64 routed experts (dense) scaled by a
/// precomputed gate, plus the shared experts.
#[allow(clippy::too_many_arguments)]
fn build_moe(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    layer: usize,
    m: usize,
    shift: u32,
    silu_t: usize,
) -> usize {
    let dir = "models/deepseek-v2-lite/weights";
    let gate = load_i32(&format!("{dir}/L{layer}_gate_i32.bin"));
    let shared_g = load_i32(&format!("{dir}/L{layer}_shared_gate_i32.bin"));
    let shared_u = load_i32(&format!("{dir}/L{layer}_shared_up_i32.bin"));
    let shared_d = load_i32(&format!("{dir}/L{layer}_shared_down_i32.bin"));
    let eg = load_i32(&format!("{dir}/L{layer}_experts_gate_i32.bin"));
    let eu = load_i32(&format!("{dir}/L{layer}_experts_up_i32.bin"));
    let ed = load_i32(&format!("{dir}/L{layer}_experts_down_i32.bin"));

    let mut acc: Option<usize> = None;
    for expert in 0..N_ROUTED {
        let eg_w = store.push(eg[expert * H * MOE_PAD..(expert + 1) * H * MOE_PAD].to_vec());
        let eu_w = store.push(eu[expert * H * MOE_PAD..(expert + 1) * H * MOE_PAD].to_vec());
        let ed_w = store.push(ed[expert * MOE_PAD * H..(expert + 1) * MOE_PAD * H].to_vec());
        let g = store.push(vec![]);
        let g_rem = store.push(vec![]);
        let u = store.push(vec![]);
        let u_rem = store.push(vec![]);
        let silu_idx = store.push_idx(vec![]);
        let silu = store.push(vec![]);
        let act = store.push(vec![]);
        let d = store.push(vec![]);
        let d_rem = store.push(vec![]);
        ops.push(Op::Projection { x, w: eg_w, bias: store.push(zeros(m * MOE_PAD)), out: g, rem: g_rem, m, k: H, n: MOE_PAD, shift });
        ops.push(Op::Projection { x, w: eu_w, bias: store.push(zeros(m * MOE_PAD)), out: u, rem: u_rem, m, k: H, n: MOE_PAD, shift });
        ops.push(Op::GeluIndex { x: g, out: silu_idx, offset: 1 << 23, table_len: 1 << 24 });
        ops.push(Op::Lookup { idx: silu_idx, out: silu, table: silu_t });
        ops.push(Op::ScaleVec { x: silu, scale: u, out: act, shift: 16 });
        ops.push(Op::Projection { x: act, w: ed_w, bias: store.push(zeros(m * H)), out: d, rem: d_rem, m, k: MOE_PAD, n: H, shift });
        let gate_col: Vec<Goldilocks> = (0..m).map(|t| gate[t * N_ROUTED + expert]).collect();
        let ge = store.push(broadcast_token(&gate_col, H));
        let weighted = store.push(vec![]);
        ops.push(Op::ScaleVec { x: d, scale: ge, out: weighted, shift: 16 });
        match acc {
            None => acc = Some(weighted),
            Some(a) => {
                let s = store.push(vec![]);
                ops.push(Op::Add { a, b: weighted, c: s });
                acc = Some(s);
            }
        }
    }
    let mut out = acc.unwrap();
    // shared experts (always active)
    let sg_w = store.push(shared_g);
    let su_w = store.push(shared_u);
    let sd_w = store.push(shared_d);
    let sg = store.push(vec![]);
    let sg_rem = store.push(vec![]);
    let su = store.push(vec![]);
    let su_rem = store.push(vec![]);
    let ss_idx = store.push_idx(vec![]);
    let ss = store.push(vec![]);
    let sa = store.push(vec![]);
    let sd = store.push(vec![]);
    let sd_rem = store.push(vec![]);
    ops.push(Op::Projection { x, w: sg_w, bias: store.push(zeros(m * SHARED_PAD)), out: sg, rem: sg_rem, m, k: H, n: SHARED_PAD, shift });
    ops.push(Op::Projection { x, w: su_w, bias: store.push(zeros(m * SHARED_PAD)), out: su, rem: su_rem, m, k: H, n: SHARED_PAD, shift });
    ops.push(Op::GeluIndex { x: sg, out: ss_idx, offset: 1 << 23, table_len: 1 << 24 });
    ops.push(Op::Lookup { idx: ss_idx, out: ss, table: silu_t });
    ops.push(Op::ScaleVec { x: ss, scale: su, out: sa, shift: 16 });
    ops.push(Op::Projection { x: sa, w: sd_w, bias: store.push(zeros(m * H)), out: sd, rem: sd_rem, m, k: SHARED_PAD, n: H, shift });
    let s = store.push(vec![]);
    ops.push(Op::Add { a: out, b: sd, c: s });
    out = s;
    out
}

/// Assemble the full DeepSeek-V2-Lite graph. Returns `(x0_id, logits_id)`.
pub fn build_deepseek(store: &mut Store, ops: &mut Vec<Op>, dir: &str, m: usize, shift: u32) -> (usize, usize) {
    let exp_table = load_i32("models/gpt2/weights/exp_table_i32.bin");
    let rsqrt_table = load_i32("models/gpt2/weights/rsqrt_table_i32.bin");
    let silu_table = load_i32(&format!("{dir}/silu_table_i32.bin"));
    let x0 = load_i32(&format!("{dir}/x0_i32.bin"));
    let lm_head = load_i32(&format!("{dir}/lm_head_i32.bin"));
    let final_norm = load_i32(&format!("{dir}/final_norm_i32.bin"));
    let cos = load_i32(&format!("{dir}/rope_cos_i32.bin"));
    let sin = load_i32(&format!("{dir}/rope_sin_i32.bin"));

    let x0_t = store.push(x0[..m * H].to_vec());
    let exp_t = store.push(exp_table);
    let rsqrt_t = store.push(rsqrt_table);
    let silu_t = store.push(silu_table);
    let mask_t = store.push(causal_mask(m));
    let cos_t = store.push(cos);
    let sin_t = store.push(sin);

    let mut x_cur = x0_t;
    for layer in 0..LAYERS {
        let dir = "models/deepseek-v2-lite/weights";
        let in_norm = load_i32(&format!("{dir}/L{layer}_in_norm_i32.bin"));
        let post_norm = load_i32(&format!("{dir}/L{layer}_post_attn_norm_i32.bin"));
        let in_norm_t = store.push(broadcast(&in_norm, m));
        let h = store.push(vec![]);
        ops.push(Op::RmsNorm { x: x_cur, w: in_norm_t, out: h, rsqrt_table: rsqrt_t, m, d: H, n_real: H });
        let attn = build_attention(store, ops, h, layer, m, shift, exp_t, rsqrt_t, mask_t, cos_t, sin_t);
        let x2 = store.push(vec![]);
        ops.push(Op::Add { a: x_cur, b: attn, c: x2 });
        let post_norm_t = store.push(broadcast(&post_norm, m));
        let h2 = store.push(vec![]);
        ops.push(Op::RmsNorm { x: x2, w: post_norm_t, out: h2, rsqrt_table: rsqrt_t, m, d: H, n_real: H });
        let ff = if layer == 0 {
            build_dense_ffn(store, ops, h2, m, shift, silu_t)
        } else {
            build_moe(store, ops, h2, layer, m, shift, silu_t)
        };
        let x3 = store.push(vec![]);
        ops.push(Op::Add { a: x2, b: ff, c: x3 });
        x_cur = x3;
    }

    let final_norm_t = store.push(broadcast(&final_norm, m));
    let h_final = store.push(vec![]);
    ops.push(Op::RmsNorm { x: x_cur, w: final_norm_t, out: h_final, rsqrt_table: rsqrt_t, m, d: H, n_real: H });
    let lm_t = store.push(lm_head);
    let logits = store.push(vec![]);
    ops.push(Op::MatMul { a: h_final, b: lm_t, c: logits, m, k: H, n: VOCAB_PAD });
    (x0_t, logits)
}

/// Dense FFN for layer 0 (SwiGLU with intermediate 10944).
fn build_dense_ffn(store: &mut Store, ops: &mut Vec<Op>, x: usize, m: usize, shift: u32, silu_t: usize) -> usize {
    let dir = "models/deepseek-v2-lite/weights";
    let gw = load_i32(&format!("{dir}/L0_gate_w_i32.bin"));
    let uw = load_i32(&format!("{dir}/L0_up_w_i32.bin"));
    let dw = load_i32(&format!("{dir}/L0_down_w_i32.bin"));
    let gw_t = store.push(gw);
    let uw_t = store.push(uw);
    let dw_t = store.push(dw);
    let g = store.push(vec![]);
    let g_rem = store.push(vec![]);
    let u = store.push(vec![]);
    let u_rem = store.push(vec![]);
    let silu_idx = store.push_idx(vec![]);
    let silu = store.push(vec![]);
    let act = store.push(vec![]);
    let d = store.push(vec![]);
    let d_rem = store.push(vec![]);
    ops.push(Op::Projection { x, w: gw_t, bias: store.push(zeros(m * DENSE_PAD)), out: g, rem: g_rem, m, k: H, n: DENSE_PAD, shift });
    ops.push(Op::Projection { x, w: uw_t, bias: store.push(zeros(m * DENSE_PAD)), out: u, rem: u_rem, m, k: H, n: DENSE_PAD, shift });
    ops.push(Op::GeluIndex { x: g, out: silu_idx, offset: 1 << 23, table_len: 1 << 24 });
    ops.push(Op::Lookup { idx: silu_idx, out: silu, table: silu_t });
    ops.push(Op::ScaleVec { x: silu, scale: u, out: act, shift: 16 });
    ops.push(Op::Projection { x: act, w: dw_t, bias: store.push(zeros(m * H)), out: d, rem: d_rem, m, k: DENSE_PAD, n: H, shift });
    d
}
