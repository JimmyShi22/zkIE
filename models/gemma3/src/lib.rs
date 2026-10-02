//! Gemma 3 270M (18 layers, post-norm sandwich, GQA + QK-norm + RoPE, gated
//! GELU MLP) op-graph builder. Assembles the full forward into `Vec<compose::Op>`
//! using the op primitives from `zkie-ops`. Weights/tables are read from
//! `models/gemma3/weights/` (i32 fixed-point `.bin` files).

use std::fs;
use std::sync::Arc;

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::from_i32;
use zkie_core::common::weights_io::WeightMmap;
use zkie_ops::compose::{causal_mask, Op, Store};

pub const H: usize = 640;
pub const H_PAD: usize = 1024;
pub const INTER: usize = 2048;
pub const HEADS: usize = 4;
pub const HDIM: usize = 256;
pub const VOCAB: usize = 262144;
pub const LAYERS: usize = 18;

/// Full-attention layers (rest are sliding-window); from the model config.
pub const FULL_LAYERS: [bool; LAYERS] = [
    false, false, false, false, false, true, false, false, false, false, false, true, false,
    false, false, false, false, true,
];

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

fn zeros(n: usize) -> Vec<Goldilocks> {
    vec![Goldilocks::ZERO; n]
}

/// Build one Gemma 3 layer (attention + gated-GELU MLP with sandwich norms).
#[allow(clippy::too_many_arguments)]
fn build_layer(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    layer: usize,
    m: usize,
    shift: u32,
    exp_t: usize,
    gelu_t: usize,
    rsqrt_t: usize,
    mask_t: usize,
    cos_t: usize,
    sin_t: usize,
) -> usize {
    let dir = "models/gemma3/weights";
    let q_w = load_i32(&format!("{dir}/L{layer}_q_w_i32.bin"));
    let k_w = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_k_w_i32.bin")).unwrap());
    let v_w = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_v_w_i32.bin")).unwrap());
    let o_w = load_i32(&format!("{dir}/L{layer}_o_w_i32.bin"));
    let gate_w = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_gate_w_i32.bin")).unwrap());
    let up_w = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_up_w_i32.bin")).unwrap());
    let down_w = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_down_w_i32.bin")).unwrap());
    let in_norm = load_i32(&format!("{dir}/L{layer}_in_norm_i32.bin"));
    let post_attn_norm = load_i32(&format!("{dir}/L{layer}_post_attn_norm_i32.bin"));
    let pre_ffn_norm = load_i32(&format!("{dir}/L{layer}_pre_ffn_norm_i32.bin"));
    let post_ffn_norm = load_i32(&format!("{dir}/L{layer}_post_ffn_norm_i32.bin"));
    let q_norm = load_i32(&format!("{dir}/L{layer}_q_norm_i32.bin"));
    let k_norm = load_i32(&format!("{dir}/L{layer}_k_norm_i32.bin"));

    // input layernorm
    let in_norm_t = store.push(broadcast(&in_norm, m));
    let h = store.push(vec![]);
    ops.push(Op::RmsNorm { x, w: in_norm_t, out: h, rsqrt_table: rsqrt_t, m, d: H_PAD, n_real: H });

    // K/V projections (single KV head, shared across the 4 query heads)
    let k_w_t = store.push_mmap(k_w.clone(), 0, k_w.len());
    let v_w_t = store.push_mmap(v_w.clone(), 0, v_w.len());
    let k_b = store.push(zeros(m * HDIM));
    let v_b = store.push(zeros(m * HDIM));
    let k = store.push(vec![]);
    let v = store.push(vec![]);
    let kr = store.push(vec![]);
    let vr = store.push(vec![]);
    ops.push(Op::Projection { x: h, w: k_w_t, bias: k_b, out: k, rem: kr, m, k: H_PAD, n: HDIM, shift });
    ops.push(Op::Projection { x: h, w: v_w_t, bias: v_b, out: v, rem: vr, m, k: H_PAD, n: HDIM, shift });

    // K QK-norm + RoPE
    let k_norm_t = store.push(broadcast(&k_norm, m));
    let kn = store.push(vec![]);
    ops.push(Op::RmsNorm { x: k, w: k_norm_t, out: kn, rsqrt_table: rsqrt_t, m, d: HDIM, n_real: HDIM });
    let k_rot = store.push(vec![]);
    ops.push(Op::RoPE { x: kn, cos: cos_t, sin: sin_t, out: k_rot, m, d: HDIM, shift });
    let kt = store.push(vec![]);
    ops.push(Op::Transpose { x: k_rot, out: kt, m, k: HDIM });

    let q_norm_t = store.push(broadcast(&q_norm, m));
    let mut head_outs = Vec::new();
    for head in 0..HEADS {
        // per-head Q projection, QK-norm, RoPE, attention, output projection
        let qh_w = store.push(slice_cols(&q_w, HEADS * HDIM, head * HDIM, (head + 1) * HDIM));
        let qh_b = store.push(zeros(m * HDIM));
        let qh = store.push(vec![]);
        let qh_rem = store.push(vec![]);
        let qn = store.push(vec![]);
        let q_rot = store.push(vec![]);
        let q_scaled = store.push(vec![]);
        let oh = store.push(slice_rows(&o_w, H_PAD, head * HDIM, (head + 1) * HDIM));
        let ob = store.push(zeros(m * H_PAD));
        let scores = store.push(vec![]);
        let scores_16 = store.push(vec![]);
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let attn = store.push(vec![]);
        let attn_16 = store.push(vec![]);
        let out_h = store.push(vec![]);
        let out_rem = store.push(vec![]);
        ops.push(Op::Projection { x: h, w: qh_w, bias: qh_b, out: qh, rem: qh_rem, m, k: H_PAD, n: HDIM, shift });
        ops.push(Op::RmsNorm { x: qh, w: q_norm_t, out: qn, rsqrt_table: rsqrt_t, m, d: HDIM, n_real: HDIM });
        ops.push(Op::RoPE { x: qn, cos: cos_t, sin: sin_t, out: q_rot, m, d: HDIM, shift });
        ops.push(Op::Scale { x: q_rot, out: q_scaled, factor: 4096, shift: 16 });
        ops.push(Op::MatMul { a: q_scaled, b: kt, c: scores, m, k: HDIM, n: m });
        ops.push(Op::Scale { x: scores, out: scores_16, factor: 1, shift: 16 });
        ops.push(Op::StableSoftmaxIndex { x: scores_16, mask: mask_t, out: idx, offset: 1 << 21, table_len: 1 << 21, m, n: m });
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_t, m, n: m });
        ops.push(Op::MatMul { a: probs, b: v, c: attn, m, k: m, n: HDIM });
        ops.push(Op::Scale { x: attn, out: attn_16, factor: 1, shift: 16 });
        ops.push(Op::Projection { x: attn_16, w: oh, bias: ob, out: out_h, rem: out_rem, m, k: HDIM, n: H_PAD, shift });
        head_outs.push(out_h);
    }
    let mut attn_acc = head_outs[0];
    for &ho in &head_outs[1..] {
        let s = store.push(vec![]);
        ops.push(Op::Add { a: attn_acc, b: ho, c: s });
        attn_acc = s;
    }

    // post-attention norm + residual
    let post_attn_t = store.push(broadcast(&post_attn_norm, m));
    let attn_norm = store.push(vec![]);
    ops.push(Op::RmsNorm { x: attn_acc, w: post_attn_t, out: attn_norm, rsqrt_table: rsqrt_t, m, d: H_PAD, n_real: H });
    let x2 = store.push(vec![]);
    ops.push(Op::Add { a: x, b: attn_norm, c: x2 });

    // gated GELU MLP with pre/post feedforward norms
    let pre_ffn_t = store.push(broadcast(&pre_ffn_norm, m));
    let h2 = store.push(vec![]);
    ops.push(Op::RmsNorm { x: x2, w: pre_ffn_t, out: h2, rsqrt_table: rsqrt_t, m, d: H_PAD, n_real: H });
    let gate_w_t = store.push_mmap(gate_w.clone(), 0, gate_w.len());
    let up_w_t = store.push_mmap(up_w.clone(), 0, up_w.len());
    let down_w_t = store.push_mmap(down_w.clone(), 0, down_w.len());
    let gate_b = store.push(zeros(m * INTER));
    let up_b = store.push(zeros(m * INTER));
    let down_b = store.push(zeros(m * H_PAD));
    let gate = store.push(vec![]);
    let gate_rem = store.push(vec![]);
    let up = store.push(vec![]);
    let up_rem = store.push(vec![]);
    let gelu_idx = store.push_idx(vec![]);
    let gelu = store.push(vec![]);
    let act = store.push(vec![]);
    let down = store.push(vec![]);
    let down_rem = store.push(vec![]);
    ops.push(Op::Projection { x: h2, w: gate_w_t, bias: gate_b, out: gate, rem: gate_rem, m, k: H_PAD, n: INTER, shift });
    ops.push(Op::Projection { x: h2, w: up_w_t, bias: up_b, out: up, rem: up_rem, m, k: H_PAD, n: INTER, shift });
    ops.push(Op::GeluIndex { x: gate, out: gelu_idx, offset: 1 << 23, table_len: 1 << 24 });
    ops.push(Op::Lookup { idx: gelu_idx, out: gelu, table: gelu_t });
    ops.push(Op::ScaleVec { x: gelu, scale: up, out: act, shift: 16 });
    ops.push(Op::Projection { x: act, w: down_w_t, bias: down_b, out: down, rem: down_rem, m, k: INTER, n: H_PAD, shift });

    let post_ffn_t = store.push(broadcast(&post_ffn_norm, m));
    let down_norm = store.push(vec![]);
    ops.push(Op::RmsNorm { x: down, w: post_ffn_t, out: down_norm, rsqrt_table: rsqrt_t, m, d: H_PAD, n_real: H });
    let out = store.push(vec![]);
    ops.push(Op::Add { a: x2, b: down_norm, c: out });
    out
}

/// Assemble the full Gemma 3 graph (embedding + 18 layers + final RMSNorm +
/// tied lm_head). Returns `(x0_id, logits_id)`.
pub fn build_gemma3(store: &mut Store, ops: &mut Vec<Op>, dir: &str, m: usize, shift: u32) -> (usize, usize) {
    let exp_table = load_i32("models/gpt2/weights/exp_table_i32.bin");
    let rsqrt_table = load_i32("models/gpt2/weights/rsqrt_table_i32.bin");
    let gelu_table = load_i32("models/gpt2/weights/gelu_table_i32.bin");
    let x0 = load_i32(&format!("{dir}/x0_i32.bin"));
    let lm_head = Arc::new(WeightMmap::open(&format!("{dir}/lm_head_i32.bin")).unwrap());
    let final_norm = load_i32(&format!("{dir}/final_norm_i32.bin"));
    let cos_local = load_i32(&format!("{dir}/rope_cos_local_i32.bin"));
    let sin_local = load_i32(&format!("{dir}/rope_sin_local_i32.bin"));
    let cos_global = load_i32(&format!("{dir}/rope_cos_global_i32.bin"));
    let sin_global = load_i32(&format!("{dir}/rope_sin_global_i32.bin"));

    let x0_t = store.push(x0[..m * H_PAD].to_vec());
    let exp_t = store.push(exp_table);
    let rsqrt_t = store.push(rsqrt_table);
    let gelu_t = store.push(gelu_table);
    let mask_t = store.push(causal_mask(m));
    let cos_local_t = store.push(cos_local);
    let sin_local_t = store.push(sin_local);
    let cos_global_t = store.push(cos_global);
    let sin_global_t = store.push(sin_global);

    let mut x_cur = x0_t;
    for layer in 0..LAYERS {
        let (cos_t, sin_t) = if FULL_LAYERS[layer] {
            (cos_global_t, sin_global_t)
        } else {
            (cos_local_t, sin_local_t)
        };
        x_cur = build_layer(store, ops, x_cur, layer, m, shift, exp_t, gelu_t, rsqrt_t, mask_t, cos_t, sin_t);
    }

    let final_norm_t = store.push(broadcast(&final_norm, m));
    let h_final = store.push(vec![]);
    ops.push(Op::RmsNorm { x: x_cur, w: final_norm_t, out: h_final, rsqrt_table: rsqrt_t, m, d: H_PAD, n_real: H });
    let lm_t = store.push_mmap(lm_head.clone(), 0, lm_head.len());
    let logits = store.push(vec![]);
    ops.push(Op::MatMul { a: h_final, b: lm_t, c: logits, m, k: H_PAD, n: VOCAB });
    (x0_t, logits)
}
