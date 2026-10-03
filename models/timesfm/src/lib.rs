//! TimesFM 1.0 200M: prologue (RevIN embed + SiLU FFN + freq) + 20 layers
//! (RMSNorm -> causal attention -> residual -> LayerNorm -> ReLU FFN -> residual)
//! + horizon SiLU head + rescale. Assembles into `Vec<compose::Op>`.

use std::fs;
use std::sync::Arc;

use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i32, from_i64};
use zkie_core::common::weights_io::WeightMmap;
use zkie_ops::compose::{Op, Store};

pub const H: usize = 1280;
pub const H_PAD: usize = 2048;
pub const SEQ: usize = 16;
pub const HEADS: usize = 16;
pub const HDIM: usize = 80;
pub const HDIM_PAD: usize = 128;
pub const LAYERS: usize = 20;
pub const SILU_OFF: i64 = 1 << 19;
pub const SILU_LEN: usize = 1 << 20;
pub const EXP_LEN: usize = 1 << 21;
pub const EXP_OFF: i64 = 1 << 21;

pub fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

pub fn load_i64(path: &str) -> Vec<i64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn broadcast(b: &[Goldilocks], rows: usize) -> Vec<Goldilocks> {
    (0..rows).flat_map(|_| b.iter().copied()).collect()
}

/// Slice a fused [H_PAD, 4096] qkv weight into a per-head [H_PAD, HDIM_PAD] block.
/// `part` is 0 (Q), 1 (K) or 2 (V); the real columns for `head` are
/// `part*H + head*HDIM .. part*H + (head+1)*HDIM`.
fn slice_qkv_head(qkv: &[Goldilocks], part: usize, head: usize) -> Vec<Goldilocks> {
    let col0 = part * H + head * HDIM;
    let mut out = vec![Goldilocks::ZERO; H_PAD * HDIM_PAD];
    for r in 0..H_PAD {
        for j in 0..HDIM {
            out[r * HDIM_PAD + j] = qkv[r * 4096 + col0 + j];
        }
    }
    out
}

fn slice_bias_head(b: &[Goldilocks], part: usize, head: usize) -> Vec<Goldilocks> {
    let c0 = part * H + head * HDIM;
    let mut out = vec![Goldilocks::ZERO; HDIM_PAD];
    out[..HDIM].copy_from_slice(&b[c0..c0 + HDIM]);
    out
}

/// Slice a [H_PAD, H_PAD] o_proj weight into a per-head [HDIM_PAD, H_PAD] block
/// (rows are the head dim, cols are the hidden dim).
fn slice_oproj_head(o: &[Goldilocks], head: usize) -> Vec<Goldilocks> {
    let r0 = head * HDIM;
    let mut out = vec![Goldilocks::ZERO; HDIM_PAD * H_PAD];
    for r in 0..HDIM {
        for c in 0..H {
            out[r * H_PAD + c] = o[(r0 + r) * H_PAD + c];
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn build_layer(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    layer: usize,
    shift: u32,
    exp_t: usize,
    rsqrt_t: usize,
    silu_t: usize,
    q_scale_t: usize,
    mask_t: usize,
) -> usize {
    let dir = "models/timesfm/weights";
    let qkv_w = load_i32(&format!("{dir}/L{layer}_qkv_w_i32.bin"));
    let qkv_b = load_i32(&format!("{dir}/L{layer}_qkv_b_i32.bin"));
    let o_w = load_i32(&format!("{dir}/L{layer}_o_proj_w_i32.bin"));
    let o_b = load_i32(&format!("{dir}/L{layer}_o_proj_b_i32.bin"));
    let gate_w = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_gate_w_i32.bin")).unwrap());
    let gate_b = load_i32(&format!("{dir}/L{layer}_gate_b_i32.bin"));
    let down_w = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_down_w_i32.bin")).unwrap());
    let down_b = load_i32(&format!("{dir}/L{layer}_down_b_i32.bin"));
    let lnw = load_i32(&format!("{dir}/L{layer}_lnw_i32.bin"));
    let mlp_w = load_i32(&format!("{dir}/L{layer}_mlp_w_i32.bin"));
    let mlp_b = load_i32(&format!("{dir}/L{layer}_mlp_b_i32.bin"));

    // RMSNorm (input_layernorm)
    let lnw_t = store.push(broadcast(&lnw, SEQ));
    let h = store.push(vec![]);
    ops.push(Op::RmsNorm { x, w: lnw_t, out: h, rsqrt_table: rsqrt_t, m: SEQ, d: H_PAD, n_real: H });

    // Causal attention (16 heads, fused QKV split per head)
    let mut head_outs = Vec::new();
    for head in 0..HEADS {
        let qh = store.push(slice_qkv_head(&qkv_w, 0, head));
        let kh = store.push(slice_qkv_head(&qkv_w, 1, head));
        let vh = store.push(slice_qkv_head(&qkv_w, 2, head));
        let oh = store.push(slice_oproj_head(&o_w, head));
        let qb = store.push(broadcast(&slice_bias_head(&qkv_b, 0, head), SEQ));
        let kb = store.push(broadcast(&slice_bias_head(&qkv_b, 1, head), SEQ));
        let vb = store.push(broadcast(&slice_bias_head(&qkv_b, 2, head), SEQ));
        let ob = store.push(broadcast(&vec![Goldilocks::ZERO; H_PAD], SEQ));
        let q = store.push(vec![]);
        let k = store.push(vec![]);
        let v = store.push(vec![]);
        let qr = store.push(vec![]);
        let kr = store.push(vec![]);
        let vr = store.push(vec![]);
        let q_scaled = store.push(vec![]);
        let kt = store.push(vec![]);
        let scores = store.push(vec![]);
        let scores_16 = store.push(vec![]);
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let attn = store.push(vec![]);
        let attn_16 = store.push(vec![]);
        let out_h = store.push(vec![]);
        let out_rem = store.push(vec![]);
        ops.push(Op::Projection { x: h, w: qh, bias: qb, out: q, rem: qr, m: SEQ, k: H_PAD, n: HDIM_PAD, shift });
        ops.push(Op::Projection { x: h, w: kh, bias: kb, out: k, rem: kr, m: SEQ, k: H_PAD, n: HDIM_PAD, shift });
        ops.push(Op::Projection { x: h, w: vh, bias: vb, out: v, rem: vr, m: SEQ, k: H_PAD, n: HDIM_PAD, shift });
        ops.push(Op::ScaleVec { x: q, scale: q_scale_t, out: q_scaled, shift: 16 });
        ops.push(Op::Transpose { x: k, out: kt, m: SEQ, k: HDIM_PAD });
        ops.push(Op::MatMul { a: q_scaled, b: kt, c: scores, m: SEQ, k: HDIM_PAD, n: SEQ });
        ops.push(Op::Scale { x: scores, out: scores_16, factor: 1, shift: 16 });
        ops.push(Op::StableSoftmaxIndex { x: scores_16, mask: mask_t, out: idx, offset: EXP_OFF, table_len: EXP_LEN, m: SEQ, n: SEQ });
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_t, m: SEQ, n: SEQ });
        ops.push(Op::MatMul { a: probs, b: v, c: attn, m: SEQ, k: SEQ, n: HDIM_PAD });
        ops.push(Op::Scale { x: attn, out: attn_16, factor: 1, shift: 16 });
        ops.push(Op::Projection { x: attn_16, w: oh, bias: ob, out: out_h, rem: out_rem, m: SEQ, k: HDIM_PAD, n: H_PAD, shift });
        head_outs.push(out_h);
    }
    let mut attn_acc = head_outs[0];
    for &ho in &head_outs[1..] {
        let s = store.push(vec![]);
        ops.push(Op::Add { a: attn_acc, b: ho, c: s });
        attn_acc = s;
    }
    let o_b_t = store.push(broadcast(&o_b, SEQ));
    let attn_biased = store.push(vec![]);
    ops.push(Op::Add { a: attn_acc, b: o_b_t, c: attn_biased });
    let x2 = store.push(vec![]);
    ops.push(Op::Add { a: x, b: attn_biased, c: x2 });

    // MLP: LayerNorm -> gate -> ReLU -> down -> residual
    let mlp_w_t = store.push(broadcast(&mlp_w, SEQ));
    let mlp_b_t = store.push(broadcast(&mlp_b, SEQ));
    let ln = store.push(vec![]);
    ops.push(Op::LayerNormCentered { x: x2, w: mlp_w_t, b: mlp_b_t, out: ln, rsqrt_table: rsqrt_t, m: SEQ, d: H_PAD, n_real: H });
    let gate_w_t = store.push_mmap(gate_w.clone(), 0, gate_w.len());
    let gate_b_t = store.push(broadcast(&gate_b, SEQ));
    let down_w_t = store.push_mmap(down_w.clone(), 0, down_w.len());
    let down_b_t = store.push(broadcast(&down_b, SEQ));
    let gate = store.push(vec![]);
    let gate_rem = store.push(vec![]);
    let relu = store.push(vec![]);
    let ff = store.push(vec![]);
    let ff_rem = store.push(vec![]);
    ops.push(Op::Projection { x: ln, w: gate_w_t, bias: gate_b_t, out: gate, rem: gate_rem, m: SEQ, k: H_PAD, n: H_PAD, shift });
    ops.push(Op::Relu { x: gate, out: relu });
    ops.push(Op::Projection { x: relu, w: down_w_t, bias: down_b_t, out: ff, rem: ff_rem, m: SEQ, k: H_PAD, n: H_PAD, shift });
    let out = store.push(vec![]);
    ops.push(Op::Add { a: x2, b: ff, c: out });
    out
}

pub fn build_timesfm(store: &mut Store, ops: &mut Vec<Op>, dir: &str, shift: u32) -> usize {
    let exp_table = load_i32("models/gpt2/weights/exp_table_i32.bin");
    let rsqrt_table = load_i32("models/gpt2/weights/rsqrt_table_i32.bin");
    let silu_table = load_i32(&format!("{dir}/silu_table_i32.bin"));
    let q_scale = load_i32(&format!("{dir}/q_scale_i32.bin"));
    let mask = load_i32(&format!("{dir}/mask_q_i32.bin"));
    let cat = load_i32(&format!("{dir}/cat_i32.bin"));
    let gather = load_i32(&format!("{dir}/gather_i32.bin"));
    let embedding = load_i32(&format!("{dir}/embedding_i32.bin"));
    let pro_hid_w = load_i32(&format!("{dir}/pro_hid_w_i32.bin"));
    let pro_hid_b = load_i32(&format!("{dir}/pro_hid_b_i32.bin"));
    let pro_out_w = load_i32(&format!("{dir}/pro_out_w_i32.bin"));
    let pro_out_b = load_i32(&format!("{dir}/pro_out_b_i32.bin"));
    let pro_res_w = load_i32(&format!("{dir}/pro_res_w_i32.bin"));
    let pro_res_b = load_i32(&format!("{dir}/pro_res_b_i32.bin"));
    let head_hid_w = load_i32(&format!("{dir}/head_hid_w_i32.bin"));
    let head_hid_b = load_i32(&format!("{dir}/head_hid_b_i32.bin"));
    let head_out_w = load_i32(&format!("{dir}/head_out_w_i32.bin"));
    let head_out_b = load_i32(&format!("{dir}/head_out_b_i32.bin"));
    let head_res_w = load_i32(&format!("{dir}/head_res_w_i32.bin"));
    let head_res_b = load_i32(&format!("{dir}/head_res_b_i32.bin"));
    let scale = load_i64(&format!("{dir}/scale_i64.bin"));

    let exp_t = store.push(exp_table);
    let rsqrt_t = store.push(rsqrt_table);
    let silu_t = store.push(silu_table);
    let q_scale_t = store.push(q_scale);
    let mask_t = store.push(mask);

    // Prologue
    let cat_t = store.push(cat);
    let gather_t = store.push(gather);
    let emb_t = store.push(broadcast(&embedding, SEQ));
    let pro_hid_w_t = store.push(pro_hid_w);
    let pro_hid_b_t = store.push(broadcast(&pro_hid_b, SEQ));
    let pro_out_w_t = store.push(pro_out_w);
    let pro_out_b_t = store.push(broadcast(&pro_out_b, SEQ));
    let pro_res_w_t = store.push(pro_res_w);
    let pro_res_b_t = store.push(broadcast(&pro_res_b, SEQ));
    let hid = store.push(vec![]);
    let hid_rem = store.push(vec![]);
    let silu_idx = store.push_idx(vec![]);
    let silu = store.push(vec![]);
    let o = store.push(vec![]);
    let o_rem = store.push(vec![]);
    let r = store.push(vec![]);
    let r_rem = store.push(vec![]);
    let add = store.push(vec![]);
    ops.push(Op::Projection { x: cat_t, w: pro_hid_w_t, bias: pro_hid_b_t, out: hid, rem: hid_rem, m: SEQ, k: 64, n: H_PAD, shift });
    ops.push(Op::GeluIndex { x: hid, out: silu_idx, offset: SILU_OFF, table_len: SILU_LEN });
    ops.push(Op::Lookup { idx: silu_idx, out: silu, table: silu_t });
    ops.push(Op::Projection { x: silu, w: pro_out_w_t, bias: pro_out_b_t, out: o, rem: o_rem, m: SEQ, k: H_PAD, n: H_PAD, shift });
    ops.push(Op::Projection { x: cat_t, w: pro_res_w_t, bias: pro_res_b_t, out: r, rem: r_rem, m: SEQ, k: 64, n: H_PAD, shift });
    ops.push(Op::Add { a: o, b: r, c: add });
    let x0 = store.push(vec![]);
    ops.push(Op::Add { a: add, b: gather_t, c: x0 });
    let x1 = store.push(vec![]);
    ops.push(Op::Add { a: x0, b: emb_t, c: x1 });

    let mut x_cur = x1;
    for layer in 0..LAYERS {
        x_cur = build_layer(store, ops, x_cur, layer, shift, exp_t, rsqrt_t, silu_t, q_scale_t, mask_t);
    }

    // Epilogue (horizon FFN) + rescale
    let head_hid_w_t = store.push(head_hid_w);
    let head_hid_b_t = store.push(broadcast(&head_hid_b, SEQ));
    let head_out_w_t = store.push(head_out_w);
    let head_out_b_t = store.push(broadcast(&head_out_b, SEQ));
    let head_res_w_t = store.push(head_res_w);
    let head_res_b_t = store.push(broadcast(&head_res_b, SEQ));
    let hh = store.push(vec![]);
    let hh_rem = store.push(vec![]);
    let hh_idx = store.push_idx(vec![]);
    let hh_silu = store.push(vec![]);
    let ho = store.push(vec![]);
    let ho_rem = store.push(vec![]);
    let hr = store.push(vec![]);
    let hr_rem = store.push(vec![]);
    let add31 = store.push(vec![]);
    ops.push(Op::Projection { x: x_cur, w: head_hid_w_t, bias: head_hid_b_t, out: hh, rem: hh_rem, m: SEQ, k: H_PAD, n: H_PAD, shift });
    ops.push(Op::GeluIndex { x: hh, out: hh_idx, offset: SILU_OFF, table_len: SILU_LEN });
    ops.push(Op::Lookup { idx: hh_idx, out: hh_silu, table: silu_t });
    ops.push(Op::Projection { x: hh_silu, w: head_out_w_t, bias: head_out_b_t, out: ho, rem: ho_rem, m: SEQ, k: H_PAD, n: H_PAD, shift });
    ops.push(Op::Projection { x: x_cur, w: head_res_w_t, bias: head_res_b_t, out: hr, rem: hr_rem, m: SEQ, k: H_PAD, n: H_PAD, shift });
    ops.push(Op::Add { a: ho, b: hr, c: add31 });
    let scaled = store.push(vec![]);
    ops.push(Op::Scale { x: add31, out: scaled, factor: scale[0], shift: 16 });
    let u2_t = store.push(broadcast(&[from_i64(scale[1])], SEQ * H_PAD));
    let out = store.push(vec![]);
    ops.push(Op::Add { a: scaled, b: u2_t, c: out });
    out
}
