//! Load real GPT-2 L0 weights, slice them per attention head, assemble ONE
//! pre-norm multi-head transformer layer from op primitives, and prove it at a
//! small sequence length to validate the real-weight wiring end-to-end.

use std::fs;

use zkie_gkr::compose::{prove_shard, verify_shard, Op, Store};
use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::fixed_point::{from_i32, from_i64};

const D: usize = 1024;
const FFN: usize = 4096;
const HEADS: usize = 12;
const DH: usize = 64;

fn load_i32(path: &str) -> Vec<Goldilocks> {
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

fn main() {
    let mut rng = XorShift64::new(0xBEEF);
    let (m, shift) = (4usize, 16u32);
    let dir = "models/gpt2_stack";

    let q_w = load_i32(&format!("{dir}/L0_q_w_i32.bin"));
    let k_w = load_i32(&format!("{dir}/L0_k_w_i32.bin"));
    let v_w = load_i32(&format!("{dir}/L0_v_w_i32.bin"));
    let o_w = load_i32(&format!("{dir}/L0_o_proj_w_i32.bin"));
    let q_b = load_i32(&format!("{dir}/L0_q_b_i32.bin"));
    let k_b = load_i32(&format!("{dir}/L0_k_b_i32.bin"));
    let v_b = load_i32(&format!("{dir}/L0_v_b_i32.bin"));
    let o_b = load_i32(&format!("{dir}/L0_o_proj_b_i32.bin"));
    let ln1_w = load_i32(&format!("{dir}/L0_ln1_w_i32.bin"));
    let ln1_b = load_i32(&format!("{dir}/L0_ln1_b_i32.bin"));
    let ln2_w = load_i32(&format!("{dir}/L0_ln2_w_i32.bin"));
    let ln2_b = load_i32(&format!("{dir}/L0_ln2_b_i32.bin"));
    let fc_w = load_i32(&format!("{dir}/L0_fc_w_i32.bin"));
    let fc_b = load_i32(&format!("{dir}/L0_fc_b_i32.bin"));
    let proj_w = load_i32(&format!("{dir}/L0_proj_w_i32.bin"));
    let proj_b = load_i32(&format!("{dir}/L0_proj_b_i32.bin"));

    let table_len = 1usize << 18;
    let exp_table: Vec<Goldilocks> = (0..table_len).map(|j| from_i64((j % 255 + 1) as i64)).collect();
    let rsqrt_table: Vec<Goldilocks> = (0..table_len).map(|j| from_i64((j % 255 + 1) as i64)).collect();
    let gelu_table = load_i32(&format!("{dir}/gelu_table_i32.bin"));
    let gelu_len = 1usize << 16;
    let gelu_table: Vec<Goldilocks> = gelu_table.into_iter().take(gelu_len).collect();

    let mut store = Store::new();
    let x = store.push((0..m * D).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect());
    let exp_t = store.push(exp_table);
    let rsqrt_t = store.push(rsqrt_table);
    let gelu_t = store.push(gelu_table);

    // Pre-norm for attention.
    let ln1_w_t = store.push(broadcast(&ln1_w, m));
    let ln1_b_t = store.push(broadcast(&ln1_b, m));
    let h = store.push(vec![]);
    let mut ops = vec![Op::Layernorm { x, w: ln1_w_t, b: ln1_b_t, out: h, rsqrt_table: rsqrt_t, m, d: D }];

    // Multi-head attention on h.
    let mut head_outs = Vec::new();
    for head in 0..HEADS {
        let qh = store.push(slice_cols(&q_w, D, head * DH, (head + 1) * DH));
        let kh = store.push(slice_cols(&k_w, D, head * DH, (head + 1) * DH));
        let vh = store.push(slice_cols(&v_w, D, head * DH, (head + 1) * DH));
        let oh = store.push(slice_rows(&o_w, D, head * DH, (head + 1) * DH));
        let qb = store.push(broadcast(&q_b[head * DH..(head + 1) * DH], m));
        let kb = store.push(broadcast(&k_b[head * DH..(head + 1) * DH], m));
        let vb = store.push(broadcast(&v_b[head * DH..(head + 1) * DH], m));
        let ob = store.push(broadcast(&o_b, m));
        let q = store.push(vec![]);
        let k = store.push(vec![]);
        let v = store.push(vec![]);
        let qr = store.push(vec![]);
        let kr = store.push(vec![]);
        let vr = store.push(vec![]);
        let kt = store.push(vec![]);
        let scores = store.push(vec![]);
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let attn = store.push(vec![]);
        let out_h = store.push(vec![]);
        let out_rem = store.push(vec![]);
        ops.push(Op::Projection { x: h, w: qh, bias: qb, out: q, rem: qr, m, k: D, n: DH, shift });
        ops.push(Op::Projection { x: h, w: kh, bias: kb, out: k, rem: kr, m, k: D, n: DH, shift });
        ops.push(Op::Projection { x: h, w: vh, bias: vb, out: v, rem: vr, m, k: D, n: DH, shift });
        ops.push(Op::Transpose { x: k, out: kt, m, k: DH });
        ops.push(Op::MatMul { a: q, b: kt, c: scores, m, k: DH, n: m });
        ops.push(Op::SoftmaxIndex { x: scores, out: idx, table_len });
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_t, m, n: m });
        ops.push(Op::MatMul { a: probs, b: v, c: attn, m, k: m, n: DH });
        ops.push(Op::Projection { x: attn, w: oh, bias: ob, out: out_h, rem: out_rem, m, k: DH, n: D, shift });
        head_outs.push(out_h);
    }
    let mut attn_acc = head_outs[0];
    for &ho in &head_outs[1..] {
        let s = store.push(vec![]);
        ops.push(Op::Add { a: attn_acc, b: ho, c: s });
        attn_acc = s;
    }
    let x2 = store.push(vec![]);
    ops.push(Op::Add { a: x, b: attn_acc, c: x2 });

    // Pre-norm FFN.
    let ln2_w_t = store.push(broadcast(&ln2_w, m));
    let ln2_b_t = store.push(broadcast(&ln2_b, m));
    let h2 = store.push(vec![]);
    ops.push(Op::Layernorm { x: x2, w: ln2_w_t, b: ln2_b_t, out: h2, rsqrt_table: rsqrt_t, m, d: D });
    let fc_w_t = store.push(fc_w);
    let fc_b_t = store.push(broadcast(&fc_b, m));
    let proj_w_t = store.push(proj_w);
    let proj_b_t = store.push(broadcast(&proj_b, m));
    let fc = store.push(vec![]);
    let fc_rem = store.push(vec![]);
    let gelu_idx = store.push_idx(vec![]);
    let act = store.push(vec![]);
    let proj2 = store.push(vec![]);
    let proj2_rem = store.push(vec![]);
    ops.push(Op::Projection { x: h2, w: fc_w_t, bias: fc_b_t, out: fc, rem: fc_rem, m, k: D, n: FFN, shift });
    ops.push(Op::SoftmaxIndex { x: fc, out: gelu_idx, table_len: gelu_len });
    ops.push(Op::Lookup { idx: gelu_idx, out: act, table: gelu_t });
    ops.push(Op::Projection { x: act, w: proj_w_t, bias: proj_b_t, out: proj2, rem: proj2_rem, m, k: FFN, n: D, shift });
    let out = store.push(vec![]);
    ops.push(Op::Add { a: x2, b: proj2, c: out });

    let proof = prove_shard(&mut store, &ops, &[out], &mut rng);
    assert!(verify_shard(&store, &ops, &proof), "real-weight GPT-2 layer proof failed");
    println!("real GPT-2 L0 layer (m={}, {} heads) proven + verified OK", m, HEADS);
}
