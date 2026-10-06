//! Prove the full GPT-2 (12 layers, m=512) op graph with the shard-DAG composer
//! at several `ops_per_shard` granularities, verifying cross-shard binding on the
//! real model and measuring the granularity knob.

use std::fs;

use zkie_ops::compose::{causal_mask, prove_committed_shard_dag, verify_committed_shard_dag, Op, Store};
use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i32, to_i32, to_i64};
use zkie_core::pcs::whir::Whir;

const D: usize = 1024;
const FFN: usize = 4096;
const HEADS: usize = 12;
const DH: usize = 64;
const LAYERS: usize = 12;
const N_REAL: usize = 768;

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
) -> usize {
    let dir = "models/gpt2/weights";
    let q_w = load_i32(&format!("{dir}/L{layer}_q_w_i32.bin"));
    let k_w = load_i32(&format!("{dir}/L{layer}_k_w_i32.bin"));
    let v_w = load_i32(&format!("{dir}/L{layer}_v_w_i32.bin"));
    let o_w = load_i32(&format!("{dir}/L{layer}_o_proj_w_i32.bin"));
    let q_b = load_i32(&format!("{dir}/L{layer}_q_b_i32.bin"));
    let k_b = load_i32(&format!("{dir}/L{layer}_k_b_i32.bin"));
    let v_b = load_i32(&format!("{dir}/L{layer}_v_b_i32.bin"));
    let o_b = load_i32(&format!("{dir}/L{layer}_o_proj_b_i32.bin"));
    let ln1_w = load_i32(&format!("{dir}/L{layer}_ln1_w_i32.bin"));
    let ln1_b = load_i32(&format!("{dir}/L{layer}_ln1_b_i32.bin"));
    let ln2_w = load_i32(&format!("{dir}/L{layer}_ln2_w_i32.bin"));
    let ln2_b = load_i32(&format!("{dir}/L{layer}_ln2_b_i32.bin"));
    let fc_w = load_i32(&format!("{dir}/L{layer}_fc_w_i32.bin"));
    let fc_b = load_i32(&format!("{dir}/L{layer}_fc_b_i32.bin"));
    let proj_w = load_i32(&format!("{dir}/L{layer}_proj_w_i32.bin"));
    let proj_b = load_i32(&format!("{dir}/L{layer}_proj_b_i32.bin"));

    let table_len = 1usize << 21;
    let ln1_w_t = store.push(broadcast(&ln1_w, m));
    let ln1_b_t = store.push(broadcast(&ln1_b, m));
    let h = store.push(vec![]);
    ops.push(Op::LayerNormCentered { x, w: ln1_w_t, b: ln1_b_t, out: h, rsqrt_table: rsqrt_t, m, d: D, n_real: N_REAL });

    let mut head_outs = Vec::new();
    for head in 0..HEADS {
        let qh = store.push(slice_cols(&q_w, D, head * DH, (head + 1) * DH));
        let kh = store.push(slice_cols(&k_w, D, head * DH, (head + 1) * DH));
        let vh = store.push(slice_cols(&v_w, D, head * DH, (head + 1) * DH));
        let oh = store.push(slice_rows(&o_w, D, head * DH, (head + 1) * DH));
        let qb = store.push(broadcast(&q_b[head * DH..(head + 1) * DH], m));
        let kb = store.push(broadcast(&k_b[head * DH..(head + 1) * DH], m));
        let vb = store.push(broadcast(&v_b[head * DH..(head + 1) * DH], m));
        let ob = store.push(broadcast(&vec![from_i32(0); D], m));
        let q = store.push(vec![]);
        let k = store.push(vec![]);
        let v = store.push(vec![]);
        let q_scaled = store.push(vec![]);
        let k_scaled = store.push(vec![]);
        let qr = store.push(vec![]);
        let kr = store.push(vec![]);
        let vr = store.push(vec![]);
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
        ops.push(Op::Projection { x: h, w: qh, bias: qb, out: q, rem: qr, m, k: D, n: DH, shift });
        ops.push(Op::Projection { x: h, w: kh, bias: kb, out: k, rem: kr, m, k: D, n: DH, shift });
        ops.push(Op::Projection { x: h, w: vh, bias: vb, out: v, rem: vr, m, k: D, n: DH, shift });
        ops.push(Op::Scale { x: q, out: q_scaled, factor: 23170, shift: 16 });
        ops.push(Op::Scale { x: k, out: k_scaled, factor: 23170, shift: 16 });
        ops.push(Op::Transpose { x: k_scaled, out: kt, m, k: DH });
        ops.push(Op::MatMul { a: q_scaled, b: kt, c: scores, m, k: DH, n: m });
        ops.push(Op::Scale { x: scores, out: scores_16, factor: 1, shift: 16 });
        ops.push(Op::StableSoftmaxIndex { x: scores_16, mask: mask_t, out: idx, offset: 1 << 21, table_len, m, n: m });
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_t, m, n: m });
        ops.push(Op::MatMul { a: probs, b: v, c: attn, m, k: m, n: DH });
        ops.push(Op::Scale { x: attn, out: attn_16, factor: 1, shift: 16 });
        ops.push(Op::Projection { x: attn_16, w: oh, bias: ob, out: out_h, rem: out_rem, m, k: DH, n: D, shift });
        head_outs.push(out_h);
    }
    let mut attn_acc = head_outs[0];
    for &ho in &head_outs[1..] {
        let s = store.push(vec![]);
        ops.push(Op::Add { a: attn_acc, b: ho, c: s });
        attn_acc = s;
    }
    // Add the attention output-projection bias ONCE (after summing heads),
    // matching the float reference `x2 = x + sum(attn_heads) + o_b`.
    let o_b_t = store.push(broadcast(&o_b, m));
    let attn_biased = store.push(vec![]);
    ops.push(Op::Add { a: attn_acc, b: o_b_t, c: attn_biased });
    let x2 = store.push(vec![]);
    ops.push(Op::Add { a: x, b: attn_biased, c: x2 });

    let ln2_w_t = store.push(broadcast(&ln2_w, m));
    let ln2_b_t = store.push(broadcast(&ln2_b, m));
    let h2 = store.push(vec![]);
    ops.push(Op::LayerNormCentered { x: x2, w: ln2_w_t, b: ln2_b_t, out: h2, rsqrt_table: rsqrt_t, m, d: D, n_real: N_REAL });
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
    ops.push(Op::GeluIndex { x: fc, out: gelu_idx, offset: 1 << 23, table_len: 1 << 24 });
    ops.push(Op::Lookup { idx: gelu_idx, out: act, table: gelu_t });
    ops.push(Op::Projection { x: act, w: proj_w_t, bias: proj_b_t, out: proj2, rem: proj2_rem, m, k: FFN, n: D, shift });
    let out = store.push(vec![]);
    ops.push(Op::Add { a: x2, b: proj2, c: out });
    out
}

fn main() {
    let (m, shift) = (512usize, 16u32);
    let dir = "models/gpt2/weights";

    let exp_table = load_i32("models/gpt2/weights/exp_table_i32.bin");
    let rsqrt_table = load_i32("models/gpt2/weights/rsqrt_table_i32.bin");
    let gelu_table = load_i32(&format!("{dir}/gelu_table_i32.bin"));
    let embedding = load_i32(&format!("{dir}/embedding_i32.bin"));
    let lnf_w = load_i32(&format!("{dir}/ln_f_w_i32.bin"));
    let lnf_b = load_i32(&format!("{dir}/ln_f_b_i32.bin"));
    let lm_head_w = load_i32(&format!("{dir}/lm_head_w_i32.bin"));

    let mut store = Store::new();
    let x0 = store.push(embedding[..m * D].to_vec());
    let exp_t = store.push(exp_table);
    let rsqrt_t = store.push(rsqrt_table);
    let gelu_t = store.push(gelu_table);
    let mask_t = store.push(causal_mask(m));

    let mut ops = Vec::new();
    let mut x_cur = x0;
    for layer in 0..LAYERS {
        x_cur = build_layer(&mut store, &mut ops, x_cur, layer, m, shift, exp_t, gelu_t, rsqrt_t, mask_t);
    }

    // Final layernorm + lm_head (bare matmul, no bias).
    let lnf_w_t = store.push(broadcast(&lnf_w, m));
    let lnf_b_t = store.push(broadcast(&lnf_b, m));
    let h_final = store.push(vec![]);
    ops.push(Op::LayerNormCentered { x: x_cur, w: lnf_w_t, b: lnf_b_t, out: h_final, rsqrt_table: rsqrt_t, m, d: D, n_real: N_REAL });
    let lm_w_t = store.push(lm_head_w);
    let logits = store.push(vec![]);
    ops.push(Op::MatMul { a: h_final, b: lm_w_t, c: logits, m, k: D, n: 65536 });

    let gt = load_i32(&format!("{dir}/gt_argmax_512_i32.bin"));

    // WholeModel (~1 shard) and Layers(1) (176 ops/layer -> 13 shards).
    for ops_per_shard in [176usize] {
        let mut rng = XorShift64::new(0xBEEF);
        let t0 = std::time::Instant::now();
        let whir = Whir::new_testing(19);
        let proof = prove_committed_shard_dag(&mut store, &ops, ops_per_shard, &whir, &mut rng);
        let prove_t = t0.elapsed();
        let t1 = std::time::Instant::now();
        assert!(verify_committed_shard_dag(&mut store, &ops, ops_per_shard, &whir, &proof), "sharded proof failed");
        let verify_t = t1.elapsed();

        let logits = store.get(logits);
        let mut matches = 0;
        for i in 0..m {
            let mut best = 0usize;
            let mut best_v = logits[i * 65536];
            for j in 1..50257 {
                let v = logits[i * 65536 + j];
                if to_i64(v) > to_i64(best_v) {
                    best_v = v;
                    best = j;
                }
            }
            if best as i32 == to_i32(gt[i]) {
                matches += 1;
            }
        }

        println!(
            "ops_per_shard={} ({} shards, {} cross-binds): prove {:?}, verify {:?}, argmax {}/{}, open_stats={:?}, verify_stats={:?}, global_open={}",
            ops_per_shard,
            proof.shards.len(),
            proof.boundary_tensors.len(),
            prove_t,
            verify_t,
            matches,
            m,
            whir.open_stats(),
            whir.verify_stats(),
            zkie_core::pcs::whir::global_open_count(),
        );
    }
}
