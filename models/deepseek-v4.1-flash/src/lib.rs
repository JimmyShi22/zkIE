//! DeepSeek-V4.1-Flash text-only op-graph builder.
//!
//! The first milestone is a single quantized `BlockQuantizedMatMul` on the real
//! HuggingFace safetensors. It validates the FP8-E4M3/E8M0 codec and the native
//! quantized matmul proof against actual layer-0 weights. The full 40-layer
//! CSA2/CED/Engram graph is layered on top of this primitive.

use std::sync::Arc;

use zkie_core::common::quantized_io::{QuantDtype, SafetensorFile};
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_ops::compose::{causal_mask, Op, Store};

// Architecture constants from DeepSeek-V4.1-Flash `config.json` / `tech_clean.txt`.
pub const N_LAYERS: usize = 40;
pub const DIM: usize = 5120;
pub const N_HEADS: usize = 64;
pub const HEAD_DIM: usize = 512;
pub const Q_LORA_RANK: usize = 1280;
pub const O_LORA_RANK: usize = 1024;
pub const O_GROUPS: usize = 8;
pub const N_ROUTED_EXPERTS: usize = 384;
pub const N_SHARED_EXPERTS: usize = 1;
pub const N_ACTIVATED_EXPERTS: usize = 6;
pub const MOE_INTER_DIM: usize = 2304;
pub const SLIDING_WINDOW: usize = 128;
pub const HC_MULT: usize = 4;
pub const QUANT_BLOCK: usize = 32;
pub const VOCAB: usize = 129280;

// Field-level matmul/MLE primitives require power-of-two tensor lengths. These
// are the zero-padded dimensions the builder must use consistently everywhere;
// `Store::materialize` already pads quantized weights, and the builder must pad
// activations and BF16/F32 weights to the same values.
pub const DIM_PAD: usize = DIM.next_power_of_two();
pub const Q_LORA_RANK_PAD: usize = Q_LORA_RANK.next_power_of_two();
pub const O_LORA_RANK_PAD: usize = O_LORA_RANK.next_power_of_two();
pub const MOE_INTER_DIM_PAD: usize = MOE_INTER_DIM.next_power_of_two();
pub const N_ROUTED_EXPERTS_PAD: usize = N_ROUTED_EXPERTS.next_power_of_two();
pub const VOCAB_PAD: usize = VOCAB.next_power_of_two();
pub const HEAD_QK: usize = 64;
pub const HEAD_NOPE: usize = HEAD_DIM - HEAD_QK;
pub const HEAD_QK_PAD: usize = HEAD_QK.next_power_of_two();
pub const HEAD_NOPE_PAD: usize = HEAD_NOPE.next_power_of_two();

pub fn open_shard(path: &str) -> Arc<SafetensorFile> {
    Arc::new(
        SafetensorFile::open(path)
            .unwrap_or_else(|e| panic!("open safetensors shard {path}: {e}")),
    )
}

pub fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn broadcast_norm(v: &[i64], m: usize, pad: usize) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; m * pad];
    for i in 0..m {
        for j in 0..v.len().min(pad) {
            out[i * pad + j] = from_i64(v[j]);
        }
    }
    out
}

/// Slice a row-major `[rows x cols]` matrix (already dequantized to `i64`) into
/// the contiguous row band `[r0, r1)`. Used to split the group-interleaved
/// `wo_a` weight into one `[O_LORA_RANK x per_group]` block per output group.
fn slice_rows(mat: &[i64], cols: usize, r0: usize, r1: usize) -> Vec<Goldilocks> {
    (r0..r1)
        .flat_map(|r| mat[r * cols..(r + 1) * cols].iter().map(|&v| from_i64(v)))
        .collect()
}

/// Slice a row-major `[rows x cols]` matrix into the contiguous column band
/// `[c0, c1)`. Used to split `wo_b` into one `[DIM_PAD x O_LORA_RANK]` block per
/// output group.
fn slice_cols(mat: &[i64], cols: usize, c0: usize, c1: usize) -> Vec<Goldilocks> {
    let rows = mat.len() / cols;
    (0..rows)
        .flat_map(|r| mat[r * cols + c0..r * cols + c1].iter().map(|&v| from_i64(v)))
        .collect()
}

/// RoPE cos/sin tables for a `qk`-wide rotary tail, at fixed point 2^16.
pub fn build_rope_tables(store: &mut Store, qk: usize) -> (usize, usize) {
    let half = qk / 2;
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for i in 0..half {
        let theta = (10000f64).powf(-2.0 * i as f64 / qk as f64);
        cos.push(from_i64((theta.cos() * 65536.0).round() as i64));
        sin.push(from_i64((theta.sin() * 65536.0).round() as i64));
    }
    (store.push(cos), store.push(sin))
}

/// Load a BF16 safetensors matrix (fixed point 2^16) transposed, so it can feed
/// `MatMul`/`Projection` in `[k, n]` layout (input cols x output rows), padded
/// to powers of two. `rows`/`cols` are the raw safetensors shape `[rows, cols]`;
/// the returned tensor is `[cols_pad, rows_pad]`.
fn push_bf16_transposed(
    store: &mut Store,
    file: Arc<SafetensorFile>,
    name: &str,
    rows: usize,
    cols: usize,
) -> usize {
    let v: Vec<i64> = file.read_bf16(name).iter().map(|&x| to_i64(x)).collect();
    let rows_pad = rows.next_power_of_two();
    let cols_pad = cols.next_power_of_two();
    let mut out = vec![Goldilocks::ZERO; cols_pad * rows_pad];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows_pad + r] = from_i64(v[r * cols + c]);
        }
    }
    store.push(out)
}

/// Load an F32 safetensors 1-D tensor (fixed point 2^16) padded to `n_pad`.
fn push_f32_padded(store: &mut Store, file: Arc<SafetensorFile>, name: &str, n_pad: usize) -> usize {
    let v: Vec<i64> = file.read_f32(name).iter().map(|&x| to_i64(x)).collect();
    let mut out = vec![Goldilocks::ZERO; n_pad];
    for j in 0..v.len().min(n_pad) {
        out[j] = from_i64(v[j]);
    }
    store.push(out)
}

/// MoE routing gate: `x -> top-k selection + normalized gate weights`.
///
/// ```text
/// z       = x @ gate_w^T
/// s       = sqrt(softplus(z))            (table lookup)
/// biased  = s + gate_b                    (row-broadcast)
/// sel     = top-6(biased)
/// weights = s * sel, then row-normalized  (SumNorm)
/// ```
///
/// Returns `(sel, thr, weights, d1, d2)`; `weights` is the `[m, N_EXPERTS_PAD]`
/// gate table (normalized scores at the selected experts, zero elsewhere).
pub fn build_moe_gate(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    x: usize,
    m: usize,
    shift: u32,
    sqrt_softplus_table: usize,
) -> (usize, usize, usize, usize, usize) {
    let n = N_ROUTED_EXPERTS_PAD;
    let gate_w = push_bf16_transposed(
        store,
        file.clone(),
        &format!("layers.{layer}.ffn.gate.weight"),
        N_ROUTED_EXPERTS,
        DIM,
    );
    let gate_b = push_f32_padded(
        store,
        file.clone(),
        &format!("layers.{layer}.ffn.gate.bias"),
        n,
    );

    let z_raw = store.push(vec![]);
    ops.push(Op::MatMul { a: x, b: gate_w, c: z_raw, m, k: DIM_PAD, n });
    let z = store.push(vec![]);
    ops.push(Op::Scale { x: z_raw, out: z, factor: 1, shift });

    let s_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: z, out: s_idx, offset: 1 << 23, table_len: 1 << 24 });
    let s = store.push(vec![]);
    ops.push(Op::Lookup { idx: s_idx, out: s, table: sqrt_softplus_table });

    let bias_mn = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![gate_b; m], out: bias_mn });
    let biased = store.push(vec![]);
    ops.push(Op::Add { a: s, b: bias_mn, c: biased });

    let sel = store.push(vec![]);
    let thr = store.push(vec![]);
    let _topk_gate = store.push(vec![]);
    let d1 = store.push(vec![]);
    let d2 = store.push(vec![]);
    ops.push(Op::TopKSelect {
        x: biased,
        sel,
        thr,
        gate: _topk_gate,
        d1,
        d2,
        m,
        n,
        k: N_ACTIVATED_EXPERTS,
    });

    let s_sel = store.push(vec![]);
    ops.push(Op::ScaleVec { x: s, scale: sel, out: s_sel, shift: 0 });
    let weights = store.push(vec![]);
    ops.push(Op::SumNorm { x: s_sel, out: weights, m, n });
    (sel, thr, weights, d1, d2)
}

/// Shared expert SwiGLU MLP: `out = (silu(clip(x @ w1^T, 10)) * clip(x @ w3^T,
/// -10, 10)) @ w2^T`. All three weights are FP8 block-quantized; the activation
/// and clipping are table lookups at the GELU fixed-point convention.
pub fn build_shared_expert(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    x: usize,
    m: usize,
    shift: u32,
    silu_table: usize,
    clip_table: usize,
) -> usize {
    let w1 = push_quantized_fp8(
        store,
        file.clone(),
        &format!("layers.{layer}.ffn.shared_experts.w1.weight"),
        &format!("layers.{layer}.ffn.shared_experts.w1.scale"),
    );
    let w2 = push_quantized_fp8(
        store,
        file.clone(),
        &format!("layers.{layer}.ffn.shared_experts.w2.weight"),
        &format!("layers.{layer}.ffn.shared_experts.w2.scale"),
    );
    let w3 = push_quantized_fp8(
        store,
        file.clone(),
        &format!("layers.{layer}.ffn.shared_experts.w3.weight"),
        &format!("layers.{layer}.ffn.shared_experts.w3.scale"),
    );

    let gate_raw = build_quantized_matmul(store, ops, x, w1, m, DIM_PAD, MOE_INTER_DIM_PAD);
    let gate = store.push(vec![]);
    ops.push(Op::Scale { x: gate_raw, out: gate, factor: 1, shift });
    let gate_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: gate, out: gate_idx, offset: 1 << 23, table_len: 1 << 24 });
    let gate_act = store.push(vec![]);
    ops.push(Op::Lookup { idx: gate_idx, out: gate_act, table: silu_table });

    let up_raw = build_quantized_matmul(store, ops, x, w3, m, DIM_PAD, MOE_INTER_DIM_PAD);
    let up = store.push(vec![]);
    ops.push(Op::Scale { x: up_raw, out: up, factor: 1, shift });
    let up_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: up, out: up_idx, offset: 1 << 23, table_len: 1 << 24 });
    let up_clip = store.push(vec![]);
    ops.push(Op::Lookup { idx: up_idx, out: up_clip, table: clip_table });

    let hidden = store.push(vec![]);
    ops.push(Op::ScaleVec { x: gate_act, scale: up_clip, out: hidden, shift: 16 });

    let out_raw = build_quantized_matmul(store, ops, hidden, w2, m, MOE_INTER_DIM_PAD, DIM_PAD);
    let out = store.push(vec![]);
    ops.push(Op::Scale { x: out_raw, out, factor: 1, shift });
    out
}

pub fn push_quantized_fp8(
    store: &mut Store,
    file: Arc<SafetensorFile>,
    weight: &str,
    scale: &str,
) -> usize {
    store.push_quantized(
        file,
        weight.to_string(),
        scale.to_string(),
        QuantDtype::Fp8E4M3,
        0,
        QUANT_BLOCK,
    )
}

pub fn push_quantized_fp4(
    store: &mut Store,
    file: Arc<SafetensorFile>,
    weight: &str,
    scale: &str,
    logical_cols: usize,
) -> usize {
    store.push_quantized(
        file,
        weight.to_string(),
        scale.to_string(),
        QuantDtype::Int8,
        logical_cols,
        QUANT_BLOCK,
    )
}

/// Build a one-op graph for `x @ W^T`, where `W` is a quantized safetensors
/// tensor and `m` is the token/row count. `x` must already be inserted in the
/// store. This is the layer-0 quantization primitive, kept intentionally small
/// so the block-quantized proof path can be tested before the full model.
pub fn build_quantized_matmul(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    w: usize,
    m: usize,
    k: usize,
    n: usize,
) -> usize {
    let c = store.push(vec![]);
    ops.push(Op::BlockQuantizedMatMul {
        a: x,
        w,
        c,
        m,
        k,
        n,
        block_size: QUANT_BLOCK,
    });
    c
}

/// Full-mode SparseIndexer assembled from existing primitives: score the
/// candidates, apply the compress visibility mask, take top-k, then sort the
/// selected indices into position order. `w` is `n x k` (candidate count x
/// feature dim). Returns `(gate, sorted_indices)`.
pub fn build_sparse_indexer_full(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    w: usize,
    m: usize,
    k: usize,
    n: usize,
    ratio: usize,
    topk: usize,
    shift: u32,
) -> (usize, usize) {
    let raw = build_quantized_matmul(store, ops, x, w, m, k, n);
    let score = store.push(vec![]);
    ops.push(Op::Scale { x: raw, out: score, factor: 1, shift });
    let mask = store.push(vec![]);
    ops.push(Op::CompressMask { out: mask, m, n, ratio });
    let masked = store.push(vec![]);
    ops.push(Op::Add { a: score, b: mask, c: masked });
    let sel = store.push(vec![]);
    let thr = store.push(vec![]);
    let gate = store.push(vec![]);
    let d1 = store.push(vec![]);
    let d2 = store.push(vec![]);
    ops.push(Op::TopKSelect { x: masked, sel, thr, gate, d1, d2, m, n, k: topk });
    let sorted = store.push_idx(vec![]);
    ops.push(Op::SortSelected { sel, out: sorted, m, n, k: topk });
    (gate, sorted)
}

pub fn build_sparse_indexer_reindex(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    w: usize,
    m: usize,
    k: usize,
    n: usize,
    ratio: usize,
    block_size: usize,
    topk_blocks: usize,
    topk: usize,
    shift: u32,
) -> (usize, usize) {
    let raw = build_quantized_matmul(store, ops, x, w, m, k, n);
    let score = store.push(vec![]);
    ops.push(Op::Scale { x: raw, out: score, factor: 1, shift });
    let mask = store.push(vec![]);
    ops.push(Op::CompressMask { out: mask, m, n, ratio });
    let masked = store.push(vec![]);
    ops.push(Op::Add { a: score, b: mask, c: masked });
    let cand = store.push(vec![]);
    ops.push(Op::CandidateBlocks { x: masked, out: cand, m, n, block_size, topk_blocks });
    let gated = store.push(vec![]);
    ops.push(Op::ScaleVec { x: masked, scale: cand, out: gated, shift: 0 });
    let sel = store.push(vec![]);
    let thr = store.push(vec![]);
    let gate = store.push(vec![]);
    let d1 = store.push(vec![]);
    let d2 = store.push(vec![]);
    ops.push(Op::TopKSelect { x: gated, sel, thr, gate, d1, d2, m, n, k: topk });
    let sorted = store.push_idx(vec![]);
    ops.push(Op::SortSelected { sel, out: sorted, m, n, k: topk });
    (gate, sorted)
}

/// HyperConnection sublayer wrap for a precomputed `mixes` tensor. The sublayer
/// is currently the identity (`y' = y`); attention/FFN sublayers are inserted
/// between `HyperPre` and `HyperPost` by the full block builder.
pub fn build_hyperconnection_mix(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    mixes: usize,
    scale: usize,
    base: usize,
    m: usize,
    hc: usize,
    d: usize,
    iters: usize,
    eps: i64,
) -> usize {
    let pre = store.push(vec![]);
    let post = store.push(vec![]);
    let comb = store.push(vec![]);
    ops.push(Op::SinkhornSplit { mixes, scale, base, pre, post, comb, m, hc, iters, eps });
    let y = store.push(vec![]);
    ops.push(Op::HyperPre { x, pre, out: y, m, hc, d });
    // Identity sublayer for this slice; replace with attention/FFN output.
    let y2 = y;
    let out = store.push(vec![]);
    ops.push(Op::HyperPost { y: y2, residual: x, post, comb, out, m, hc, d });
    out
}

/// Engram table lookup + wkv projection. `hash_idx` is the output of
/// `EngramHash`; `embed` is the quantized FP8 table (dequantized on demand),
/// `wkv` is the quantized output projection weight. Gating is applied by the
/// caller as an extra projection before this step.
pub fn build_engram_lookup(
    store: &mut Store,
    ops: &mut Vec<Op>,
    hash_idx: usize,
    embed: usize,
    wkv: usize,
    n: usize,
    dim: usize,
    out_dim: usize,
) -> usize {
    let e = store.push(vec![]);
    ops.push(Op::GatherRows { idx: hash_idx, table: embed, out: e, dim });
    let raw = build_quantized_matmul(store, ops, e, wkv, n, dim, out_dim);
    let out = store.push(vec![]);
    ops.push(Op::Scale { x: raw, out, factor: 1, shift: 16 });
    out
}

/// Provable row gather: one `Lookup` per column, then concat. Each column table
/// is a public constant, so the gather is covered by the existing lookup proof.
pub fn build_gather_rows_proven(
    store: &mut Store,
    ops: &mut Vec<Op>,
    idx: usize,
    table: usize,
    dim: usize,
) -> usize {
    let vals = store.materialize(table).into_owned();
    let rows = vals.len() / dim;
    let mut cols = Vec::with_capacity(dim);
    for j in 0..dim {
        let col = (0..rows).map(|r| vals[r * dim + j]).collect::<Vec<_>>();
        cols.push(store.push(col));
    }
    let mut parts = Vec::with_capacity(dim);
    for j in 0..dim {
        let o = store.push(vec![]);
        ops.push(Op::Lookup { idx, out: o, table: cols[j] });
        parts.push(o);
    }
    let out = store.push(vec![]);
    ops.push(Op::Concat { parts, out });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::XorShift64;
    use zkie_core::common::fixed_point::from_i64;
    use zkie_ops::compose::{prove_shard, verify_shard};

    #[test]
    fn sparse_indexer_full_roundtrip() {
        let m = 2usize;
        let k = 2usize;
        let n = 4usize;
        let mut store = Store::new();
        let mut ops = Vec::new();
        let x = store.push(vec![from_i64(1 << 16); m * k]);
        let w = store.push((0..n * k).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
        let (gate, sorted) = build_sparse_indexer_full(&mut store, &mut ops, x, w, m, k, n, 1, 2, 16);
        let mut rng = XorShift64::new(7);
        let proof = prove_shard(&mut store, &ops, &[], &mut rng);
        assert!(verify_shard(&store.owned_slices(), &ops, &proof));
        assert_eq!(store.get(gate).len(), m * n);
        assert_eq!(store.idx[sorted].len(), m * 2);
    }

    #[test]
    fn sparse_indexer_reindex_roundtrip() {
        let m = 2usize;
        let k = 2usize;
        let n = 4usize;
        let mut store = Store::new();
        let mut ops = Vec::new();
        let x = store.push(vec![from_i64(1 << 16); m * k]);
        let w = store.push((0..n * k).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
        let (gate, sorted) = build_sparse_indexer_reindex(&mut store, &mut ops, x, w, m, k, n, 1, 2, 1, 2, 16);
        let mut rng = XorShift64::new(8);
        let proof = prove_shard(&mut store, &ops, &[], &mut rng);
        assert!(verify_shard(&store.owned_slices(), &ops, &proof));
        assert_eq!(store.idx[sorted].len(), m * 2);
        assert_eq!(store.get(gate).len(), m * n);
    }

    #[test]
    fn hyperconnection_mix_roundtrip() {
        let m = 1usize;
        let hc = 4usize;
        let d = 2usize;
        let mix_hc = (2 + hc) * hc;
        let mut store = Store::new();
        let mut ops = Vec::new();
        let x = store.push((0..m * hc * d).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
        let mixes = store.push((0..m * mix_hc).map(|i| from_i64((i as i64 + 1) * 100)).collect::<Vec<_>>());
        let scale = store.push((0..3).map(|i| from_i64(65536 / (i as i64 + 1))).collect::<Vec<_>>());
        let base = store.push((0..mix_hc).map(|i| from_i64(i as i64 * 10)).collect::<Vec<_>>());
        let out = build_hyperconnection_mix(&mut store, &mut ops, x, mixes, scale, base, m, hc, d, 20, 1);
        let mut rng = XorShift64::new(11);
        let proof = prove_shard(&mut store, &ops, &[], &mut rng);
        assert!(verify_shard(&store.owned_slices(), &ops, &proof));
        assert_eq!(store.get(out).len(), m * hc * d);
    }

    #[test]
    fn engram_lookup_roundtrip() {
        let n = 2usize;
        let dim = 4usize;
        let out_dim = 2usize;
        let mut store = Store::new();
        let mut ops = Vec::new();
        let table = store.push((0..8 * dim).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
        let idx = store.push_idx(vec![0, 2]);
        let wkv = store.push((0..out_dim * dim).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
        let out = build_engram_lookup(&mut store, &mut ops, idx, table, wkv, n, dim, out_dim);
        let mut rng = XorShift64::new(12);
        let proof = prove_shard(&mut store, &ops, &[], &mut rng);
        assert!(verify_shard(&store.owned_slices(), &ops, &proof));
        assert_eq!(store.get(out).len(), n * out_dim);
    }

    #[test]
    fn gather_rows_proven_roundtrip() {
        let mut store = Store::new();
        let mut ops = Vec::new();
        let table = store.push((0..12).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
        let idx = store.push_idx(vec![0, 2]);
        let out = build_gather_rows_proven(&mut store, &mut ops, idx, table, 4);
        let mut rng = XorShift64::new(13);
        let proof = prove_shard(&mut store, &ops, &[], &mut rng);
        assert!(verify_shard(&store.owned_slices(), &ops, &proof));
        assert_eq!(store.get(out).len(), 2 * 4);
    }
}

/// MLA input projection for one layer: `x -> (q, kv)` through the quantized
/// `wq_a`, `wq_b` and `wkv` projections plus `q_norm` / `kv_norm`. `x` must
/// already be materialized in the store as `[m, DIM_PAD]`. The returned `q` is
/// `[m, N_HEADS * HEAD_DIM]` (each head laid out as `nope || rope`) and `kv` is
/// `[m, HEAD_DIM]` (`nope || rope`).
pub fn build_deepseek_v41(
    store: &mut Store,
    ops: &mut Vec<Op>,
    dir: &str,
    layer: usize,
    m: usize,
    shift: u32,
) -> (usize, usize) {
    let file = open_shard(&format!("{dir}/model-00003-of-00048.safetensors"));
    let x = store.push(vec![from_i64(1 << 16); m * DIM_PAD]);

    let wq_a = push_quantized_fp8(
        store,
        file.clone(),
        &format!("layers.{layer}.attn.wq_a.weight"),
        &format!("layers.{layer}.attn.wq_a.scale"),
    );
    let wq_b = push_quantized_fp8(
        store,
        file.clone(),
        &format!("layers.{layer}.attn.wq_b.weight"),
        &format!("layers.{layer}.attn.wq_b.scale"),
    );
    let wkv = push_quantized_fp8(
        store,
        file.clone(),
        &format!("layers.{layer}.attn.wkv.weight"),
        &format!("layers.{layer}.attn.wkv.scale"),
    );

    let rsqrt_table = store.push(load_i32("models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let q_norm_real: Vec<i64> = file
        .read_bf16(&format!("layers.{layer}.attn.q_norm.weight"))
        .iter()
        .map(|&v| to_i64(v))
        .collect();
    let kv_norm_real: Vec<i64> = file
        .read_bf16(&format!("layers.{layer}.attn.kv_norm.weight"))
        .iter()
        .map(|&v| to_i64(v))
        .collect();
    let q_norm = store.push(broadcast_norm(&q_norm_real, m, Q_LORA_RANK_PAD));
    let kv_norm = store.push(broadcast_norm(&kv_norm_real, m, HEAD_DIM));

    let qr_raw = build_quantized_matmul(store, ops, x, wq_a, m, DIM_PAD, Q_LORA_RANK_PAD);
    let qr = store.push(vec![]);
    ops.push(Op::Scale {
        x: qr_raw,
        out: qr,
        factor: 1,
        shift,
    });
    let qrn = store.push(vec![]);
    ops.push(Op::RmsNorm {
        x: qr,
        w: q_norm,
        out: qrn,
        rsqrt_table,
        limb_table,
        m,
        d: Q_LORA_RANK_PAD,
        n_real: Q_LORA_RANK,
    });
    let q_raw = build_quantized_matmul(
        store,
        ops,
        qrn,
        wq_b,
        m,
        Q_LORA_RANK_PAD,
        N_HEADS * HEAD_DIM,
    );
    let q = store.push(vec![]);
    ops.push(Op::Scale {
        x: q_raw,
        out: q,
        factor: 1,
        shift,
    });

    let kv_raw = build_quantized_matmul(store, ops, x, wkv, m, DIM_PAD, HEAD_DIM);
    let kv = store.push(vec![]);
    ops.push(Op::Scale {
        x: kv_raw,
        out: kv,
        factor: 1,
        shift,
    });
    let kvn = store.push(vec![]);
    ops.push(Op::RmsNorm {
        x: kv,
        w: kv_norm,
        out: kvn,
        rsqrt_table,
        limb_table,
        m,
        d: HEAD_DIM,
        n_real: HEAD_DIM,
    });
    (q, kvn)
}

/// Dense multi-head attention over the projected `q` / `kv` tensors, with the
/// per-head attention sink (`attn_sink`) applied as a softmax bias. `kv` is the
/// shared `[m, HEAD_DIM]` latent (`nope || rope`); each head slices its own
/// `[m, N_HEADS * HEAD_DIM]` query block and RoPE-rotates the rope tail. Returns
/// the concatenated attention output `[m, N_HEADS * HEAD_DIM]`.
///
/// This is the current validated slice (dense causal attention + sink). The
/// sliding-window compressed-KV top-k path (SparseAttention) is layered on top
/// next.
#[allow(clippy::too_many_arguments)]
pub fn build_mla_attention(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    m: usize,
    shift: u32,
    q: usize,
    kv: usize,
    cos_t: usize,
    sin_t: usize,
    exp_table: usize,
    limb_table: usize,
) -> usize {
    let nope = HEAD_NOPE;
    let qk = HEAD_QK;
    let nope_pad = HEAD_NOPE_PAD;
    let mask = store.push(causal_mask(m));

    let kvn = store.push(vec![]);
    let kvr = store.push(vec![]);
    ops.push(Op::Slice { x: kv, out: kvn, start: 0, len: m * nope });
    ops.push(Op::Slice { x: kv, out: kvr, start: m * nope, len: m * qk });
    let zero_n = store.push(vec![Goldilocks::ZERO; m * (nope_pad - nope)]);
    let kvn_pad = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvn, zero_n], out: kvn_pad });
    let ktn = store.push(vec![]);
    let ktr = store.push(vec![]);
    ops.push(Op::Transpose { x: kvn_pad, out: ktn, m, k: nope_pad });
    ops.push(Op::Transpose { x: kvr, out: ktr, m, k: qk });

    let mut head_outs = Vec::new();
    for head in 0..N_HEADS {
        let qn = store.push(vec![]);
        let qr = store.push(vec![]);
        ops.push(Op::Slice { x: q, out: qn, start: head * HEAD_DIM, len: nope });
        ops.push(Op::Slice { x: q, out: qr, start: head * HEAD_DIM + nope, len: qk });
        let qn_pad = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![qn, zero_n], out: qn_pad });
        let qr_rot = store.push(vec![]);
        ops.push(Op::RoPE {
            x: qr,
            cos: cos_t,
            sin: sin_t,
            out: qr_rot,
            range_table: limb_table,
            m,
            d: qk,
            shift,
        });
        let scores_n = store.push(vec![]);
        let scores_r = store.push(vec![]);
        let scores = store.push(vec![]);
        ops.push(Op::MatMul { a: qn_pad, b: ktn, c: scores_n, m, k: nope_pad, n: m });
        ops.push(Op::MatMul { a: qr_rot, b: ktr, c: scores_r, m, k: qk, n: m });
        ops.push(Op::Add { a: scores_n, b: scores_r, c: scores });
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let probs_bias = store.push(vec![]);
        ops.push(Op::StableSoftmaxIndex {
            x: scores,
            mask,
            out: idx,
            offset: 1 << 21,
            table_len: 1 << 21,
            m,
            n: m,
        });
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_table, m, n: m });
        let sink_fixed = {
            let v = to_i64(file.read_f32(&format!("layers.{layer}.attn.attn_sink"))[head]);
            ((v as f64 / 65536.0).exp() * 65536.0).round() as i64
        };
        let sink_bias = store.push(vec![from_i64(sink_fixed); m]);
        ops.push(Op::SoftmaxBias { e, bias: sink_bias, out: probs_bias, m, n: m });
        let attn_n = store.push(vec![]);
        let attn_r = store.push(vec![]);
        ops.push(Op::MatMul { a: probs_bias, b: kvn_pad, c: attn_n, m, k: m, n: nope_pad });
        ops.push(Op::MatMul { a: probs_bias, b: kvr, c: attn_r, m, k: m, n: qk });
        let attn_n16 = store.push(vec![]);
        let attn_r16 = store.push(vec![]);
        ops.push(Op::Scale { x: attn_n, out: attn_n16, factor: 1, shift });
        ops.push(Op::Scale { x: attn_r, out: attn_r16, factor: 1, shift });
        let attn_n_sl = store.push(vec![]);
        ops.push(Op::Slice { x: attn_n16, out: attn_n_sl, start: 0, len: m * nope });
        let head_out = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![attn_n_sl, attn_r16], out: head_out });
        head_outs.push(head_out);
    }

    // Concatenate all heads into `o` [m, N_HEADS * HEAD_DIM].
    let mut o = head_outs[0];
    for &h in &head_outs[1..] {
        let cat = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![o, h], out: cat });
        o = cat;
    }
    o
}

/// MLA output projection: `o -> attention_out` through the group-interleaved
/// `wo_a` / `wo_b` low-rank projection (`O_GROUPS` groups of `O_LORA_RANK`).
/// `o` is `[m, N_HEADS * HEAD_DIM]`; the result is `[m, DIM_PAD]`.
pub fn build_oproj(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    m: usize,
    shift: u32,
    o: usize,
) -> usize {
    let wa = push_quantized_fp8(
        store,
        file.clone(),
        &format!("layers.{layer}.attn.wo_a.weight"),
        &format!("layers.{layer}.attn.wo_a.scale"),
    );
    let wb = push_quantized_fp8(
        store,
        file.clone(),
        &format!("layers.{layer}.attn.wo_b.weight"),
        &format!("layers.{layer}.attn.wo_b.scale"),
    );
    let wa_mat: Vec<i64> = store.materialize(wa).iter().map(|&v| to_i64(v)).collect();
    let wb_mat: Vec<i64> = store.materialize(wb).iter().map(|&v| to_i64(v)).collect();

    let per_group = N_HEADS * HEAD_DIM / O_GROUPS;
    let mut outs = Vec::with_capacity(O_GROUPS);
    for g in 0..O_GROUPS {
        let o_g = store.push(vec![]);
        ops.push(Op::Slice {
            x: o,
            out: o_g,
            start: g * per_group,
            len: per_group,
        });
        let wa_g = store.push(slice_rows(
            &wa_mat,
            per_group,
            g * O_LORA_RANK,
            (g + 1) * O_LORA_RANK,
        ));
        let oa_raw = build_quantized_matmul(store, ops, o_g, wa_g, m, per_group, O_LORA_RANK);
        let oa = store.push(vec![]);
        ops.push(Op::Scale { x: oa_raw, out: oa, factor: 1, shift });
        let wb_g = store.push(slice_cols(
            &wb_mat,
            O_GROUPS * O_LORA_RANK,
            g * O_LORA_RANK,
            (g + 1) * O_LORA_RANK,
        ));
        let out_raw = build_quantized_matmul(store, ops, oa, wb_g, m, O_LORA_RANK, DIM_PAD);
        let out = store.push(vec![]);
        ops.push(Op::Scale { x: out_raw, out, factor: 1, shift });
        outs.push(out);
    }
    let mut acc = outs[0];
    for &out in &outs[1..] {
        let s = store.push(vec![]);
        ops.push(Op::Add { a: acc, b: out, c: s });
        acc = s;
    }
    acc
}
