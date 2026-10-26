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
use zkie_ops::compose::{Op, Store};

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
    let sel_idx = store.push_idx(vec![]);
    ops.push(Op::SelToIndices { sel, out: sel_idx, m, n });
    let sorted = store.push_idx(vec![]);
    ops.push(Op::SortIndices { idx: sel_idx, out: sorted, m, n: topk });
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
    let sel_idx = store.push_idx(vec![]);
    ops.push(Op::SelToIndices { sel, out: sel_idx, m, n });
    let sorted = store.push_idx(vec![]);
    ops.push(Op::SortIndices { idx: sel_idx, out: sorted, m, n: topk });
    (gate, sorted)
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
}

/// Builder entry point. Today it implements the validated layer-0 MLA
/// projection slice (`wq_a`, `wq_b`, `wkv`) and returns the projected `q` and
/// `kv` tensors. Attention, `wo_a`/`wo_b`, MoE, HyperConnection, and Engram are
/// layered on top next; this function will grow to the full 40-layer graph.
pub fn build_deepseek_v41(
    store: &mut Store,
    ops: &mut Vec<Op>,
    dir: &str,
    m: usize,
    shift: u32,
) -> (usize, usize) {
    let file = open_shard(&format!("{dir}/model-00003-of-00048.safetensors"));
    let x = store.push(vec![from_i64(1 << 16); m * DIM_PAD]);

    let wq_a = push_quantized_fp8(
        store,
        file.clone(),
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
    );
    let wq_b = push_quantized_fp8(
        store,
        file.clone(),
        "layers.0.attn.wq_b.weight",
        "layers.0.attn.wq_b.scale",
    );
    let wkv = push_quantized_fp8(
        store,
        file.clone(),
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wkv.scale",
    );

    let rsqrt_table = store.push(load_i32("models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let q_norm_real: Vec<i64> = file
        .read_bf16("layers.0.attn.q_norm.weight")
        .iter()
        .map(|&v| to_i64(v))
        .collect();
    let kv_norm_real: Vec<i64> = file
        .read_bf16("layers.0.attn.kv_norm.weight")
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
