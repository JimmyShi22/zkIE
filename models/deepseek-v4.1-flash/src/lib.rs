//! DeepSeek-V4.1-Flash text-only op-graph builder.
//!
//! The first milestone is a single quantized `BlockQuantizedMatMul` on the real
//! HuggingFace safetensors. It validates the FP8-E4M3/E8M0 codec and the native
//! quantized matmul proof against actual layer-0 weights. The full 40-layer
//! CSA2/CED/Engram graph is layered on top of this primitive.

use std::sync::Arc;

use zkie_core::common::quantized_io::{QuantDtype, SafetensorFile};
use zkie_core::common::fixed_point::from_i64;
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

    let qr_raw = build_quantized_matmul(store, ops, x, wq_a, m, DIM_PAD, Q_LORA_RANK_PAD);
    let qr = store.push(vec![]);
    ops.push(Op::Scale {
        x: qr_raw,
        out: qr,
        factor: 1,
        shift,
    });
    let q_raw = build_quantized_matmul(
        store,
        ops,
        qr,
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
    (q, kv)
}
