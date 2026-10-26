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
use zkie_ops::compose::{Op, QuantizedExpert, Store};

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
/// MoE routed-expert scaling (`routed_scaling_factor = 1.5`) in Q16. The
/// official `Gate.forward` multiplies the row-normalized top-k scores by this
/// factor after normalization.
pub const ROUTE_SCALE_Q16: i64 = 98304;
/// `hc_eps = 1e-6` in Q16 rounds to zero (one Q16 LSB is ~1.5e-5), so the
/// Sinkhorn split runs with `eps = 0`.
pub const HC_EPS_Q16: i64 = 0;
pub const ENGRAM_HEAD_DIM: usize = 256;
pub const ENGRAM_N_HASH: usize = 24;

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

/// Shard file owning a layer's non-engram weights. Layer `l` lives in
/// `model-{l+3:05d}-of-00048.safetensors` (verified against the checkpoint's
/// `model.safetensors.index.json`).
pub fn layer_shard(dir: &str, layer: usize) -> String {
    format!("{dir}/model-{:05}-of-00048.safetensors", layer + 3)
}

/// Per-layer compress ratio for the 40 backbone layers (0,1: 0; 2..=19: 2;
/// 20..=39: 1), matching `config.json` `compress_ratios`.
pub fn compress_ratio(layer: usize) -> usize {
    if layer < 2 {
        0
    } else if layer <= 19 {
        2
    } else {
        1
    }
}

/// Layers that compress their own KV and publish it into the shared cache.
pub fn is_kv_source(layer: usize) -> bool {
    matches!(layer, 2 | 8 | 14 | 20)
}
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

fn round_div_i64(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
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

/// Gather `table[idx]` into a new tensor with one proven `Lookup` op.
fn gather(store: &mut Store, ops: &mut Vec<Op>, table: usize, idx: Vec<u32>) -> usize {
    let idx_t = store.push_idx(idx);
    let out = store.push(vec![]);
    ops.push(Op::Lookup { idx: idx_t, out, table });
    out
}

/// Transpose a row-major `[m, k]` tensor into `[k, m]` via a proven `Lookup`
/// with the public transpose index, instead of the unproven `Transpose` op.
fn transpose_gather(store: &mut Store, ops: &mut Vec<Op>, x: usize, m: usize, k: usize) -> usize {
    let idx = (0..k * m)
        .map(|ji| {
            let j = ji / m;
            let i = ji % m;
            (i * k + j) as u32
        })
        .collect::<Vec<_>>();
    gather(store, ops, x, idx)
}

/// Stack `parts` (each `[m, d]`) into `[m, len*d]` (token-major). The input
/// tensors are token-major `[m, d]`; the output interleaves them per token.
fn stack_rows(store: &mut Store, ops: &mut Vec<Op>, parts: &[usize], m: usize, d: usize) -> usize {
    let table = store.push(vec![]);
    ops.push(Op::Concat { parts: parts.to_vec(), out: table });
    let np = parts.len();
    let mut idx = Vec::with_capacity(m * np * d);
    for i in 0..m {
        for p in 0..np {
            for j in 0..d {
                idx.push((p * m * d + i * d + j) as u32);
            }
        }
    }
    gather(store, ops, table, idx)
}

/// Adjacent-pair RoPE permutation index for the whole `[m, qk]` tail: each
/// token's `qk` tail is permuted from natural interleaved order into
/// `[even, odd]`, matching the official DeepSeek `rope`.
fn rope_perm_rows(m: usize, qk: usize) -> Vec<u32> {
    let half = qk / 2;
    let mut p = Vec::with_capacity(m * qk);
    for i in 0..m {
        for j in 0..half {
            p.push((i * qk + 2 * j) as u32);
        }
        for j in 0..half {
            p.push((i * qk + 2 * j + 1) as u32);
        }
    }
    p
}

/// Per-head query gather indices for head `h` over the row-major
/// `[m, N_HEADS * HEAD_DIM]` query tensor: `(nope_idx, rope_idx)`.
fn head_query_indices(h: usize, m: usize) -> (Vec<u32>, Vec<u32>) {
    let stride = N_HEADS * HEAD_DIM;
    let base = h * HEAD_DIM;
    let mut nope = Vec::with_capacity(m * HEAD_NOPE);
    let mut rope = Vec::with_capacity(m * HEAD_QK);
    for i in 0..m {
        for j in 0..HEAD_NOPE {
            nope.push((i * stride + base + j) as u32);
        }
        for j in 0..HEAD_QK {
            rope.push((i * stride + base + HEAD_NOPE + j) as u32);
        }
    }
    (nope, rope)
}

/// Column-band gather indices over a row-major `[m, stride]` tensor: rows keep
/// their order, columns `[c0, c1)` are gathered within every row.
fn col_band(m: usize, stride: usize, c0: usize, c1: usize) -> Vec<u32> {
    let mut idx = Vec::with_capacity(m * (c1 - c0));
    for i in 0..m {
        for j in c0..c1 {
            idx.push((i * stride + j) as u32);
        }
    }
    idx
}

/// Pad each row of a row-major `[m, d]` tensor to `[m, d_pad]` with zeros, via
/// one proven `Lookup`. Unlike a flat `Concat`, the padding is interleaved
/// per row.
fn pad_rows(
    store: &mut Store,
    ops: &mut Vec<Op>,
    mat: usize,
    m: usize,
    d: usize,
    d_pad: usize,
) -> usize {
    let zeros = store.push(vec![Goldilocks::ZERO; m * (d_pad - d)]);
    let table = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![mat, zeros], out: table });
    let mut idx = Vec::with_capacity(m * d_pad);
    for i in 0..m {
        for j in 0..d {
            idx.push((i * d + j) as u32);
        }
        for j in 0..(d_pad - d) {
            idx.push((m * d + i * (d_pad - d) + j) as u32);
        }
    }
    gather(store, ops, table, idx)
}

/// Row-major interleave of two tensors `a [m, da]` and `b [m, db]` into
/// `[m, da+db]` (each row is `a` followed by `b`), via one proven `Lookup`.
fn interleave2(
    store: &mut Store,
    ops: &mut Vec<Op>,
    a: usize,
    b: usize,
    m: usize,
    da: usize,
    db: usize,
) -> usize {
    let table = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![a, b], out: table });
    let mut idx = Vec::with_capacity(m * (da + db));
    for i in 0..m {
        for j in 0..da {
            idx.push((i * da + j) as u32);
        }
        for j in 0..db {
            idx.push((m * da + i * db + j) as u32);
        }
    }
    gather(store, ops, table, idx)
}

/// Inverse of `rope_perm_rows`: map a `[m, qk]` tensor back from `[even, odd]`
/// order into natural interleaved order.
fn unrope_perm_rows(m: usize, qk: usize) -> Vec<u32> {
    let half = qk / 2;
    let mut p = Vec::with_capacity(m * qk);
    for i in 0..m {
        for j in 0..half {
            p.push((i * qk + j) as u32);
            p.push((i * qk + half + j) as u32);
        }
    }
    p
}

/// Expand a `[m, d]` tensor into `hc` identical copies laid out as
/// `[m, hc, d]` (token-major, copy-major), matching the official
/// `x.unsqueeze(2).repeat(1, 1, hc_mult, 1)` expansion.
pub fn expand_to_hc(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    m: usize,
    hc: usize,
    d: usize,
) -> usize {
    let parts = vec![x; hc];
    stack_rows(store, ops, &parts, m, d)
}

/// Gather the real `d_real` columns of each of the `hc` residual copies into a
/// contiguous `[m, hc * d_real]` tensor, dropping the per-copy zero padding.
/// `residual` is `[m, hc, d_stored]` with each copy padded to `d_stored` width.
fn flatten_hc_real(
    store: &mut Store,
    ops: &mut Vec<Op>,
    residual: usize,
    m: usize,
    hc: usize,
    d_real: usize,
    d_stored: usize,
) -> usize {
    let mut idx = Vec::with_capacity(m * hc * d_real);
    for i in 0..m {
        for h in 0..hc {
            for j in 0..d_real {
                idx.push((i * hc * d_stored + h * d_stored + j) as u32);
            }
        }
    }
    gather(store, ops, residual, idx)
}

/// RMSNorm over the first `n_real` columns of a `[m, d_pad]` tensor with a
/// per-row broadcast weight `w` (also `[m, d_pad]`, real entries in Q16).
fn build_rms_norm(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    w: usize,
    rsqrt_table: usize,
    limb_table: usize,
    m: usize,
    d_pad: usize,
    n_real: usize,
) -> usize {
    let out = store.push(vec![]);
    ops.push(Op::RmsNorm { x, w, out, rsqrt_table, limb_table, m, d: d_pad, n_real });
    out
}

/// Build `[m, n_pad]` where the first `m` columns are `mat [m, m]`, column `m`
/// is `col [m]` (the per-head attention sink), and the remaining columns are
/// zero. `n_pad` must be `(m + 1).next_power_of_two()`, keeping the softmax
/// length a power of two. Produced via one proven `Lookup`.
/// Build `[m, n_pad]` where the first `n_score` columns are `mat [m, n_score]`,
/// column `n_score` is `col [m]` (the per-head attention sink), and the
/// remaining columns are zero. `n_pad` must be `(n_score + 1).next_power_of_two()`.
fn pad_sink_matrix(
    store: &mut Store,
    ops: &mut Vec<Op>,
    mat: usize,
    col: usize,
    m: usize,
    n_score: usize,
    n_pad: usize,
) -> usize {
    let fill = n_pad - n_score - 1;
    let zeros = store.push(vec![Goldilocks::ZERO; m * fill]);
    let table = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![mat, col, zeros], out: table });
    let mut idx = Vec::with_capacity(m * n_pad);
    for i in 0..m {
        for j in 0..n_score {
            idx.push((i * n_score + j) as u32);
        }
        idx.push((m * n_score + i) as u32);
        for j in 0..fill {
            idx.push((m * n_score + m + i * fill + j) as u32);
        }
    }
    gather(store, ops, table, idx)
}

/// Position-dependent RoPE cos/sin tables for a `qk`-wide rotary tail, at fixed
/// point 2^16. `cos[p*half+i] = cos(p * 10000^(-2i/qk))`.
pub fn build_rope_tables(store: &mut Store, m: usize, qk: usize) -> (usize, usize) {
    let half = qk / 2;
    let mut cos = Vec::with_capacity(m * half);
    let mut sin = Vec::with_capacity(m * half);
    for p in 0..m {
        for i in 0..half {
            let theta = p as f64 * (10000f64).powf(-2.0 * i as f64 / qk as f64);
            cos.push(from_i64((theta.cos() * 65536.0).round() as i64));
            sin.push(from_i64((theta.sin() * 65536.0).round() as i64));
        }
    }
    (store.push(cos), store.push(sin))
}

/// RoPE cos/sin tables for arbitrary positions (e.g. compressed KV latents sit
/// at `g * ratio` instead of consecutive positions).
pub fn build_rope_tables_at(store: &mut Store, positions: &[usize], qk: usize) -> (usize, usize) {
    let half = qk / 2;
    let n = positions.len();
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for &p in positions {
        for i in 0..half {
            let theta = p as f64 * (10000f64).powf(-2.0 * i as f64 / qk as f64);
            cos.push(from_i64((theta.cos() * 65536.0).round() as i64));
            sin.push(from_i64((theta.sin() * 65536.0).round() as i64));
        }
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

/// Load an F32 safetensors 1-D tensor (fixed point 2^16) padded to `n_pad` with
/// `pad`. The MoE gate bias uses `pad = -(1<<30)` so the 384..512 padded expert
/// slots can never win a top-k selection (their `sqrt(softplus(0))` score is
/// otherwise ~0.83 and would compete with real experts).
fn push_f32_padded(
    store: &mut Store,
    file: Arc<SafetensorFile>,
    name: &str,
    n_pad: usize,
    pad: i64,
) -> usize {
    let v: Vec<i64> = file.read_f32(name).iter().map(|&x| to_i64(x)).collect();
    let mut out = vec![from_i64(pad); n_pad];
    for j in 0..v.len().min(n_pad) {
        out[j] = from_i64(v[j]);
    }
    store.push(out)
}

/// Load an F32 safetensors 1-D tensor (fixed point 2^16) unpadded.
fn push_f32_vec(store: &mut Store, file: Arc<SafetensorFile>, name: &str) -> usize {
    store.push(file.read_f32(name))
}

/// Load an F32 safetensors matrix transposed to `[cols_pad, rows_pad]` so it
/// can feed `MatMul` in `[k, n]` layout. `rows`/`cols` are the raw shape
/// `[rows, cols]`.
fn push_f32_transposed(
    store: &mut Store,
    file: Arc<SafetensorFile>,
    name: &str,
    rows: usize,
    cols: usize,
) -> usize {
    let v: Vec<i64> = file.read_f32(name).iter().map(|&x| to_i64(x)).collect();
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

/// HyperConnection mix: `mixes = rms(x4_flat) @ hc_fn^T` (bias added inside
/// `SinkhornSplit`). `x4_pad` is the 4-stream residual flattened and row-padded
/// to `d_pad` (a power of two); `hc_fn_t` is `[d_pad, mix_hc_pad]`.
pub fn build_hc_mixes(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x4_pad: usize,
    hc_fn_t: usize,
    m: usize,
    d_real: usize,
    d_pad: usize,
    mix_hc_pad: usize,
    rsqrt_table: usize,
    _limb_table: usize,
) -> usize {
    // Official `hc_mixes`: `mixes = (x @ hc_fn^T) * rsqrt`, i.e. project first
    // and scale by the per-row rsqrt afterwards. Doing the matmul before the
    // rsqrt avoids rounding the normalized input to Q16, which would otherwise
    // inject ~100 Q16 of error per projection.
    let mix_raw = store.push(vec![]);
    ops.push(Op::MatMul { a: x4_pad, b: hc_fn_t, c: mix_raw, m, k: d_pad, n: mix_hc_pad });
    let mix_q16 = store.push(vec![]);
    ops.push(Op::Scale { x: mix_raw, out: mix_q16, factor: 1, shift: 16 });
    let rstd = store.push(vec![]);
    ops.push(Op::Rsqrt {
        x: x4_pad,
        out: rstd,
        rsqrt_table,
        m,
        d: d_pad,
        n_real: d_real,
    });
    let mix_scaled = store.push(vec![]);
    ops.push(Op::BroadcastScale { x: mix_q16, scale: rstd, out: mix_scaled, h: mix_hc_pad });
    let mix = store.push(vec![]);
    ops.push(Op::Scale { x: mix_scaled, out: mix, factor: 1, shift: 16 });
    mix
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
    limb_table: usize,
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
        -(1i64 << 30),
    );

    let z_raw = store.push(vec![]);
    ops.push(Op::MatMul { a: x, b: gate_w, c: z_raw, m, k: DIM_PAD, n });
    let z = store.push(vec![]);
    ops.push(Op::Scale { x: z_raw, out: z, factor: 1, shift });

    let s_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: z, out: s_idx, offset: 1 << 23, table_len: 1 << 24, limb_table });
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
    let weights_norm = store.push(vec![]);
    ops.push(Op::SumNorm { x: s_sel, out: weights_norm, m, n });
    // Official `Gate.forward`: `weights /= sum; weights *= route_scale`.
    let weights = store.push(vec![]);
    ops.push(Op::Scale { x: weights_norm, out: weights, factor: ROUTE_SCALE_Q16, shift: 16 });
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
    limb_table: usize,
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
    ops.push(Op::GeluIndex { x: gate, out: gate_idx, offset: 1 << 23, table_len: 1 << 24, limb_table });
    let gate_act = store.push(vec![]);
    ops.push(Op::Lookup { idx: gate_idx, out: gate_act, table: silu_table });

    let up_raw = build_quantized_matmul(store, ops, x, w3, m, DIM_PAD, MOE_INTER_DIM_PAD);
    let up = store.push(vec![]);
    ops.push(Op::Scale { x: up_raw, out: up, factor: 1, shift });
    let up_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: up, out: up_idx, offset: 1 << 23, table_len: 1 << 24, limb_table });
    let up_clip = store.push(vec![]);
    ops.push(Op::Lookup { idx: up_idx, out: up_clip, table: clip_table });

    let hidden = store.push(vec![]);
    ops.push(Op::ScaleVec { x: gate_act, scale: up_clip, out: hidden, shift: 16 });

    let out_raw = build_quantized_matmul(store, ops, hidden, w2, m, MOE_INTER_DIM_PAD, DIM_PAD);
    let out = store.push(vec![]);
    ops.push(Op::Scale { x: out_raw, out, factor: 1, shift });
    out
}

/// Dequantize and stack all 384 routed experts' FP4 weights into the
/// `GatherMatMul` layout `[n_experts, in_pad, out_pad]`. `rows`/`cols` are the
/// raw safetensors shape `[out, in]`; the per-expert dequantized matrix is
/// transposed on the fly so `w_all[e][in][out]` matches `gather_matmul_forward`.
fn load_routed_expert_stack(
    store: &mut Store,
    file: Arc<SafetensorFile>,
    layer: usize,
    suffix: &str,
    rows: usize,
    cols: usize,
) -> usize {
    let rows_pad = rows.next_power_of_two();
    let cols_pad = cols.next_power_of_two();
    let mut experts = Vec::with_capacity(N_ROUTED_EXPERTS);
    for e in 0..N_ROUTED_EXPERTS {
        experts.push(QuantizedExpert {
            file: file.clone(),
            weight: format!("layers.{layer}.ffn.experts.{e}.{suffix}.weight"),
            scale: format!("layers.{layer}.ffn.experts.{e}.{suffix}.scale"),
            logical_cols: cols,
            block_size: QUANT_BLOCK,
        });
    }
    store.push_expert_stack(experts, cols_pad, rows_pad)
}

/// Routed experts: gather the top-k selected rows, run each expert's SwiGLU MLP
/// via `GatherMatMul`, and aggregate with the normalized routing-gate weights.
///
/// ```text
/// seg   = x[token_id]                       (gather selected token rows)
/// g     = seg @ w1^T, u = seg @ w3^T        (FP4, GatherMatMul)
/// h     = silu(clip(g,10)) * clip(u,-10,10)
/// d     = h @ w2^T
/// out   = sum_pos gate_weight[pos] * d[pos] (Gather)
/// ```
#[allow(clippy::too_many_arguments)]
pub fn build_routed_experts(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    x: usize,
    sel: usize,
    weights: usize,
    m: usize,
    shift: u32,
    silu_table: usize,
    clip_table: usize,
    limb_table: usize,
) -> usize {
    let n = N_ROUTED_EXPERTS_PAD;
    let s = m * N_ACTIVATED_EXPERTS;
    let s_pad = s.next_power_of_two();

    let w1_all = load_routed_expert_stack(store, file.clone(), layer, "w1", MOE_INTER_DIM, DIM);
    let w2_all = load_routed_expert_stack(store, file.clone(), layer, "w2", DIM, MOE_INTER_DIM);
    let w3_all = load_routed_expert_stack(store, file.clone(), layer, "w3", MOE_INTER_DIM, DIM);

    let token_id = store.push_idx(vec![]);
    let expert_of = store.push_idx(vec![]);
    ops.push(Op::Scatter { sel, token_id, expert_of, m, n });

    let token_id_pad = store.push_idx(vec![]);
    let expert_of_pad = store.push_idx(vec![]);
    ops.push(Op::PadIdx { base: token_id, out: token_id_pad, len: s_pad });
    ops.push(Op::PadIdx { base: expert_of, out: expert_of_pad, len: s_pad });

    let idx_seg = store.push_idx(vec![]);
    ops.push(Op::IdxBroadcast { base: token_id_pad, out: idx_seg, b: DIM_PAD });
    let seg = store.push(vec![]);
    ops.push(Op::Lookup { idx: idx_seg, out: seg, table: x });
    let mut mask = vec![Goldilocks::ZERO; s_pad];
    for i in 0..s {
        mask[i] = Goldilocks::ONE;
    }
    let mask_t = store.push(mask);
    let seg_z = store.push(vec![]);
    ops.push(Op::BroadcastScale { x: seg, scale: mask_t, out: seg_z, h: DIM_PAD });

    let g = store.push(vec![]);
    ops.push(Op::GatherMatMul {
        a: seg_z,
        expert_of: expert_of_pad,
        w: w1_all,
        c: g,
        s: s_pad,
        k: DIM_PAD,
        n: MOE_INTER_DIM_PAD,
        n_experts: N_ROUTED_EXPERTS,
    });
    let u = store.push(vec![]);
    ops.push(Op::GatherMatMul {
        a: seg_z,
        expert_of: expert_of_pad,
        w: w3_all,
        c: u,
        s: s_pad,
        k: DIM_PAD,
        n: MOE_INTER_DIM_PAD,
        n_experts: N_ROUTED_EXPERTS,
    });

    let g16 = store.push(vec![]);
    ops.push(Op::Scale { x: g, out: g16, factor: 1, shift });
    let u16 = store.push(vec![]);
    ops.push(Op::Scale { x: u, out: u16, factor: 1, shift });

    let g_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: g16, out: g_idx, offset: 1 << 23, table_len: 1 << 24, limb_table });
    let gate_act = store.push(vec![]);
    ops.push(Op::Lookup { idx: g_idx, out: gate_act, table: silu_table });
    let u_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: u16, out: u_idx, offset: 1 << 23, table_len: 1 << 24, limb_table });
    let up_clip = store.push(vec![]);
    ops.push(Op::Lookup { idx: u_idx, out: up_clip, table: clip_table });

    let hidden = store.push(vec![]);
    ops.push(Op::ScaleVec { x: gate_act, scale: up_clip, out: hidden, shift: 16 });

    let d = store.push(vec![]);
    ops.push(Op::GatherMatMul {
        a: hidden,
        expert_of: expert_of_pad,
        w: w2_all,
        c: d,
        s: s_pad,
        k: MOE_INTER_DIM_PAD,
        n: DIM_PAD,
        n_experts: N_ROUTED_EXPERTS,
    });
    let d16 = store.push(vec![]);
    ops.push(Op::Scale { x: d, out: d16, factor: 1, shift });

    let idx_gate = store.push_idx(vec![]);
    ops.push(Op::CombineIdx { a: token_id_pad, b: expert_of_pad, out: idx_gate, scale: n });
    let gate_pos = store.push(vec![]);
    ops.push(Op::Lookup { idx: idx_gate, out: gate_pos, table: weights });

    let acc = store.push(vec![]);
    ops.push(Op::Gather {
        weighted: d16,
        token_id: token_id_pad,
        gate: gate_pos,
        out: acc,
        m,
        h: DIM_PAD,
    });
    let out = store.push(vec![]);
    ops.push(Op::Scale { x: acc, out, factor: 1, shift });
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
    let (_pre, post, comb, y) =
        build_hc_enter(store, ops, x, mixes, scale, base, m, hc, d, iters, eps);
    // Identity sublayer for this slice; attention/FFN insert between enter/exit.
    let y2 = y;
    build_hc_exit(store, ops, y2, x, post, comb, m, hc, d)
}

/// First half of a HyperConnection sublayer: Sinkhorn-split the mixes, collapse
/// the `hc` residual streams with `pre`, and return the collapsed `y` plus the
/// `post`/`comb` coefficients the exit step needs.
#[allow(clippy::too_many_arguments)]
pub fn build_hc_enter(
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
) -> (usize, usize, usize, usize) {
    let pre = store.push(vec![]);
    let post = store.push(vec![]);
    let comb = store.push(vec![]);
    ops.push(Op::SinkhornSplit { mixes, scale, base, pre, post, comb, m, hc, iters, eps });
    let y = store.push(vec![]);
    ops.push(Op::HyperPre { x, pre, out: y, m, hc, d });
    (pre, post, comb, y)
}

/// Second half of a HyperConnection sublayer: expand the sublayer output `y`
/// back into `hc` residual streams with `post`/`comb`.
pub fn build_hc_exit(
    store: &mut Store,
    ops: &mut Vec<Op>,
    y: usize,
    residual: usize,
    post: usize,
    comb: usize,
    m: usize,
    hc: usize,
    d: usize,
) -> usize {
    let out = store.push(vec![]);
    ops.push(Op::HyperPost { y, residual, post, comb, out, m, hc, d });
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
pub fn build_mla_qkv(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    m: usize,
    shift: u32,
    x: usize,
    rsqrt_table: usize,
    limb_table: usize,
) -> (usize, usize) {
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

/// Convenience wrapper for `build_mla_qkv` with an all-ones `[m, DIM_PAD]`
/// input and locally-built `rsqrt`/limb tables. Used by the standalone
/// attention slice examples; the full block builder calls `build_mla_qkv`
/// directly with the normalized hidden state.
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
    let rsqrt_table = store.push(load_i32("models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    build_mla_qkv(
        store,
        ops,
        file,
        layer,
        m,
        shift,
        x,
        rsqrt_table,
        limb_table,
    )
}

/// Dense multi-head attention over the projected `q` / `kv` tensors, with the
/// per-head attention sink (`attn_sink`) applied as a softmax bias. `kv` is the
/// shared `[m, HEAD_DIM]` latent (`nope || rope`); each head gathers its own
/// `[m, N_HEADS * HEAD_DIM]` query block and RoPE-rotates the interleaved rope
/// tail. The `1/sqrt(head_dim)` attention scale is split across `q` and `kv`
/// (factor `2^16 / head_dim^0.25`, as in the GPT-2 path) before the dot product
/// and folded back to Q16 before the softmax. Returns the concatenated attention
/// output `[m, N_HEADS * HEAD_DIM]`.
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
    let n_pad = (m + 1).next_power_of_two();
    let scale = (65536.0f64 / (HEAD_DIM as f64).powf(0.25)).round() as i64;
    // Causal mask over the `m` real columns plus one unmasked sink column, with
    // the tail padded to `-2^30` so those positions collapse under softmax.
    let mut mask_aug = Vec::with_capacity(m * n_pad);
    for i in 0..m {
        for j in 0..m {
            let lo = i.saturating_sub(SLIDING_WINDOW - 1);
            mask_aug.push(if j > i || j < lo { from_i64(-(1i64 << 30)) } else { Goldilocks::ZERO });
        }
        mask_aug.push(Goldilocks::ZERO);
        for _ in (m + 1)..n_pad {
            mask_aug.push(from_i64(-(1i64 << 30)));
        }
    }
    let mask = store.push(mask_aug);

    // Split the shared latent into its non-rotary and rotary tails. These are
    // column bands of the row-major `[m, HEAD_DIM]` tensor, so they must be
    // gathered per token rather than flat-sliced.
    let kvn = gather(store, ops, kv, col_band(m, HEAD_DIM, 0, nope));
    let kvr = gather(store, ops, kv, col_band(m, HEAD_DIM, nope, nope + qk));

    // Rotate the KV rotary tail once (interleaved pairing). The permuted output
    // is used for the key dot product; a natural-order copy is used as the
    // attention value.
    let kvr_perm = gather(store, ops, kvr, rope_perm_rows(m, qk));
    let kvr_rot = store.push(vec![]);
    ops.push(Op::RoPE {
        x: kvr_perm,
        cos: cos_t,
        sin: sin_t,
        out: kvr_rot,
        range_table: limb_table,
        m,
        d: qk,
        shift,
    });
    let kvr_rot_natural = gather(store, ops, kvr_rot, unrope_perm_rows(m, qk));

    // Value tensors (unscaled Q16): nope padded to a power of two, rope kept at
    // its natural 64 columns (already a power of two).
    let kvn_pad = pad_rows(store, ops, kvn, m, nope, nope_pad);
    // Augment the value tensors with zero rows for the sink and padding slots.
    let zero_rows_nope = store.push(vec![Goldilocks::ZERO; (n_pad - m) * nope_pad]);
    let zero_rows_qk = store.push(vec![Goldilocks::ZERO; (n_pad - m) * qk]);
    let kvn_pad_aug = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvn_pad, zero_rows_nope], out: kvn_pad_aug });
    let kvr_aug = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvr_rot_natural, zero_rows_qk], out: kvr_aug });

    // Key tensors for the score dot products (scaled by `scale`).
    let kvn_scaled = store.push(vec![]);
    ops.push(Op::Scale { x: kvn_pad, out: kvn_scaled, factor: scale, shift });
    let ktn = transpose_gather(store, ops, kvn_scaled, m, nope_pad);
    let kvr_scaled = store.push(vec![]);
    ops.push(Op::Scale { x: kvr_rot, out: kvr_scaled, factor: scale, shift });
    let ktr = transpose_gather(store, ops, kvr_scaled, m, qk);

    // Negated sine table for the output inverse RoPE. The official model
    // conjugates the query rotation (`apply_rotary_emb(..., inverse=True)`),
    // which in fixed point is exactly a RoPE step with `sin` negated.
    let sin_neg = store.push(store.get(sin_t).iter().map(|&v| -v).collect::<Vec<_>>());

    let mut head_outs = Vec::new();
    for head in 0..N_HEADS {
        let (nope_idx, rope_idx) = head_query_indices(head, m);
        let qn = gather(store, ops, q, nope_idx);
        let qr = gather(store, ops, q, rope_idx);
        let qn_pad = pad_rows(store, ops, qn, m, nope, nope_pad);
        let qn_scaled = store.push(vec![]);
        ops.push(Op::Scale { x: qn_pad, out: qn_scaled, factor: scale, shift });
        let qr_perm = gather(store, ops, qr, rope_perm_rows(m, qk));
        let qr_rot = store.push(vec![]);
        ops.push(Op::RoPE {
            x: qr_perm,
            cos: cos_t,
            sin: sin_t,
            out: qr_rot,
            range_table: limb_table,
            m,
            d: qk,
            shift,
        });
        let qr_scaled = store.push(vec![]);
        ops.push(Op::Scale { x: qr_rot, out: qr_scaled, factor: scale, shift });
        let scores_n = store.push(vec![]);
        let scores_r = store.push(vec![]);
        let scores = store.push(vec![]);
        ops.push(Op::MatMul { a: qn_scaled, b: ktn, c: scores_n, m, k: nope_pad, n: m });
        ops.push(Op::MatMul { a: qr_scaled, b: ktr, c: scores_r, m, k: qk, n: m });
        ops.push(Op::Add { a: scores_n, b: scores_r, c: scores });
        let scores_16 = store.push(vec![]);
        ops.push(Op::Scale { x: scores, out: scores_16, factor: 1, shift });
        // Append the per-head attention sink as one extra softmax column. The
        // sink score is a public model constant and enters the max-subtraction
        // and the softmax denominator exactly like a real position.
        let sink_score = to_i64(file.read_f32(&format!("layers.{layer}.attn.attn_sink"))[head]);
        let sink_col = store.push(vec![from_i64(sink_score); m]);
        let scores_aug = pad_sink_matrix(store, ops, scores_16, sink_col, m, m, n_pad);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let idx = build_stable_softmax_index(store, ops, scores_aug, mask, m, n_pad, 1 << 21, 1 << 21, limb_table);
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_table, m, n: n_pad });
        let attn_n = store.push(vec![]);
        let attn_r = store.push(vec![]);
        ops.push(Op::MatMul { a: probs, b: kvn_pad_aug, c: attn_n, m, k: n_pad, n: nope_pad });
        ops.push(Op::MatMul { a: probs, b: kvr_aug, c: attn_r, m, k: n_pad, n: qk });
        let attn_n16 = store.push(vec![]);
        let attn_r16 = store.push(vec![]);
        ops.push(Op::Scale { x: attn_n, out: attn_n16, factor: 1, shift });
        ops.push(Op::Scale { x: attn_r, out: attn_r16, factor: 1, shift });
        let attn_n_sl = gather(store, ops, attn_n16, col_band(m, nope_pad, 0, nope));
        // Un-rotate the attention output rope tail back to natural space, so
        // the concatenated head output matches the official MLA convention.
        let attn_r_perm = gather(store, ops, attn_r16, rope_perm_rows(m, qk));
        let attn_r_unrot = store.push(vec![]);
        ops.push(Op::RoPE {
            x: attn_r_perm,
            cos: cos_t,
            sin: sin_neg,
            out: attn_r_unrot,
            range_table: limb_table,
            m,
            d: qk,
            shift,
        });
        let attn_r_nat = gather(store, ops, attn_r_unrot, unrope_perm_rows(m, qk));
        let head_out = interleave2(store, ops, attn_n_sl, attn_r_nat, m, nope, qk);
        head_outs.push(head_out);
    }

    // Token-major interleave of all heads into `o` [m, N_HEADS * HEAD_DIM].
    stack_rows(store, ops, &head_outs, m, HEAD_DIM)
}

/// MLA attention over a sliding window plus (optionally) a shared compressed-KV
/// bank, with a per-head attention sink. For prefill with `m <= SLIDING_WINDOW`
/// (seq16), the window covers every position, so this reduces to a dense
/// attention over the concatenation `[window_kv; compress_kv]` with a custom
/// causal mask (window positions visible up to `i`, compressed positions up to
/// `i / ratio`), plus the sink column. This is the exact official semantics for
/// seq16; the true sliding-window gather only differs once `m > SLIDING_WINDOW`.
///
/// `window_kv` is the pre-RoPE `[m, HEAD_DIM]` latent; `compress_latent` is the
/// pre-RoPE `[m / ratio, HEAD_DIM]` compressed latent (empty slice if `ratio == 0`).
/// Returns the concatenated attention output `[m, N_HEADS * HEAD_DIM]`.
#[allow(clippy::too_many_arguments)]
pub fn build_mla_attention_compress(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    m: usize,
    ratio: usize,
    shift: u32,
    q: usize,
    window_kv: usize,
    compress_latent: usize,
    cos_t: usize,
    sin_t: usize,
    compress_cos: usize,
    compress_sin: usize,
    exp_table: usize,
    limb_table: usize,
) -> usize {
    let nope = HEAD_NOPE;
    let qk = HEAD_QK;
    let nope_pad = HEAD_NOPE_PAD;
    let groups = if ratio > 0 { m / ratio } else { 0 };
    let n_kv = m + groups;
    // The score matmul needs a power-of-two position count, so the KV bank is
    // padded up to `n_kv_pad`, and the softmax length is `n_kv_pad + 1` (sink)
    // padded up again.
    let n_kv_pad = n_kv.next_power_of_two();
    let n_pad = (n_kv_pad + 1).next_power_of_two();
    let scale = (65536.0f64 / (HEAD_DIM as f64).powf(0.25)).round() as i64;

    // Window nope/rope, with the rope tail rotated (permuted layout for scoring).
    let kw_n = gather(store, ops, window_kv, col_band(m, HEAD_DIM, 0, nope));
    let kw_r = gather(store, ops, window_kv, col_band(m, HEAD_DIM, nope, nope + qk));
    let kw_r_perm = gather(store, ops, kw_r, rope_perm_rows(m, qk));
    let kw_r_rot = store.push(vec![]);
    ops.push(Op::RoPE {
        x: kw_r_perm,
        cos: cos_t,
        sin: sin_t,
        out: kw_r_rot,
        range_table: limb_table,
        m,
        d: qk,
        shift,
    });
    let kw_r_nat = gather(store, ops, kw_r_rot, unrope_perm_rows(m, qk));

    // Concatenate window and compressed tensors uniformly (window-only path
    // uses the window tensors directly).
    let (kvn_cat, kvr_score_cat, kvr_val_cat) = if groups == 0 {
        (kw_n, kw_r_rot, kw_r_nat)
    } else {
        let kc_n = gather(store, ops, compress_latent, col_band(groups, HEAD_DIM, 0, nope));
        let kc_r = gather(store, ops, compress_latent, col_band(groups, HEAD_DIM, nope, nope + qk));
        let kc_r_perm = gather(store, ops, kc_r, rope_perm_rows(groups, qk));
        let kc_r_rot = store.push(vec![]);
        ops.push(Op::RoPE {
            x: kc_r_perm,
            cos: compress_cos,
            sin: compress_sin,
            out: kc_r_rot,
            range_table: limb_table,
            m: groups,
            d: qk,
            shift,
        });
        let kc_r_nat = gather(store, ops, kc_r_rot, unrope_perm_rows(groups, qk));
        let n = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![kw_n, kc_n], out: n });
        let r_score = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![kw_r_rot, kc_r_rot], out: r_score });
        let r_val = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![kw_r_nat, kc_r_nat], out: r_val });
        (n, r_score, r_val)
    };

    let kvn_pad = pad_rows(store, ops, kvn_cat, n_kv, nope, nope_pad);
    // Pad the KV position count up to a power of two for the score matmul.
    let zero_kvn = store.push(vec![Goldilocks::ZERO; (n_kv_pad - n_kv) * nope_pad]);
    let kvn_full = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvn_pad, zero_kvn], out: kvn_full });
    let zero_kvr = store.push(vec![Goldilocks::ZERO; (n_kv_pad - n_kv) * qk]);
    let kvr_score_full = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvr_score_cat, zero_kvr.clone()], out: kvr_score_full });
    let kvr_val_full = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvr_val_cat, zero_kvr], out: kvr_val_full });

    // Augment values with zero rows for the sink and softmax padding slots.
    let zero_rows_nope = store.push(vec![Goldilocks::ZERO; (n_pad - n_kv_pad) * nope_pad]);
    let zero_rows_qk = store.push(vec![Goldilocks::ZERO; (n_pad - n_kv_pad) * qk]);
    let kvn_aug = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvn_full, zero_rows_nope], out: kvn_aug });
    let kvr_aug = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvr_val_full, zero_rows_qk], out: kvr_aug });

    // Keys for scoring.
    let kvn_scaled = store.push(vec![]);
    ops.push(Op::Scale { x: kvn_full, out: kvn_scaled, factor: scale, shift });
    let ktn = transpose_gather(store, ops, kvn_scaled, n_kv_pad, nope_pad);
    let kvr_scaled = store.push(vec![]);
    ops.push(Op::Scale { x: kvr_score_full, out: kvr_scaled, factor: scale, shift });
    let ktr = transpose_gather(store, ops, kvr_scaled, n_kv_pad, qk);

    // Custom causal mask: window columns 0..m (<= i), compressed columns
    // m..n_kv (<= i/ratio), KV padding columns, the sink column (always), and
    // -2^30 softmax padding.
    let mut mask_aug = Vec::with_capacity(m * n_pad);
    for i in 0..m {
        for j in 0..m {
            let lo = i.saturating_sub(SLIDING_WINDOW - 1);
            mask_aug.push(if j <= i && j >= lo { Goldilocks::ZERO } else { from_i64(-(1i64 << 30)) });
        }
        for j in 0..groups {
            mask_aug.push(if j <= i / ratio { Goldilocks::ZERO } else { from_i64(-(1i64 << 30)) });
        }
        for _ in n_kv..n_kv_pad {
            mask_aug.push(from_i64(-(1i64 << 30)));
        }
        mask_aug.push(Goldilocks::ZERO);
        for _ in (n_kv_pad + 1)..n_pad {
            mask_aug.push(from_i64(-(1i64 << 30)));
        }
    }
    let mask = store.push(mask_aug);

    let sin_neg = store.push(store.get(sin_t).iter().map(|&v| -v).collect::<Vec<_>>());

    let mut head_outs = Vec::new();
    for head in 0..N_HEADS {
        let (nope_idx, rope_idx) = head_query_indices(head, m);
        let qn = gather(store, ops, q, nope_idx);
        let qr = gather(store, ops, q, rope_idx);
        let qn_pad = pad_rows(store, ops, qn, m, nope, nope_pad);
        let qn_scaled = store.push(vec![]);
        ops.push(Op::Scale { x: qn_pad, out: qn_scaled, factor: scale, shift });
        let qr_perm = gather(store, ops, qr, rope_perm_rows(m, qk));
        let qr_rot = store.push(vec![]);
        ops.push(Op::RoPE {
            x: qr_perm,
            cos: cos_t,
            sin: sin_t,
            out: qr_rot,
            range_table: limb_table,
            m,
            d: qk,
            shift,
        });
        let qr_scaled = store.push(vec![]);
        ops.push(Op::Scale { x: qr_rot, out: qr_scaled, factor: scale, shift });
        let scores_n = store.push(vec![]);
        let scores_r = store.push(vec![]);
        let scores = store.push(vec![]);
        ops.push(Op::MatMul { a: qn_scaled, b: ktn, c: scores_n, m, k: nope_pad, n: n_kv_pad });
        ops.push(Op::MatMul { a: qr_scaled, b: ktr, c: scores_r, m, k: qk, n: n_kv_pad });
        ops.push(Op::Add { a: scores_n, b: scores_r, c: scores });
        let scores_16 = store.push(vec![]);
        ops.push(Op::Scale { x: scores, out: scores_16, factor: 1, shift });
        let sink_score = to_i64(file.read_f32(&format!("layers.{layer}.attn.attn_sink"))[head]);
        let sink_col = store.push(vec![from_i64(sink_score); m]);
        let scores_aug = pad_sink_matrix(store, ops, scores_16, sink_col, m, n_kv_pad, n_pad);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let idx = build_stable_softmax_index(store, ops, scores_aug, mask, m, n_pad, 1 << 21, 1 << 21, limb_table);
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_table, m, n: n_pad });
        let attn_n = store.push(vec![]);
        let attn_r = store.push(vec![]);
        ops.push(Op::MatMul { a: probs, b: kvn_aug, c: attn_n, m, k: n_pad, n: nope_pad });
        ops.push(Op::MatMul { a: probs, b: kvr_aug, c: attn_r, m, k: n_pad, n: qk });
        let attn_n16 = store.push(vec![]);
        let attn_r16 = store.push(vec![]);
        ops.push(Op::Scale { x: attn_n, out: attn_n16, factor: 1, shift });
        ops.push(Op::Scale { x: attn_r, out: attn_r16, factor: 1, shift });
        let attn_n_sl = gather(store, ops, attn_n16, col_band(m, nope_pad, 0, nope));
        let attn_r_perm = gather(store, ops, attn_r16, rope_perm_rows(m, qk));
        let attn_r_unrot = store.push(vec![]);
        ops.push(Op::RoPE {
            x: attn_r_perm,
            cos: cos_t,
            sin: sin_neg,
            out: attn_r_unrot,
            range_table: limb_table,
            m,
            d: qk,
            shift,
        });
        let attn_r_nat = gather(store, ops, attn_r_unrot, unrope_perm_rows(m, qk));
        let head_out = interleave2(store, ops, attn_n_sl, attn_r_nat, m, nope, qk);
        head_outs.push(head_out);
    }

    stack_rows(store, ops, &head_outs, m, HEAD_DIM)
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
        let o_g = gather(store, ops, o, col_band(m, N_HEADS * HEAD_DIM, g * per_group, (g + 1) * per_group));
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

/// Compressor: pool `ratio` consecutive tokens into one compressed KV latent
/// with a learned softmax gate, then RMSNorm over `HEAD_DIM`. `ratio == 1` is a
/// plain projection (no gate). Matches the official `Compressor.forward` for
/// prefill (`start_pos == 0`, `seqlen % ratio == 0`). `x` is `[m, DIM_PAD]`.
///
/// Returns the pre-RoPE latent `[m / ratio, HEAD_DIM]` (row-major, unpadded).
pub fn build_compressor(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    x: usize,
    m: usize,
    ratio: usize,
    shift: u32,
    rsqrt_table: usize,
    limb_table: usize,
    exp_table: usize,
) -> usize {
    let hd = HEAD_DIM;
    let groups = m / ratio;
    let wkv_t = push_bf16_transposed(
        store,
        file.clone(),
        &format!("layers.{layer}.attn.compressor.wkv.weight"),
        hd,
        DIM,
    );
    let kv_raw = store.push(vec![]);
    ops.push(Op::MatMul { a: x, b: wkv_t, c: kv_raw, m, k: DIM_PAD, n: hd });
    let kv = store.push(vec![]);
    ops.push(Op::Scale { x: kv_raw, out: kv, factor: 1, shift });

    let norm_w = {
        let v: Vec<i64> = file
            .read_bf16(&format!("layers.{layer}.attn.compressor.norm.weight"))
            .iter()
            .map(|&z| to_i64(z))
            .collect();
        store.push(broadcast_norm(&v, groups, hd))
    };

    if ratio == 1 {
        return build_rms_norm(store, ops, kv, norm_w, rsqrt_table, limb_table, m, hd, hd);
    }

    let wgate_t = push_bf16_transposed(
        store,
        file.clone(),
        &format!("layers.{layer}.attn.compressor.wgate.weight"),
        hd,
        DIM,
    );
    let score_raw = store.push(vec![]);
    ops.push(Op::MatMul { a: x, b: wgate_t, c: score_raw, m, k: DIM_PAD, n: hd });
    let score = store.push(vec![]);
    ops.push(Op::Scale { x: score_raw, out: score, factor: 1, shift });

    // Gather score and kv into `[groups*hd, ratio]`: row `(g, d)` holds the
    // `ratio` in-group tokens' values for that (group, dim).
    let mut idx = Vec::with_capacity(groups * hd * ratio);
    for g in 0..groups {
        for d in 0..hd {
            for t in 0..ratio {
                idx.push(((g * ratio + t) * hd + d) as u32);
            }
        }
    }
    let score2 = gather(store, ops, score, idx.clone());
    let kv2 = gather(store, ops, kv, idx);

    // Softmax over the in-group axis (no causal mask).
    let zero_mask = store.push(vec![Goldilocks::ZERO; groups * hd * ratio]);
    let sidx = build_stable_softmax_index(
        store, ops, score2, zero_mask, groups * hd, ratio, 1 << 21, 1 << 21, limb_table,
    );
    let e = store.push(vec![]);
    let probs = store.push(vec![]);
    ops.push(Op::Softmax { idx: sidx, e, out: probs, table: exp_table, m: groups * hd, n: ratio });

    // Pooled sum over the in-group axis. `ratio` is a power of two here, so
    // each column can be gathered and combined with elementwise scale + add.
    let mut pooled = store.push(vec![Goldilocks::ZERO; groups * hd]);
    for t in 0..ratio {
        let mut col = Vec::with_capacity(groups * hd);
        for i in 0..groups * hd {
            col.push((i * ratio + t) as u32);
        }
        let pt = gather(store, ops, probs, col.clone());
        let kt = gather(store, ops, kv2, col);
        let term = store.push(vec![]);
        ops.push(Op::ScaleVec { x: kt, scale: pt, out: term, shift: 16 });
        if t == 0 {
            pooled = term;
        } else {
            let s = store.push(vec![]);
            ops.push(Op::Add { a: pooled, b: term, c: s });
            pooled = s;
        }
    }

    build_rms_norm(store, ops, pooled, norm_w, rsqrt_table, limb_table, groups, hd, hd)
}

/// Engram sub-block: n-gram hash lookup + FP8 embedding + wkv projection +
/// gated write into the Hyper-Connection residual stream. `x4` is
/// `[m, HC_MULT, DIM_PAD]`; `hash_ids` holds the `m * ENGRAM_N_HASH` row ids for
/// this layer (precomputed by `extract_v41_engram.py`). Returns the updated
/// `[m, HC_MULT, DIM_PAD]` residual.
#[allow(clippy::too_many_arguments)]
pub fn build_engram(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    x4: usize,
    m: usize,
    hash_ids: &[u32],
    shift: u32,
    rsqrt_table: usize,
    limb_table: usize,
    gate_table: usize,
) -> usize {
    let hc = HC_MULT;
    let flat = ENGRAM_N_HASH * ENGRAM_HEAD_DIM;
    let flat_pad = flat.next_power_of_two();
    let out_dim = DIM * (hc + 1);
    let out_pad = out_dim.next_power_of_two();
    let hc_d = hc * DIM;
    let base = format!("layers.{layer}.engram");

    // Embedding row lookup (dequantize only the requested rows).
    let e = store.push(file.read_fp8_rows(
        &format!("{base}.embed.weight"),
        &format!("{base}.embed.scale"),
        32,
        hash_ids,
    ));
    let e_pad = pad_rows(store, ops, e, m, flat, flat_pad);

    // wkv projection.
    let wkv = push_quantized_fp8(
        store,
        file.clone(),
        &format!("{base}.wkv.weight"),
        &format!("{base}.wkv.scale"),
    );
    let kv_raw = build_quantized_matmul(store, ops, e_pad, wkv, m, flat_pad, out_pad);
    let kv = store.push(vec![]);
    ops.push(Op::Scale { x: kv_raw, out: kv, factor: 1, shift });
    let key_real = gather(store, ops, kv, col_band(m, out_pad, 0, hc_d));
    let value_real = gather(store, ops, kv, col_band(m, out_pad, hc_d, hc_d + DIM));

    // The residual stream, collapsed to its real columns.
    let h_real = flatten_hc_real(store, ops, x4, m, hc, DIM, DIM_PAD);
    // Pad the per-copy 5120-wide tensors to the field-friendly 8192. Padding
    // is zero, so the dot-product sums and the RMSNorm `n_real` are unchanged.
    let h = pad_rows(store, ops, h_real, m * hc, DIM, DIM_PAD);
    let key = pad_rows(store, ops, key_real, m * hc, DIM, DIM_PAD);
    let value = pad_rows(store, ops, value_real, m, DIM, DIM_PAD);

    // weight = q_weight * k_weight (elementwise, Q16), broadcast per token.
    // q_weight/k_weight are constants, so the product is materialized at build
    // time rather than as a runtime op.
    let qw: Vec<i64> = file
        .read_bf16(&format!("{base}.q_weight"))
        .iter()
        .map(|&x| to_i64(x))
        .collect();
    let kw: Vec<i64> = file
        .read_bf16(&format!("{base}.k_weight"))
        .iter()
        .map(|&x| to_i64(x))
        .collect();
    let weight: Vec<Goldilocks> = qw
        .iter()
        .zip(kw.iter())
        .map(|(a, b)| from_i64(round_div_i64(a * b, 1 << 16)))
        .collect();
    let mut weight_m = Vec::with_capacity(m * hc * DIM_PAD);
    for _ in 0..m {
        for h in 0..hc {
            weight_m.extend_from_slice(&weight[h * DIM..(h + 1) * DIM]);
            weight_m.extend(std::iter::repeat(Goldilocks::ZERO).take(DIM_PAD - DIM));
        }
    }
    let weight_m = store.push(weight_m);

    let hw = store.push(vec![]);
    ops.push(Op::ScaleVec { x: h, scale: weight_m, out: hw, shift: 16 });
    let hwk = store.push(vec![]);
    ops.push(Op::ScaleVec { x: hw, scale: key, out: hwk, shift: 16 });

    // dot = (h*w*key).sum(-1) * rstd_h * rstd_key / sqrt(d), per (token, copy).
    let ones2 = {
        let mut v = vec![Goldilocks::ZERO; DIM_PAD * 2];
        for j in 0..DIM_PAD {
            v[j * 2] = from_i64(1 << 16);
        }
        store.push(v)
    };
    let sum_raw = store.push(vec![]);
    ops.push(Op::MatMul { a: hwk, b: ones2, c: sum_raw, m: m * hc, k: DIM_PAD, n: 2 });
    let sum_col = gather(store, ops, sum_raw, col_band(m * hc, 2, 0, 1));
    let dot_sum = store.push(vec![]);
    ops.push(Op::Scale { x: sum_col, out: dot_sum, factor: 1, shift });
    let rstd_h = store.push(vec![]);
    ops.push(Op::Rsqrt { x: h, out: rstd_h, rsqrt_table, m: m * hc, d: DIM_PAD, n_real: DIM });
    let rstd_k = store.push(vec![]);
    ops.push(Op::Rsqrt { x: key, out: rstd_k, rsqrt_table, m: m * hc, d: DIM_PAD, n_real: DIM });
    let s1 = store.push(vec![]);
    ops.push(Op::ScaleVec { x: dot_sum, scale: rstd_h, out: s1, shift: 16 });
    let s2 = store.push(vec![]);
    ops.push(Op::ScaleVec { x: s1, scale: rstd_k, out: s2, shift: 16 });
    let dim_sqrt = (65536.0 / (DIM as f64).sqrt()).round() as i64;
    let dot = store.push(vec![]);
    ops.push(Op::Scale { x: s2, out: dot, factor: dim_sqrt, shift: 16 });

    // gate = sigmoid(sign(dot) * sqrt(|dot|)) via one precomputed table.
    let gate_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: dot, out: gate_idx, offset: 1 << 23, table_len: 1 << 24, limb_table });
    let gate = store.push(vec![]);
    ops.push(Op::Lookup { idx: gate_idx, out: gate, table: gate_table });

    // out = h + gate * value.
    let value4 = expand_to_hc(store, ops, value, m, hc, DIM_PAD);
    let gate4 = {
        let mut idx = Vec::with_capacity(m * hc * DIM_PAD);
        for i in 0..m {
            for h in 0..hc {
                for _ in 0..DIM_PAD {
                    idx.push((i * hc + h) as u32);
                }
            }
        }
        gather(store, ops, gate, idx)
    };
    let term = store.push(vec![]);
    ops.push(Op::ScaleVec { x: value4, scale: gate4, out: term, shift: 16 });
    let out_real = store.push(vec![]);
    ops.push(Op::Add { a: h, b: term, c: out_real });

    // `out_real` is already `[m, hc, DIM_PAD]` row-major.
    out_real
}

/// One full Hyper-Connection block for an arbitrary layer, with sliding-window
/// and/or compressed-KV MLA attention. `compress_kv` is the shared pre-RoPE
/// compressed latent published by the most recent kv-source layer (or the one
/// this layer publishes itself). Returns `(x4_out, ffn_pre, new_compress_kv)`;
/// `new_compress_kv` is `Some` when `is_kv_source(layer)`.
#[allow(clippy::too_many_arguments)]
pub fn build_v41_layer(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    m: usize,
    ratio: usize,
    routed: bool,
    shift: u32,
    x4: usize,
    pre_mix: usize,
    compress_kv: Option<usize>,
    cos_t: usize,
    sin_t: usize,
    compress_cos: usize,
    compress_sin: usize,
    exp_table: usize,
    rsqrt_table: usize,
    limb_table: usize,
    sqrt_softplus_table: usize,
    silu_table: usize,
    clip_table: usize,
) -> (usize, usize, Option<usize>) {
    let hc = HC_MULT;
    let d_stored = DIM_PAD;
    let hc_d = hc * DIM;
    let d_pad = hc_d.next_power_of_two();
    let mix_hc = (2 + hc) * hc;
    let mix_hc_pad = mix_hc.next_power_of_two();
    let iters = 20usize;
    let eps = HC_EPS_Q16;

    // ---- attention sublayer ----
    let attn_fn_t = push_f32_transposed(
        store,
        file.clone(),
        &format!("layers.{layer}.hc_attn_fn"),
        mix_hc,
        hc_d,
    );
    let attn_base = push_f32_vec(store, file.clone(), &format!("layers.{layer}.hc_attn_base"));
    let attn_scale = push_f32_vec(store, file.clone(), &format!("layers.{layer}.hc_attn_scale"));
    let x4_flat = flatten_hc_real(store, ops, x4, m, hc, DIM, d_stored);
    let x4_pad = pad_rows(store, ops, x4_flat, m, hc_d, d_pad);
    let attn_mix = build_hc_mixes(
        store,
        ops,
        x4_pad,
        attn_fn_t,
        m,
        hc_d,
        d_pad,
        mix_hc_pad,
        rsqrt_table,
        limb_table,
    );
    let (attn_pre, attn_post, attn_comb) =
        build_hc_split(store, ops, attn_mix, attn_scale, attn_base, m, hc, iters, eps);
    let attn_in = build_hc_pre(store, ops, x4, pre_mix, m, hc, d_stored);
    let attn_norm_w = {
        let v: Vec<i64> = file
            .read_bf16(&format!("layers.{layer}.attn_norm.weight"))
            .iter()
            .map(|&x| to_i64(x))
            .collect();
        store.push(broadcast_norm(&v, m, DIM_PAD))
    };
    let attn_in_norm =
        build_rms_norm(store, ops, attn_in, attn_norm_w, rsqrt_table, limb_table, m, DIM_PAD, DIM);

    let (q, kvn) = build_mla_qkv(
        store,
        ops,
        file.clone(),
        layer,
        m,
        shift,
        attn_in_norm,
        rsqrt_table,
        limb_table,
    );

    // Publish this layer's compressed latent if it is a kv source.
    let new_compress_kv = if is_kv_source(layer) {
        Some(build_compressor(
            store,
            ops,
            file.clone(),
            layer,
            attn_in_norm,
            m,
            ratio,
            shift,
            rsqrt_table,
            limb_table,
            exp_table,
        ))
    } else {
        None
    };
    let ckv = new_compress_kv.or(compress_kv);

    let o = if ratio > 0 {
        build_mla_attention_compress(
            store,
            ops,
            file.clone(),
            layer,
            m,
            ratio,
            shift,
            q,
            kvn,
            ckv.expect("compressed KV must be published before a compress layer"),
            cos_t,
            sin_t,
            compress_cos,
            compress_sin,
            exp_table,
            limb_table,
        )
    } else {
        build_mla_attention(
            store,
            ops,
            file.clone(),
            layer,
            m,
            shift,
            q,
            kvn,
            cos_t,
            sin_t,
            exp_table,
            limb_table,
        )
    };
    let attn_out = build_oproj(store, ops, file.clone(), layer, m, shift, o);
    let x4_attn = build_hc_post(store, ops, attn_out, x4, attn_post, attn_comb, m, hc, d_stored);

    // ---- FFN sublayer ----
    let ffn_fn_t = push_f32_transposed(
        store,
        file.clone(),
        &format!("layers.{layer}.hc_ffn_fn"),
        mix_hc,
        hc_d,
    );
    let ffn_base = push_f32_vec(store, file.clone(), &format!("layers.{layer}.hc_ffn_base"));
    let ffn_scale = push_f32_vec(store, file.clone(), &format!("layers.{layer}.hc_ffn_scale"));
    let x4_flat2 = flatten_hc_real(store, ops, x4_attn, m, hc, DIM, d_stored);
    let x4_pad2 = pad_rows(store, ops, x4_flat2, m, hc_d, d_pad);
    let ffn_mix = build_hc_mixes(
        store,
        ops,
        x4_pad2,
        ffn_fn_t,
        m,
        hc_d,
        d_pad,
        mix_hc_pad,
        rsqrt_table,
        limb_table,
    );
    let (ffn_pre, ffn_post, ffn_comb) =
        build_hc_split(store, ops, ffn_mix, ffn_scale, ffn_base, m, hc, iters, eps);
    let ffn_in = build_hc_pre(store, ops, x4_attn, attn_pre, m, hc, d_stored);
    let ffn_norm_w = {
        let v: Vec<i64> = file
            .read_bf16(&format!("layers.{layer}.ffn_norm.weight"))
            .iter()
            .map(|&x| to_i64(x))
            .collect();
        store.push(broadcast_norm(&v, m, DIM_PAD))
    };
    let ffn_in_norm =
        build_rms_norm(store, ops, ffn_in, ffn_norm_w, rsqrt_table, limb_table, m, DIM_PAD, DIM);

    let shared_out = build_shared_expert(
        store,
        ops,
        file.clone(),
        layer,
        ffn_in_norm,
        m,
        shift,
        silu_table,
        clip_table,
        limb_table,
    );
    let ffn_out = if routed {
        let (sel, _thr, weights, _d1, _d2) = build_moe_gate(
            store,
            ops,
            file.clone(),
            layer,
            ffn_in_norm,
            m,
            shift,
            sqrt_softplus_table,
            limb_table,
        );
        let routed_out = build_routed_experts(
            store,
            ops,
            file.clone(),
            layer,
            ffn_in_norm,
            sel,
            weights,
            m,
            shift,
            silu_table,
            clip_table,
            limb_table,
        );
        let out = store.push(vec![]);
        ops.push(Op::Add { a: shared_out, b: routed_out, c: out });
        out
    } else {
        shared_out
    };
    let x4_out = build_hc_post(store, ops, ffn_out, x4_attn, ffn_post, ffn_comb, m, hc, d_stored);

    (x4_out, ffn_pre, new_compress_kv)
}

/// Run all 40 backbone layers, returning the final `(x4, pre_mix)` before the
/// output RMSNorm + head. `x` is the token embedding `[m, DIM_PAD]`.
#[allow(clippy::too_many_arguments)]
pub fn build_v41_forward(
    store: &mut Store,
    ops: &mut Vec<Op>,
    dir: &str,
    m: usize,
    shift: u32,
    routed: bool,
    n_layers: usize,
    x: usize,
    exp_table: usize,
    rsqrt_table: usize,
    limb_table: usize,
    sqrt_softplus_table: usize,
    silu_table: usize,
    clip_table: usize,
    gate_table: usize,
    hash_ids: &[u32],
) -> (usize, usize) {
    let hc = HC_MULT;
    let x4 = expand_to_hc(store, ops, x, m, hc, DIM_PAD);
    let mut pre_mix = build_identity_pre_mix(store, m);
    let mut x4 = x4;
    let mut compress_kv: Option<usize> = None;

    for layer in 0..n_layers.min(N_LAYERS) {
        // Engram writes into the HC residual before the block.
        if layer == 1 || layer == 14 {
            let eng_shard = open_shard(&format!(
                "{dir}/model-{:05}-of-00048.safetensors",
                if layer == 1 { 47 } else { 48 }
            ));
            let col0 = if layer == 1 { 0 } else { ENGRAM_N_HASH };
            let mut layer_hash = Vec::with_capacity(m * ENGRAM_N_HASH);
            for i in 0..m {
                for j in 0..ENGRAM_N_HASH {
                    layer_hash.push(hash_ids[i * (2 * ENGRAM_N_HASH) + col0 + j]);
                }
            }
            x4 = build_engram(
                store,
                ops,
                eng_shard,
                layer,
                x4,
                m,
                &layer_hash,
                shift,
                rsqrt_table,
                limb_table,
                gate_table,
            );
        }
        let ratio = compress_ratio(layer);
        let file = open_shard(&layer_shard(dir, layer));
        let (cos, sin) = build_rope_tables(store, m, HEAD_QK);
        let groups = if ratio > 0 { m / ratio } else { 0 };
        let cpos: Vec<usize> = (0..groups).map(|g| g * ratio).collect();
        let (ccos, csin) = build_rope_tables_at(store, &cpos, HEAD_QK);
        let (x4o, pre_mixo, new_ckv) = build_v41_layer(
            store,
            ops,
            file,
            layer,
            m,
            ratio,
            routed,
            shift,
            x4,
            pre_mix,
            compress_kv,
            cos,
            sin,
            ccos,
            csin,
            exp_table,
            rsqrt_table,
            limb_table,
            sqrt_softplus_table,
            silu_table,
            clip_table,
        );
        x4 = x4o;
        pre_mix = pre_mixo;
        if new_ckv.is_some() {
            compress_kv = new_ckv;
        }
    }
    (x4, pre_mix)
}

/// Final output head: collapse the HC stream, RMSNorm, project to logits over
/// the vocabulary, and return `(logits, argmax_idx)`. `logits` is `[m, VOCAB_PAD]`
/// (Q16); `argmax_idx` is the per-row argmax over the first `VOCAB` entries.
pub fn build_final_logits(
    store: &mut Store,
    ops: &mut Vec<Op>,
    dir: &str,
    m: usize,
    shift: u32,
    x4: usize,
    pre_mix: usize,
    rsqrt_table: usize,
    limb_table: usize,
) -> (usize, usize) {
    let file = open_shard(&format!("{dir}/model-00043-of-00048.safetensors"));
    let h = build_hc_pre(store, ops, x4, pre_mix, m, HC_MULT, DIM_PAD);
    let norm_w = {
        let v: Vec<i64> = file.read_bf16("norm.weight").iter().map(|&x| to_i64(x)).collect();
        store.push(broadcast_norm(&v, m, DIM_PAD))
    };
    let h_norm = build_rms_norm(store, ops, h, norm_w, rsqrt_table, limb_table, m, DIM_PAD, DIM);
    let head_w = push_bf16_transposed(store, file, "head.weight", VOCAB, DIM);
    let logits_raw = store.push(vec![]);
    ops.push(Op::MatMul { a: h_norm, b: head_w, c: logits_raw, m, k: DIM_PAD, n: VOCAB_PAD });
    let logits = store.push(vec![]);
    ops.push(Op::Scale { x: logits_raw, out: logits, factor: 1, shift });
    let argmax = store.push_idx(vec![]);
    let sel = store.push(vec![]);
    let thr = store.push(vec![]);
    let gate = store.push(vec![]);
    let d1 = store.push(vec![]);
    let d2 = store.push(vec![]);
    ops.push(Op::Argmax { x: logits, sel, thr, gate, d1, d2, out: argmax, m, n: VOCAB_PAD });
    (logits, argmax)
}

/// One-hot initial Hyper-Connection collapse: copy 0 carries the full residual,
/// matching the official `make_identity_pre_mix`.
pub fn build_identity_pre_mix(store: &mut Store, m: usize) -> usize {
    let mut v = vec![Goldilocks::ZERO; m * HC_MULT];
    for i in 0..m {
        v[i * HC_MULT] = from_i64(1 << 16);
    }
    store.push(v)
}

/// Load a precomputed prompt embedding `[m, DIM]` (i32 Q16) and row-pad it to
/// `[m, DIM_PAD]`.
pub fn push_embedding(store: &mut Store, path: &str, m: usize) -> usize {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let vals: Vec<i64> = bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
        .collect();
    assert!(vals.len() >= m * DIM, "embedding too short: {} < {}", vals.len(), m * DIM);
    let mut out = vec![Goldilocks::ZERO; m * DIM_PAD];
    for i in 0..m {
        for j in 0..DIM {
            out[i * DIM_PAD + j] = from_i64(vals[i * DIM + j]);
        }
    }
    store.push(out)
}

/// Split an HC mix projection into `pre`/`post`/`comb` coefficients.
fn build_hc_split(
    store: &mut Store,
    ops: &mut Vec<Op>,
    mix: usize,
    scale: usize,
    base: usize,
    m: usize,
    hc: usize,
    iters: usize,
    eps: i64,
) -> (usize, usize, usize) {
    let pre = store.push(vec![]);
    let post = store.push(vec![]);
    let comb = store.push(vec![]);
    ops.push(Op::SinkhornSplit { mixes: mix, scale, base, pre, post, comb, m, hc, iters, eps });
    (pre, post, comb)
}

/// Collapse the `hc` residual copies with the incoming `pre` coefficients.
fn build_hc_pre(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x4: usize,
    pre_in: usize,
    m: usize,
    hc: usize,
    d: usize,
) -> usize {
    // `y[i,j] = round(sum_h pre[i,h] * x4[i,h,j] / 2^16)`. This is a per-token
    // batched matmul (`pre` as the per-token weight), proven by `BatchedMatMul`
    // rather than the unproven `HyperPre`.
    let y_raw = store.push(vec![]);
    ops.push(Op::BatchedMatMul { a: pre_in, w: x4, c: y_raw, s: m, k: hc, n: d });
    let y = store.push(vec![]);
    ops.push(Op::Scale { x: y_raw, out: y, factor: 1, shift: 16 });
    y
}

/// Expand the sublayer output back into `hc` residual copies: `out[i,h,j] =
/// round((y[i,j] * post[i,h] + sum_g comb[i,h,g] * residual[i,g,j]) / 2^16)`.
/// Assembled from proven primitives (BatchedMatMul + Lookup + ScaleVec + Add +
/// Scale) instead of the unproven `HyperPost`.
fn build_hc_post(
    store: &mut Store,
    ops: &mut Vec<Op>,
    y: usize,
    residual: usize,
    post: usize,
    comb: usize,
    m: usize,
    hc: usize,
    d: usize,
) -> usize {
    // term1[i,h,j] = y[i,j] * post[i,h] (raw Q32, no rounding).
    let idx_y = {
        let mut v = Vec::with_capacity(m * hc * d);
        for i in 0..m {
            for h in 0..hc {
                for j in 0..d {
                    v.push((i * d + j) as u32);
                }
            }
        }
        v
    };
    let y_b = gather(store, ops, y, idx_y);
    let idx_post = {
        let mut v = Vec::with_capacity(m * hc * d);
        for i in 0..m {
            for h in 0..hc {
                for _ in 0..d {
                    v.push((i * hc + h) as u32);
                }
            }
        }
        v
    };
    let post_b = gather(store, ops, post, idx_post);
    let term1 = store.push(vec![]);
    ops.push(Op::ScaleVec { x: y_b, scale: post_b, out: term1, shift: 0 });

    // term2[i,h,j] = sum_g comb[i,h,g] * residual[i,g,j].
    let idx_res = {
        let mut v = Vec::with_capacity(m * hc * hc * d);
        for i in 0..m {
            for h in 0..hc {
                for g in 0..hc {
                    for j in 0..d {
                        v.push((i * hc * d + g * d + j) as u32);
                    }
                }
            }
        }
        v
    };
    let res_rep = gather(store, ops, residual, idx_res);
    let term2 = store.push(vec![]);
    ops.push(Op::BatchedMatMul { a: comb, w: res_rep, c: term2, s: m * hc, k: hc, n: d });

    let sum = store.push(vec![]);
    ops.push(Op::Add { a: term1, b: term2, c: sum });
    let out = store.push(vec![]);
    ops.push(Op::Scale { x: sum, out, factor: 1, shift: 16 });
    out
}

/// Numerically-stable softmax index: `idx[i,j] = clamp(masked[i,j] - row_max[i]
/// + offset, 0, table_len-1)`, where `masked = x + mask` and `row_max` is the
/// per-row max. Assembled from proven primitives (`Add` + `TopKSelect(k=1)` +
/// `Lookup` broadcast + `Scale` + `GeluIndex`) instead of the unproven
/// `StableSoftmaxIndex`.
fn build_stable_softmax_index(
    store: &mut Store,
    ops: &mut Vec<Op>,
    x: usize,
    mask: usize,
    m: usize,
    n: usize,
    offset: i64,
    table_len: usize,
    limb_table: usize,
) -> usize {
    let masked = store.push(vec![]);
    ops.push(Op::Add { a: x, b: mask, c: masked });

    // row_max = max over the n columns (TopKSelect with k=1 returns the max as
    // its threshold).
    let sel = store.push(vec![]);
    let thr = store.push(vec![]);
    let gate = store.push(vec![]);
    let d1 = store.push(vec![]);
    let d2 = store.push(vec![]);
    ops.push(Op::TopKSelect { x: masked, sel, thr, gate, d1, d2, m, n, k: 1 });

    // Broadcast the per-row max back to [m, n] and subtract.
    let thr_b = gather(
        store,
        ops,
        thr,
        (0..m * n).map(|ij| (ij / n) as u32).collect::<Vec<_>>(),
    );
    let neg_thr = store.push(vec![]);
    ops.push(Op::Scale { x: thr_b, out: neg_thr, factor: -1, shift: 0 });
    let shifted = store.push(vec![]);
    ops.push(Op::Add { a: masked, b: neg_thr, c: shifted });

    let idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: shifted, out: idx, offset, table_len, limb_table });
    idx
}

/// Full Hyper-Connection block for one layer. `x4` is the `[m, HC_MULT, DIM_PAD]`
/// residual and `pre_mix` the `[m, HC_MULT]` collapse coming from the previous
/// layer. The attention sublayer collapses with the incoming `pre_mix`, the FFN
/// sublayer collapses with this layer's attention `pre`, and the returned `ffn_pre`
/// is handed to the next layer. Attention is the validated dense causal MLA path
/// (correct for `compress_ratio == 0`, which covers layers 0, 1, 38, 39 and any
/// `seq <= sliding_window`).
#[allow(clippy::too_many_arguments)]
pub fn build_block(
    store: &mut Store,
    ops: &mut Vec<Op>,
    file: Arc<SafetensorFile>,
    layer: usize,
    m: usize,
    shift: u32,
    routed: bool,
    x4: usize,
    pre_mix: usize,
    rsqrt_table: usize,
    limb_table: usize,
    exp_table: usize,
    sqrt_softplus_table: usize,
    silu_table: usize,
    clip_table: usize,
) -> (usize, usize, usize, usize) {
    let hc = HC_MULT;
    let d_stored = DIM_PAD;
    let hc_d = hc * DIM; // 20480 real flattened columns
    let d_pad = hc_d.next_power_of_two(); // 32768
    let mix_hc = (2 + hc) * hc; // 24
    let mix_hc_pad = mix_hc.next_power_of_two(); // 32
    let iters = 20usize;
    let eps = HC_EPS_Q16;

    // ---- attention sublayer: mixes -> collapse(incoming pre) -> norm -> MLA ----
    let attn_fn_t = push_f32_transposed(
        store,
        file.clone(),
        &format!("layers.{layer}.hc_attn_fn"),
        mix_hc,
        hc_d,
    );
    let attn_base = push_f32_vec(store, file.clone(), &format!("layers.{layer}.hc_attn_base"));
    let attn_scale = push_f32_vec(store, file.clone(), &format!("layers.{layer}.hc_attn_scale"));
    let x4_flat = flatten_hc_real(store, ops, x4, m, hc, DIM, d_stored);
    let x4_pad = pad_rows(store, ops, x4_flat, m, hc_d, d_pad);
    let attn_mix = build_hc_mixes(
        store,
        ops,
        x4_pad,
        attn_fn_t,
        m,
        hc_d,
        d_pad,
        mix_hc_pad,
        rsqrt_table,
        limb_table,
    );
    let (attn_pre, attn_post, attn_comb) =
        build_hc_split(store, ops, attn_mix, attn_scale, attn_base, m, hc, iters, eps);
    let attn_in = build_hc_pre(store, ops, x4, pre_mix, m, hc, d_stored);

    let attn_norm_w = {
        let v: Vec<i64> = file
            .read_bf16(&format!("layers.{layer}.attn_norm.weight"))
            .iter()
            .map(|&x| to_i64(x))
            .collect();
        store.push(broadcast_norm(&v, m, DIM_PAD))
    };
    let attn_in_norm =
        build_rms_norm(store, ops, attn_in, attn_norm_w, rsqrt_table, limb_table, m, DIM_PAD, DIM);

    let (cos_t, sin_t) = build_rope_tables(store, m, HEAD_QK);
    let (q, kv) = build_mla_qkv(
        store,
        ops,
        file.clone(),
        layer,
        m,
        shift,
        attn_in_norm,
        rsqrt_table,
        limb_table,
    );
    let o = build_mla_attention(
        store,
        ops,
        file.clone(),
        layer,
        m,
        shift,
        q,
        kv,
        cos_t,
        sin_t,
        exp_table,
        limb_table,
    );
    let attn_out = build_oproj(store, ops, file.clone(), layer, m, shift, o);
    let x4_attn = build_hc_post(store, ops, attn_out, x4, attn_post, attn_comb, m, hc, d_stored);

    // ---- FFN sublayer: mixes -> collapse(attention pre) -> norm -> MoE ----
    let ffn_fn_t = push_f32_transposed(
        store,
        file.clone(),
        &format!("layers.{layer}.hc_ffn_fn"),
        mix_hc,
        hc_d,
    );
    let ffn_base = push_f32_vec(store, file.clone(), &format!("layers.{layer}.hc_ffn_base"));
    let ffn_scale = push_f32_vec(store, file.clone(), &format!("layers.{layer}.hc_ffn_scale"));
    let x4_flat2 = flatten_hc_real(store, ops, x4_attn, m, hc, DIM, d_stored);
    let x4_pad2 = pad_rows(store, ops, x4_flat2, m, hc_d, d_pad);
    let ffn_mix = build_hc_mixes(
        store,
        ops,
        x4_pad2,
        ffn_fn_t,
        m,
        hc_d,
        d_pad,
        mix_hc_pad,
        rsqrt_table,
        limb_table,
    );
    let (ffn_pre, ffn_post, ffn_comb) =
        build_hc_split(store, ops, ffn_mix, ffn_scale, ffn_base, m, hc, iters, eps);
    let ffn_in = build_hc_pre(store, ops, x4_attn, attn_pre, m, hc, d_stored);

    let ffn_norm_w = {
        let v: Vec<i64> = file
            .read_bf16(&format!("layers.{layer}.ffn_norm.weight"))
            .iter()
            .map(|&x| to_i64(x))
            .collect();
        store.push(broadcast_norm(&v, m, DIM_PAD))
    };
    let ffn_in_norm =
        build_rms_norm(store, ops, ffn_in, ffn_norm_w, rsqrt_table, limb_table, m, DIM_PAD, DIM);

    let shared_out = build_shared_expert(
        store,
        ops,
        file.clone(),
        layer,
        ffn_in_norm,
        m,
        shift,
        silu_table,
        clip_table,
        limb_table,
    );
    let ffn_out = if routed {
        let (sel, _thr, weights, _d1, _d2) =
            build_moe_gate(
                store, ops, file.clone(), layer, ffn_in_norm, m, shift, sqrt_softplus_table, limb_table,
            );
        let routed_out = build_routed_experts(
            store,
            ops,
            file.clone(),
            layer,
            ffn_in_norm,
            sel,
            weights,
            m,
            shift,
            silu_table,
            clip_table,
            limb_table,
        );
        let out = store.push(vec![]);
        ops.push(Op::Add { a: shared_out, b: routed_out, c: out });
        out
    } else {
        shared_out
    };
    let x4_out = build_hc_post(store, ops, ffn_out, x4_attn, ffn_post, ffn_comb, m, hc, d_stored);

    (x4_out, ffn_pre, x4_attn, ffn_out)
}
