#!/usr/bin/env python3
"""Float64 reference for the N-layer DeepSeek-V4.1-Flash text forward.

Validates the Rust `build_v41_forward` against the official `inference/model.py`
semantics: Hyper-Connection residual stream, MLA attention (dense window +
shared compressed KV + sink), and MoE (shared expert, optionally routed).

Usage:
  python3 scripts/v41_ref_forward.py --raw /data/zkIE/models/deepseek-v4.1-flash/raw \
      --layers 3 --m 16 --no-routed
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v41_ref_cpu import (  # noqa: E402
    bf16_vec,
    dequant_fp4_weight,
    dequant_fp8_weight,
    read_safetensors,
    rms_norm,
    rope,
    sparse_attn,
)


DIM = 5120
HC = 4
MIX_HC = (2 + HC) * HC
HC_D = HC * DIM
N_HEADS = 64
HEAD_DIM = 512
QK = 64
NOPE = HEAD_DIM - QK
Q_LORA = 1280
O_LORA = 1024
O_GROUPS = 8
INTER = 2304
N_EXPERTS = 384
ROUTE_SCALE = 1.5


def fixed_float(w):
    s = w * 65536.0
    return np.where(s >= 0, np.floor(s + 0.5), np.ceil(s - 0.5)).astype(np.float64) / 65536.0


def hc_split_sinkhorn(mixes, scale, base, iters=20, eps=1e-6):
    pre = 1.0 / (1.0 + np.exp(-(mixes[:, :HC] * scale[0] + base[:HC]))) + eps
    post = 2.0 / (1.0 + np.exp(-(mixes[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC])))
    c = mixes[:, 2 * HC :].reshape(-1, HC, HC) * scale[2] + base[2 * HC :].reshape(HC, HC)
    c = np.exp(c - c.max(-1, keepdims=True))
    c = c / c.sum(-1, keepdims=True) + eps
    c = c / (c.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdims=True) + eps)
        c = c / (c.sum(-2, keepdims=True) + eps)
    return pre, post, c


def hc_mixes(x4, hc_fn, hc_scale, hc_base, eps=1e-20, hc_eps=1e-6):
    xf = x4.reshape(x4.shape[0], HC_D).astype(np.float64)
    rsqrt = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    mixes = (xf @ hc_fn.T) * rsqrt
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, eps=hc_eps)


def hc_pre(x4, pre):
    return np.sum(pre[:, :, None] * x4, axis=1)


def hc_post(y, residual, post, comb):
    return post[:, :, None] * y[:, None, :] + np.einsum("mhg,mgd->mhd", comb, residual)


def rope_at(x, positions, qk):
    inv = np.arange(qk // 2, dtype=np.float64)
    freqs = np.asarray(positions, dtype=np.float64)[:, None] * np.power(10000.0, -2.0 * inv / qk)
    x[..., -qk:] = rope(x[..., -qk:], freqs[None, :, None, :])
    return x


def build_attention(t, layer, x, m, ratio, compress_kv):
    q_norm = bf16_vec(t[f"layers.{layer}.attn.q_norm.weight"])
    kv_norm = bf16_vec(t[f"layers.{layer}.attn.kv_norm.weight"])
    wq_a = fixed_float(dequant_fp8_weight(t[f"layers.{layer}.attn.wq_a.weight"], t[f"layers.{layer}.attn.wq_a.scale"]))
    wq_b = fixed_float(dequant_fp8_weight(t[f"layers.{layer}.attn.wq_b.weight"], t[f"layers.{layer}.attn.wq_b.scale"]))
    wkv = fixed_float(dequant_fp8_weight(t[f"layers.{layer}.attn.wkv.weight"], t[f"layers.{layer}.attn.wkv.scale"]))
    wo_a = fixed_float(dequant_fp8_weight(t[f"layers.{layer}.attn.wo_a.weight"], t[f"layers.{layer}.attn.wo_a.scale"]))
    wo_b = fixed_float(dequant_fp8_weight(t[f"layers.{layer}.attn.wo_b.weight"], t[f"layers.{layer}.attn.wo_b.scale"]))
    sink = np.frombuffer(t[f"layers.{layer}.attn.attn_sink"]["data"], dtype="<f4").astype(np.float64)

    qr = rms_norm(x @ wq_a.T, q_norm)
    q = (qr @ wq_b.T).reshape(1, m, N_HEADS, HEAD_DIM)
    kv = rms_norm(x @ wkv.T, kv_norm)
    positions = np.arange(m, dtype=np.float64)
    q[..., -QK:] = rope(q[..., -QK:], (positions[:, None] * np.power(10000.0, -2.0 * np.arange(QK // 2) / QK))[None, :, None, :])
    kv[..., -QK:] = rope(kv[..., -QK:], (positions[:, None] * np.power(10000.0, -2.0 * np.arange(QK // 2) / QK))[None, :, :])

    if ratio > 0:
        latent = compress_kv.copy()
        cpos = np.arange(m // ratio, dtype=np.float64) * ratio
        latent[..., -QK:] = rope(latent[..., -QK:], (cpos[:, None] * np.power(10000.0, -2.0 * np.arange(QK // 2) / QK))[None, :, :])
        kv_all = np.concatenate([kv[None], latent[None]], axis=1)
        groups = m // ratio
        end = np.arange(m)[:, None]
        win = (end - 127).clip(0) + np.arange(m)[None, :]
        win = np.where(win > end, -1, win).astype(np.int64)
        comp = np.arange(groups)[None, :] + m
        comp = np.where(comp - m <= end // ratio, comp, -1).astype(np.int64)
        topk = np.concatenate([win, comp], axis=1)[None]
    else:
        kv_all = kv[None]
        end = np.arange(m)[:, None]
        win = (end - 127).clip(0) + np.arange(m)[None, :]
        win = np.where(win > end, -1, win).astype(np.int64)
        topk = win[None]

    o = sparse_attn(q, kv_all, sink, topk, HEAD_DIM ** -0.5)
    o[..., -QK:] = rope(o[..., -QK:], (positions[:, None] * np.power(10000.0, -2.0 * np.arange(QK // 2) / QK))[None, :, None, :], inverse=True)
    o = o[0].reshape(m, O_GROUPS, -1)
    wa = wo_a.reshape(O_GROUPS, O_LORA, -1)
    o = np.einsum("sgd,grd->sgr", o, wa)
    return o.reshape(m, -1) @ wo_b.T


def build_compressor(t, layer, x, m, ratio):
    hd = HEAD_DIM
    wkv = bf16_vec(t[f"layers.{layer}.attn.compressor.wkv.weight"]).reshape(hd, DIM)
    norm = bf16_vec(t[f"layers.{layer}.attn.compressor.norm.weight"])
    kv = x @ wkv.T
    if ratio == 1:
        return rms_norm(kv, norm)
    wgate = bf16_vec(t[f"layers.{layer}.attn.compressor.wgate.weight"]).reshape(hd, DIM)
    score = x @ wgate.T
    groups = m // ratio
    kv = kv.reshape(groups, ratio, hd)
    score = score.reshape(groups, ratio, hd)
    score = score - score.max(axis=1, keepdims=True)
    probs = np.exp(score)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return rms_norm((kv * probs).sum(axis=1), norm)


def build_shared_expert(t, layer, x):
    w1 = dequant_fp8_weight(t[f"layers.{layer}.ffn.shared_experts.w1.weight"], t[f"layers.{layer}.ffn.shared_experts.w1.scale"])
    w2 = dequant_fp8_weight(t[f"layers.{layer}.ffn.shared_experts.w2.weight"], t[f"layers.{layer}.ffn.shared_experts.w2.scale"])
    w3 = dequant_fp8_weight(t[f"layers.{layer}.ffn.shared_experts.w3.weight"], t[f"layers.{layer}.ffn.shared_experts.w3.scale"])
    g = np.clip(x @ w1.T, None, 10.0)
    u = np.clip(x @ w3.T, -10.0, 10.0)
    return ((g / (1.0 + np.exp(-g))) * u) @ w2.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--no-routed", action="store_true")
    args = ap.parse_args()

    m = args.m
    shards = {l: read_safetensors(f"{args.raw}/model-{l+3:05d}-of-00048.safetensors") for l in range(args.layers)}

    x = np.ones((m, DIM), dtype=np.float64)
    x4 = np.repeat(x[:, None, :], HC, axis=1)
    pre_mix = np.zeros((m, HC), dtype=np.float64)
    pre_mix[:, 0] = 1.0
    compress_kv = None

    for layer in range(args.layers):
        t = shards[layer]
        ratio = 0 if layer < 2 or layer >= 38 else (2 if layer <= 20 else 1)
        hc_attn_fn = np.frombuffer(t[f"layers.{layer}.hc_attn_fn"]["data"], dtype="<f4").astype(np.float64).reshape(MIX_HC, HC_D)
        hc_attn_base = np.frombuffer(t[f"layers.{layer}.hc_attn_base"]["data"], dtype="<f4").astype(np.float64)
        hc_attn_scale = np.frombuffer(t[f"layers.{layer}.hc_attn_scale"]["data"], dtype="<f4").astype(np.float64)
        attn_norm = bf16_vec(t[f"layers.{layer}.attn_norm.weight"])
        attn_pre, attn_post, attn_comb = hc_mixes(x4, hc_attn_fn, hc_attn_scale, hc_attn_base)
        attn_in = rms_norm(hc_pre(x4, pre_mix), attn_norm)

        if layer in (2, 8, 14, 20):
            compress_kv = build_compressor(t, layer, attn_in, m, ratio)
        attn_out = build_attention(t, layer, attn_in, m, ratio, compress_kv)
        x4 = hc_post(attn_out, x4, attn_post, attn_comb)

        hc_ffn_fn = np.frombuffer(t[f"layers.{layer}.hc_ffn_fn"]["data"], dtype="<f4").astype(np.float64).reshape(MIX_HC, HC_D)
        hc_ffn_base = np.frombuffer(t[f"layers.{layer}.hc_ffn_base"]["data"], dtype="<f4").astype(np.float64)
        hc_ffn_scale = np.frombuffer(t[f"layers.{layer}.hc_ffn_scale"]["data"], dtype="<f4").astype(np.float64)
        ffn_norm = bf16_vec(t[f"layers.{layer}.ffn_norm.weight"])
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x4, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
        ffn_in = rms_norm(hc_pre(x4, attn_pre), ffn_norm)
        ffn_out = build_shared_expert(t, layer, ffn_in)
        x4 = hc_post(ffn_out, x4, ffn_post, ffn_comb)
        pre_mix = ffn_pre

    fixed = np.round(x4 * 65536.0).astype(np.int64)
    print("first8", fixed[0, 0, :8].tolist())
    print("last8", fixed[-1, 0, :8].tolist())


if __name__ == "__main__":
    main()
