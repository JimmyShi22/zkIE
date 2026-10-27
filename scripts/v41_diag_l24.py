#!/usr/bin/env python3
"""Recompute layer 24 from Rust-dumped layer-23 state in float64 and diff the
result against Rust's layer-24 state. Bisects attn vs ffn and prints gate top-6
so a routing flip can be spotted."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/data/zkIE/scripts")
from v41_ref_cpu import (  # noqa: E402
    bf16_vec,
    dequant_fp4_weight,
    dequant_fp8_weight,
    read_safetensors,
    rms_norm,
    rope,
    sparse_attn,
)
from v41_ref_forward import (  # noqa: E402
    DIM,
    HC,
    HEAD_DIM,
    INTER,
    N_EXPERTS,
    QK,
    ROUTE_SCALE,
    build_attention,
    fixed_float,
    hc_mixes,
    hc_post,
    hc_pre,
)


def sqrt_softplus(x):
    return np.sqrt(np.log1p(np.exp(x)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def gate_and_routed(t, layer, x, m):
    gw = bf16_vec(t[f"layers.{layer}.ffn.gate.weight"]).reshape(N_EXPERTS, DIM)
    gb = np.frombuffer(t[f"layers.{layer}.ffn.gate.bias"]["data"], dtype="<f4").astype(np.float64)
    z = x @ gw.T
    s = sqrt_softplus(z)
    topk = 6
    sel = np.zeros((m, N_EXPERTS), dtype=np.float64)
    chosen = []
    for i in range(m):
        biased = s[i] + gb
        idx = np.argpartition(-biased, topk)[:topk]
        idx = idx[np.argsort(-biased[idx])]
        sel[i, idx] = 1.0
        chosen.append(idx.tolist())
    s_sel = s * sel
    weights = s_sel / s_sel.sum(-1, keepdims=True) * ROUTE_SCALE
    out = np.zeros((m, DIM), dtype=np.float64)
    for e in range(N_EXPERTS):
        tokens = np.where(sel[:, e] == 1.0)[0]
        if len(tokens) == 0:
            continue
        w1 = fixed_float(dequant_fp4_weight(t[f"layers.{layer}.ffn.experts.{e}.w1.weight"], t[f"layers.{layer}.ffn.experts.{e}.w1.scale"], DIM))
        w2 = fixed_float(dequant_fp4_weight(t[f"layers.{layer}.ffn.experts.{e}.w2.weight"], t[f"layers.{layer}.ffn.experts.{e}.w2.scale"], INTER))
        w3 = fixed_float(dequant_fp4_weight(t[f"layers.{layer}.ffn.experts.{e}.w3.weight"], t[f"layers.{layer}.ffn.experts.{e}.w3.scale"], DIM))
        g = silu(x[tokens] @ w1.T)
        u = np.clip(x[tokens] @ w3.T, -10.0, 10.0)
        out[tokens] += weights[tokens, e, None] * (g * u @ w2.T)
    return z, s, np.asarray(chosen), out


def shared_expert(t, layer, x):
    w1 = fixed_float(dequant_fp8_weight(t[f"layers.{layer}.ffn.shared_experts.w1.weight"], t[f"layers.{layer}.ffn.shared_experts.w1.scale"]))
    w2 = fixed_float(dequant_fp8_weight(t[f"layers.{layer}.ffn.shared_experts.w2.weight"], t[f"layers.{layer}.ffn.shared_experts.w2.scale"]))
    w3 = fixed_float(dequant_fp8_weight(t[f"layers.{layer}.ffn.shared_experts.w3.weight"], t[f"layers.{layer}.ffn.shared_experts.w3.scale"]))
    g = silu(x @ w1.T)
    u = np.clip(x @ w3.T, -10.0, 10.0)
    return (g * u) @ w2.T


def q16(v):
    s = v * 65536.0
    return np.where(s >= 0, np.floor(s + 0.5), np.ceil(s - 0.5)).astype(np.int64)


def load_i64(path, shape):
    a = np.fromfile(path, dtype="<i8").reshape(shape)
    return a.astype(np.float64) / 65536.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/data/zkIE/models/deepseek-v4.1-flash/raw")
    ap.add_argument("--dump-dir", default="/tmp")
    ap.add_argument("--m", type=int, default=16)
    args = ap.parse_args()
    m = args.m
    dp = args.dump_dir

    x4 = load_i64(f"{dp}/x4_L23.bin", (m, HC, 8192))[:, :, :DIM]
    pre_mix = load_i64(f"{dp}/pre_mix_L23.bin", (m, HC))
    ckv = load_i64(f"{dp}/compress_kv_L23.bin", (m, HEAD_DIM))
    rust_x4_24 = load_i64(f"{dp}/x4_L24.bin", (m, HC, 8192))[:, :, :DIM]

    t = read_safetensors(f"{args.raw}/model-{24+3:05d}-of-00048.safetensors")

    hc_attn_fn = np.frombuffer(t["layers.24.hc_attn_fn"]["data"], dtype="<f4").astype(np.float64).reshape((2 + HC) * HC, HC * DIM)
    hc_attn_base = np.frombuffer(t["layers.24.hc_attn_base"]["data"], dtype="<f4").astype(np.float64)
    hc_attn_scale = np.frombuffer(t["layers.24.hc_attn_scale"]["data"], dtype="<f4").astype(np.float64)
    attn_norm = bf16_vec(t["layers.24.attn_norm.weight"])
    attn_pre, attn_post, attn_comb = hc_mixes(x4, hc_attn_fn, hc_attn_scale, hc_attn_base)
    attn_in = rms_norm(hc_pre(x4, pre_mix), attn_norm)
    attn_out = build_attention(t, 24, attn_in, m, 1, ckv)
    x4_attn = hc_post(attn_out, x4, attn_post, attn_comb)

    hc_ffn_fn = np.frombuffer(t["layers.24.hc_ffn_fn"]["data"], dtype="<f4").astype(np.float64).reshape((2 + HC) * HC, HC * DIM)
    hc_ffn_base = np.frombuffer(t["layers.24.hc_ffn_base"]["data"], dtype="<f4").astype(np.float64)
    hc_ffn_scale = np.frombuffer(t["layers.24.hc_ffn_scale"]["data"], dtype="<f4").astype(np.float64)
    ffn_norm = bf16_vec(t["layers.24.ffn_norm.weight"])
    ffn_pre, ffn_post, ffn_comb = hc_mixes(x4_attn, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
    ffn_in = rms_norm(hc_pre(x4_attn, attn_pre), ffn_norm)
    z, s, chosen, routed = gate_and_routed(t, 24, ffn_in, m)
    shared = shared_expert(t, 24, ffn_in)
    ffn_out = shared + routed
    x4_24 = hc_post(ffn_out, x4_attn, ffn_post, ffn_comb)

    # Component-level diffs (float64 predictions vs Rust Q16 values).
    print("chosen top6 (token0):", chosen[0])
    print("z[0][:8]:", np.round(z[0, :8] * 65536).astype(np.int64).tolist())
    print("attn_out[0][:8]:", q16(attn_out[0, :8]).tolist())
    print("ffn_in[0][:8]:", q16(ffn_in[0, :8]).tolist())
    print("ffn_out[0][:8]:", q16(ffn_out[0, :8]).tolist())
    pred = q16(x4_24)
    rust = q16(rust_x4_24)
    diff = np.abs(pred - rust)
    print("x4_24 max abs diff:", diff.max(), "at", np.unravel_index(diff.argmax(), diff.shape))
    print("x4_24 mean abs diff:", diff.mean())
    print("pred  x4_24[0,0,:8]:", pred[0, 0, :8].tolist())
    print("rust  x4_24[0,0,:8]:", rust[0, 0, :8].tolist())
    # How different is the input already? (prediction uses Rust's own L23 state,
    # so this only reflects layer-24 numerical fidelity, not accumulated drift.)


if __name__ == "__main__":
    main()
