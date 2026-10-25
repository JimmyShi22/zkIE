#!/usr/bin/env python3
"""CPU reference for DeepSeek-V4.1-Flash quantized ops.

The full 40-layer CSA2/CED/Engram forward is still being wired into the Rust
builder. This script provides the reference primitive used by the first vertical
slice: dequantize an FP8-E4M3/E8M0 safetensors weight and compute
`x @ W^T` for an all-ones input, so it can be diffed against the Rust
`matmul_ref` example.

Usage:
  python3 scripts/v41_ref_cpu.py \
    --shard /data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors \
    --weight layers.0.attn.wo_a.weight \
    --scale layers.0.attn.wo_a.scale \
    --m 1 --k 4096 --n 8192
"""

import argparse
import json
import struct
import math

import numpy as np


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        tensors = {}
        for name, meta in header.items():
            start, end = meta["data_offsets"]
            f.seek(base + start)
            tensors[name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data": f.read(end - start),
            }
        return tensors


def dequant_fp8_weight(weight, scale, block=32):
    w = np.frombuffer(weight["data"], dtype=np.uint8).reshape(weight["shape"])
    s = np.frombuffer(scale["data"], dtype=np.uint8).reshape(scale["shape"])
    s = np.power(2.0, s.astype(np.float64) - 127.0)
    s_big = np.repeat(np.repeat(s, block, axis=0), block, axis=1)
    sign = np.where((w & 0x80) != 0, -1.0, 1.0)
    exp = (w >> 3) & 0x0F
    man = (w & 0x07).astype(np.float64)
    val = np.where(
        exp == 0,
        man * (2.0 ** -9),
        (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0),
    )
    return sign * val * s_big


def bf16_vec(weight):
    bits = np.frombuffer(weight["data"], dtype="<u2").astype(np.uint32)
    return (bits << 16).astype(np.uint32).view(np.float32).astype(np.float64)


def rms_norm(x, w, eps=1e-20):
    rsqrt = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rsqrt * w


def rope(x, freqs):
    half = x.shape[-1] // 2
    xr = x.reshape(*x.shape[:-1], half, 2)
    cos, sin = np.cos(freqs), np.sin(freqs)
    out = np.empty_like(xr)
    out[..., 0] = xr[..., 0] * cos - xr[..., 1] * sin
    out[..., 1] = xr[..., 0] * sin + xr[..., 1] * cos
    return out.reshape(x.shape)


def sparse_attn(q, kv, sink, topk_idxs, scale):
    b, s, h, d = q.shape
    o = np.zeros_like(q)
    for bi in range(b):
        for si in range(s):
            idxs = topk_idxs[bi, si]
            idxs = idxs[idxs >= 0]
            if len(idxs) == 0:
                continue
            kv_sel = kv[bi, idxs, :]
            scores = q[bi, si] @ kv_sel.T * scale
            scores = np.concatenate([scores, sink[:, None]], axis=1)
            scores = scores - np.max(scores, axis=1, keepdims=True)
            probs = np.exp(scores)
            denom = probs.sum(axis=1, keepdims=True)
            o[bi, si] = (probs[:, : len(idxs)] @ kv_sel) / denom
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--mode", choices=["matmul", "attention"], default="matmul")
    args = ap.parse_args()

    t = read_safetensors(args.shard)

    if args.mode == "attention":
        m = 16
        dim = 5120
        n_heads = 64
        head_dim = 512
        q_lora = 1280
        qk_rope = 64
        o_lora = 1024
        o_groups = 8
        sink = np.frombuffer(t["layers.0.attn.attn_sink"]["data"], dtype="<f4").astype(np.float64)
        q_norm = bf16_vec(t["layers.0.attn.q_norm.weight"])
        kv_norm = bf16_vec(t["layers.0.attn.kv_norm.weight"])
        wq_a = dequant_fp8_weight(t["layers.0.attn.wq_a.weight"], t["layers.0.attn.wq_a.scale"])
        wq_b = dequant_fp8_weight(t["layers.0.attn.wq_b.weight"], t["layers.0.attn.wq_b.scale"])
        wkv = dequant_fp8_weight(t["layers.0.attn.wkv.weight"], t["layers.0.attn.wkv.scale"])
        wo_a = dequant_fp8_weight(t["layers.0.attn.wo_a.weight"], t["layers.0.attn.wo_a.scale"])
        wo_b = dequant_fp8_weight(t["layers.0.attn.wo_b.weight"], t["layers.0.attn.wo_b.scale"])

        x = np.ones((1, m, dim), dtype=np.float64)
        qr = rms_norm(x @ wq_a.T, q_norm)
        q = (qr @ wq_b.T).reshape(1, m, n_heads, head_dim)
        kv = rms_norm(x @ wkv.T, kv_norm)

        pos = np.arange(m, dtype=np.float64)
        inv = np.arange(qk_rope // 2, dtype=np.float64)
        freqs = pos[:, None] * np.power(10000.0, -2.0 * inv / qk_rope)
        q[..., -qk_rope:] = rope(q[..., -qk_rope:], freqs[None, :, None, :])
        kv[..., -qk_rope:] = rope(kv[..., -qk_rope:], freqs[None, :, :])

        end = np.arange(m)[:, None]
        topk = (end - 127).clip(0) + np.arange(m)[None, :]
        topk = np.where(topk > end, -1, topk).astype(np.int64)[None, :, :]
        o = sparse_attn(q, kv, sink, topk, head_dim ** -0.5)

        o = o.reshape(1, m, o_groups, -1)
        wo_a = wo_a.reshape(o_groups, o_lora, -1)
        o = np.einsum("bsgd,grd->bsgr", o, wo_a)
        out = o.reshape(1, m, -1) @ wo_b.T
        fixed = np.round(out[0] * 65536.0).astype(np.int64)
        for i in range(8):
            print(f"[{i}] {int(fixed[0, i])}")
        return

    winfo = t[args.weight]
    sinfo = t[args.scale]
    w = np.frombuffer(winfo["data"], dtype=np.uint8).reshape(winfo["shape"])
    s = np.frombuffer(sinfo["data"], dtype=np.uint8).reshape(sinfo["shape"])

    block = 32
    s = np.power(2.0, s.astype(np.float64) - 127.0)
    s_big = np.repeat(np.repeat(s, block, axis=0), block, axis=1)
    sign = np.where((w & 0x80) != 0, -1.0, 1.0)
    exp = (w >> 3) & 0x0F
    man = (w & 0x07).astype(np.float64)
    val = np.where(
        exp == 0,
        man * (2.0 ** -9),
        (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0),
    )
    wdec = sign * val * s_big

    x = np.ones((args.m, args.k), dtype=np.float64)
    # Match the Rust prover's fixed-point path: each weight is rounded to i64
    # before the dot product, and the dot product is an exact integer sum.
    scaled = wdec * 65536.0
    wfixed = np.where(
        scaled >= 0.0,
        np.floor(scaled + 0.5),
        np.ceil(scaled - 0.5),
    ).astype(np.int64)
    fixed = x.astype(np.int64) @ wfixed.T
    for i in range(min(8, fixed.shape[1])):
        print(f"[{i}] {int(fixed[0, i])}")


if __name__ == "__main__":
    main()
