#!/usr/bin/env python3
"""Float64 reference for the DeepSeek-V4.1-Flash Engram sub-block.

Reads the precomputed hash ids (from extract_v41_engram.py) and computes
`Engram.forward` for one layer and an all-ones HC residual, so the Rust
`build_engram` can be diffed against it.

Usage:
  python3 scripts/v41_ref_engram.py \
    --raw /data/zkIE/models/deepseek-v4.1-flash/raw \
    --hash-ids /data/zkIE/models/deepseek-v4.1-flash/cache/engram/hash_ids_i32.bin \
    --layer 1 --m 16
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v41_ref_cpu import bf16_vec, dequant_fp8_weight, read_safetensors  # noqa: E402


DIM = 5120
HC = 4
N_HASH_COLS = 24
HEAD_DIM = 256
FLAT = N_HASH_COLS * HEAD_DIM
OUT = DIM * (HC + 1)
EPS = 1e-20


def dequant_embed_rows(t, name, scale_name, rows):
    # FP8 E4M3 rows with E8M0 per-block scales (block=32 over the 256 cols).
    w = np.frombuffer(t[name]["data"], dtype=np.uint8)
    s = np.frombuffer(t[scale_name]["data"], dtype=np.uint8)
    n_rows, cols = t[name]["shape"]
    nb_cols = t[scale_name]["shape"][1]
    out = []
    for r in rows:
        r = int(r)
        row = np.zeros(cols, dtype=np.float64)
        for c in range(cols):
            code = int(w[r * cols + c])
            sign = -1.0 if code & 0x80 else 1.0
            exp = (code >> 3) & 0x0F
            man = code & 0x07
            val = man * (2.0 ** -9) if exp == 0 else (1.0 + man / 8.0) * (2.0 ** (exp - 7))
            scale = 2.0 ** (int(s[r * nb_cols + c // 32]) - 127)
            row[c] = sign * val * scale
        out.append(row)
    return np.asarray(out, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--hash-ids", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--m", type=int, default=16)
    args = ap.parse_args()

    shard = f"{args.raw}/model-{47 if args.layer == 1 else 48:05d}-of-00048.safetensors"
    t = read_safetensors(shard)
    hash_ids = np.fromfile(args.hash_ids, dtype=np.int32).reshape(args.m, -1)
    # Per-layer column block: layer 1 uses cols [0:24], layer 14 uses [24:48].
    col0 = 0 if args.layer == 1 else N_HASH_COLS
    ids = hash_ids[:, col0 : col0 + N_HASH_COLS].reshape(-1).astype(np.int64)

    base = f"layers.{args.layer}.engram"
    e = dequant_embed_rows(t, f"{base}.embed.weight", f"{base}.embed.scale", ids)
    # ids are per (token, hash-col); e is [m*24, 256], already in row-major.
    flat = e.reshape(args.m, FLAT)
    wkv = dequant_fp8_weight(t[f"{base}.wkv.weight"], t[f"{base}.wkv.scale"])
    kv = flat @ wkv.T  # [m, 25600]
    key = kv[:, : DIM * HC].reshape(args.m, HC, DIM)
    value = kv[:, DIM * HC : DIM * (HC + 1)]
    qw = bf16_vec(t[f"{base}.q_weight"]).reshape(HC, DIM)
    kw = bf16_vec(t[f"{base}.k_weight"]).reshape(HC, DIM)
    weight = qw * kw

    h = np.ones((args.m, HC, DIM), dtype=np.float64)
    rstd_h = 1.0 / np.sqrt(np.mean(h * h, axis=-1) + EPS)
    rstd_key = 1.0 / np.sqrt(np.mean(key * key, axis=-1) + EPS)
    dot = (h * weight * key).sum(-1) * rstd_h * rstd_key * (DIM ** -0.5)
    gate = 1.0 / (1.0 + np.exp(-np.copysign(np.sqrt(np.abs(dot).clip(1e-6)), dot)))
    out = h + gate[..., None] * value[:, None, :]
    fixed = np.round(out * 65536.0).astype(np.int64)
    print("engram_first8", fixed[0, 0, :8].tolist())
    print("engram_last8", fixed[-1, 0, :8].tolist())


if __name__ == "__main__":
    main()
