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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--n", type=int, default=8192)
    args = ap.parse_args()

    t = read_safetensors(args.shard)
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
