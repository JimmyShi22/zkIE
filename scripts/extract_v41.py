#!/usr/bin/env python3
"""Extract DeepSeek-V4.1-Flash weights into little-endian i32 fixed-point bins.

Only the Python standard library is used. The safetensors header is parsed
manually, so this works on a machine without torch/safetensors/numpy.

Quantized weights are dequantized with the same tables and rounding used by
`crates/zkie-core/src/common/quant.rs` / `quantized_io.rs`: values are scaled by
2**16 and rounded to the nearest integer before being written as i32.

Usage:
  python3 scripts/extract_v41.py \
    --raw /data/zkIE/models/deepseek-v4.1-flash/raw \
    --out /data/zkIE/models/deepseek-v4.1-flash/cache \
    --layers 0
"""

import argparse
import json
import math
import os
import struct
from pathlib import Path

import numpy as np


FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
FIXED = 65536.0


def e8m0(code):
    return 2.0 ** (code - 127)


def e4m3(code):
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 3) & 0x0F
    man = code & 0x07
    if exp == 0:
        return sign * man * (2.0 ** -9)
    return sign * (1.0 + man / 8.0) * (2.0 ** (exp - 7))


def e2m1(code):
    return FP4_TABLE[code & 0x0F]


def to_i32(v):
    v = int(round(v * FIXED))
    return max(-(2**31), min(2**31 - 1, v))


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        tensors = {}
        for name, meta in header.items():
            start, end = meta["data_offsets"]
            dtype = meta["dtype"]
            shape = meta["shape"]
            f.seek(base + start)
            data = f.read(end - start)
            tensors[name] = (dtype, shape, data)
        return tensors


def write_i32(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.int32).tofile(path)


def dequant_fp8(weight, scale, block):
    w = np.asarray(weight, dtype=np.uint8)
    s = np.asarray(scale, dtype=np.uint8)
    rows, cols = w.shape
    # scale is [rows/block, cols/block]; broadcast to each element.
    s = np.power(2.0, s.astype(np.float64) - 127.0)
    s_big = np.repeat(np.repeat(s, block, axis=0), block, axis=1)
    sign = np.where((w & 0x80) != 0, -1.0, 1.0)
    exp = (w >> 3) & 0x0F
    man = (w & 0x07).astype(np.float64)
    val = np.where(exp == 0, man * (2.0 ** -9),
                   (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0))
    out = np.round(sign * val * s_big * FIXED)
    return np.clip(out, -(2**31), 2**31 - 1).astype(np.int32)


def dequant_fp4(weight, scale, logical_cols, block):
    # `weight` is bytes: each byte packs two E2M1 nibbles, low first.
    w = np.asarray(weight, dtype=np.uint8)
    s = np.asarray(scale, dtype=np.uint8)
    rows, packed_cols = w.shape
    assert packed_cols * 2 == logical_cols
    s = np.power(2.0, s.astype(np.float64) - 127.0)
    s_big = np.repeat(np.repeat(s, block, axis=0), block, axis=1)
    low = (w & 0x0F).astype(np.int64)
    high = (w >> 4).astype(np.int64)
    table = np.asarray(FP4_TABLE, dtype=np.float64)
    unpacked = np.stack([table[low], table[high]], axis=-1).reshape(rows, logical_cols)
    out = np.round(unpacked * s_big * FIXED)
    return np.clip(out, -(2**31), 2**31 - 1).astype(np.int32)


def bf16_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def extract_tensor(dtype, shape, data):
    if dtype == "F32":
        vals = np.frombuffer(data, dtype="<f4").astype(np.float64)
        return np.clip(np.round(vals * FIXED), -(2**31), 2**31 - 1).astype(np.int32)
    if dtype == "BF16":
        bits = np.frombuffer(data, dtype="<u2").astype(np.uint32)
        vals = (bits << 16).astype(np.uint32).view(np.float32).astype(np.float64)
        return np.clip(np.round(vals * FIXED), -(2**31), 2**31 - 1).astype(np.int32)
    if dtype == "F8_E4M3":
        w = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        sign = np.where((w & 0x80) != 0, -1.0, 1.0)
        exp = (w >> 3) & 0x0F
        man = (w & 0x07).astype(np.float64)
        val = np.where(exp == 0, man * (2.0 ** -9),
                       (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0))
        return np.clip(np.round(sign * val * FIXED), -(2**31), 2**31 - 1).astype(np.int32)
    if dtype == "F8_E8M0":
        s = np.frombuffer(data, dtype=np.uint8)
        vals = np.power(2.0, s.astype(np.float64) - 127.0)
        return np.clip(np.round(vals * FIXED), -(2**31), 2**31 - 1).astype(np.int32)
    if dtype == "I8":
        # Packed FP4 weights are handled together with their scale below.
        raise ValueError("raw I8 must be dequantized with its scale")
    raise ValueError(f"unsupported dtype {dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0")
    ap.add_argument("--only-text", action="store_true")
    ap.add_argument("--filter", default="", help="substring filter on tensor name")
    ap.add_argument("--rope-out", default="", help="output directory for RoPE cos/sin i32 bins")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--rope-head-dim", type=int, default=64)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)

    if args.rope_out:
        rope_dir = Path(args.rope_out)
        rope_dir.mkdir(parents=True, exist_ok=True)
        pos = np.arange(args.seq, dtype=np.float64)[:, None]
        inv = np.arange(args.rope_head_dim // 2, dtype=np.float64)[None, :]
        freqs = pos * np.power(args.rope_theta, -2.0 * inv / args.rope_head_dim)
        cos = np.clip(np.round(np.cos(freqs) * 65536.0), -(2**31), 2**31 - 1).astype(np.int32)
        sin = np.clip(np.round(np.sin(freqs) * 65536.0), -(2**31), 2**31 - 1).astype(np.int32)
        cos.tofile(rope_dir / "cos_i32.bin")
        sin.tofile(rope_dir / "sin_i32.bin")
        print(f"rope -> {rope_dir}/cos_i32.bin {rope_dir}/sin_i32.bin", flush=True)

    if not args.out:
        return

    layers = [int(x) for x in args.layers.split(",")]
    index_path = raw / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]

    # Group wanted tensor names by shard.
    wanted = []
    for name in index:
        if name.startswith("vision.") and args.only_text:
            continue
        if name.startswith("layers."):
            try:
                layer = int(name.split(".")[1])
            except ValueError:
                continue
            if layer not in layers:
                continue
        wanted.append(name)
    shard_files = sorted({index[name] for name in wanted})
    shards = {}
    for shard in shard_files:
        shards[shard] = read_safetensors(raw / shard)

    for name in wanted:
        if args.filter and args.filter not in name:
            continue
        shard = shards[index[name]]
        dtype, shape, data = shard[name]
        safe = name.replace(".", "_").replace("/", "_")
        target = out / (safe + ".i32")
        if dtype == "I8":
            # Expert weight: packed FP4. Locate its E8M0 scale by suffix.
            scale_name = name.replace("weight", "scale")
            if scale_name not in shard:
                print(f"SKIP {name}: no scale {scale_name}", flush=True)
                continue
            sdtype, sshape, sdata = shard[scale_name]
            rows, packed_cols = shape
            logical_cols = packed_cols * 2
            block = 32
            scale = np.frombuffer(sdata, dtype=np.uint8).reshape(sshape)
            wbytes = np.frombuffer(data, dtype=np.uint8).reshape(shape)
            values = dequant_fp4(wbytes, scale, logical_cols, block)
            write_i32(target, values)
        elif dtype == "F8_E4M3" and shape[-1] > 0 and len(shape) == 2:
            scale_name = name.replace("weight", "scale")
            if scale_name not in shard:
                print(f"SKIP {name}: no scale {scale_name}", flush=True)
                continue
            sdtype, sshape, sdata = shard[scale_name]
            rows, cols = shape
            block = 32
            scale = np.frombuffer(sdata, dtype=np.uint8).reshape(sshape)
            wbytes = np.frombuffer(data, dtype=np.uint8).reshape(shape)
            values = dequant_fp8(wbytes, scale, block)
            write_i32(target, values)
        else:
            values = extract_tensor(dtype, shape, data)
            write_i32(target, values)
        print(f"{name} -> {target.relative_to(out)} ({len(values)})", flush=True)


if __name__ == "__main__":
    main()
