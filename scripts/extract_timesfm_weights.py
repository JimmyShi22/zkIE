#!/usr/bin/env python3
"""Extract and quantize all TimesFM 8M weights into padded int32 binaries.

Walks the ONNX graph, takes every 2D weight initializer, quantizes it at scale
2^12 into int32, zero-pads each dimension up to a power of two, and writes one
`.bin` per tensor under models/weights/ (name_{h}x{w}_i32.bin). The GKR matmul
requires power-of-two dims, so 264 -> 512 and 792 -> 1024.

Usage: .venv-timesfm/bin/python scripts/extract_timesfm_weights.py
"""
import os

import numpy as np
import onnx

SCALE = 65536.0  # 2^16, matching the activation extraction


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    out_dir = "models/weights"
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    for init in m.graph.initializer:
        dims = [d for d in init.dims]
        if len(dims) != 2:
            continue
        arr = onnx.numpy_helper.to_array(init).astype(np.float64)
        q = np.clip(np.round(arr * SCALE), -(2**31), 2**31 - 1).astype(np.int32)
        h, w = q.shape
        ph, pw = next_pow2(h), next_pow2(w)
        pad = np.zeros((ph, pw), dtype=np.int32)
        pad[:h, :w] = q
        path = os.path.join(out_dir, f"{init.name}_{ph}x{pw}_i32.bin")
        pad.tofile(path)
        total += h * w

    print(f"wrote padded int32 weights under {out_dir}/ (total params {total})")


if __name__ == "__main__":
    main()
