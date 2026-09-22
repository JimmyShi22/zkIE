#!/usr/bin/env python3
"""Verify the int32 quantized layer-0 matmuls match the real activations.

Loads the padded int32 weights and activations, recomputes each of the four
weight matmuls (QKV, o_proj, gate, down) at scale 2^16, rescales by 2^16, and
reports the error against the extracted activation. Confirms the fixed-point
scale reconciliation for the whole layer's matmul chain.

Usage: .venv-timesfm/bin/python scripts/verify_layer0_quantization.py
"""
import numpy as np

SCALE = 65536.0


def load(path, full_shape, keep):
    a = np.fromfile(path, dtype=np.int32).astype(np.int64).reshape(full_shape)
    return a[tuple(slice(0, k) for k in keep)]


def main() -> None:
    w = "models/weights"
    a = "models/activations"
    qkv = load(f"{w}/val_94_512x1024_i32.bin", (512, 1024), (264, 792))
    o = load(f"{w}/val_122_512x512_i32.bin", (512, 512), (264, 264))
    gate = load(f"{w}/val_126_512x1024_i32.bin", (512, 1024), (264, 1024))
    down = load(f"{w}/val_128_1024x512_i32.bin", (1024, 512), (1024, 264))

    inp = load(f"{a}/mul_9_512_i32.bin", (512,), (264,))
    qkvout = load(f"{a}/val_95_1024_i32.bin", (1024,), (792,))
    view5 = load(f"{a}/view_5_512_i32.bin", (512,), (264,))
    oprj = load(f"{a}/val_123_512_i32.bin", (512,), (264,))
    ffnin = load(f"{a}/layer_norm_512_i32.bin", (512,), (264,))
    gateout = load(f"{a}/val_127_1024_i32.bin", (1024,), (1024,))
    relu = load(f"{a}/relu_1024_i32.bin", (1024,), (1024,))
    downout = load(f"{a}/val_129_512_i32.bin", (512,), (264,))

    for name, raw, expected in [
        ("qkv", inp @ qkv, qkvout),
        ("o_proj", view5 @ o, oprj),
        ("gate", ffnin @ gate, gateout),
        ("down", relu @ down, downout),
    ]:
        recon = np.round(raw / SCALE).astype(np.int64)
        d = np.abs(recon - expected)
        rel = d.mean() / (np.abs(expected).mean() + 1e-6)
        print(f"{name:8} max={d.max():6} rel={rel:.4%}")


if __name__ == "__main__":
    main()
