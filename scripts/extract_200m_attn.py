#!/usr/bin/env python3
"""Dump per-layer Q/K/V projection weights for the 200M attention, with the
per-head_dim Q scale folded into Q.

The fused qkv weight `[H, 3H]` is split into Q/K/V `[H, H]`. The Q part has the
per-head_dim scale (unsqueeze_29) folded in so the Rust prover uses a plain
matmul + affine for all three projections. Folding changes the rounding order by
~1 LSB versus the ONNX reference; this is a documented fixed-point approximation.

Usage: .venv-timesfm/bin/python scripts/extract_200m_attn.py
"""
import os
import re

import numpy as np
import onnx

SCALE = 65536.0
H = 1280
H_PAD = 2048
HDIM = 80
HEADS = 16
N_LAYERS = 20


def q(arr):
    return np.clip(np.round(arr * SCALE), -(2**31), 2**31 - 1).astype(np.int32)


def pad1(arr, n):
    o = np.zeros(n, dtype=np.int32)
    o[: arr.size] = arr
    return o


def pad2(arr, r, c):
    o = np.zeros((r, c), dtype=np.int32)
    o[: arr.shape[0], : arr.shape[1]] = arr
    return o


def main():
    base = "models"
    out_dir = f"{base}/full_stack_200m"
    os.makedirs(out_dir, exist_ok=True)

    m = onnx.load(f"{base}/timesfm_1_0_200m.onnx")
    g = m.graph
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in g.initializer}
    q_scale = q(inits["unsqueeze_29"].reshape(HDIM))

    # Find each layer's fused qkv weight name by walking the MatMul -> Add chain.
    layer_weights = {}
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        w = [x for x in n.input if x in inits and len(inits[x].shape) == 2]
        if not w:
            continue
        out = n.output[0]
        for n2 in g.node:
            if n2.op_type == "Add" and out in n2.input:
                for b in n2.input:
                    mm = re.search(r"layers\.(\d+)\.", b)
                    if mm:
                        layer_weights.setdefault(int(mm.group(1)), []).append((w[0], out))
                        break

    for L in range(N_LAYERS):
        ws = layer_weights[L]
        assert len(ws) == 4, f"layer {L}: expected 4 weights, got {len(ws)}"
        qkv_name = ws[0][0]
        qkv_w = q(inits[qkv_name])  # [H, 3H]
        p = f"stacked_transformer.layers.{L}"
        qkv_b = q(inits[f"{p}.self_attn.qkv_proj.bias"])  # [3H]

        q_w, k_w, v_w = qkv_w[:, 0:H], qkv_w[:, H:2 * H], qkv_w[:, 2 * H:3 * H]
        q_b, k_b, v_b = qkv_b[0:H], qkv_b[H:2 * H], qkv_b[2 * H:3 * H]

        # Fold q_scale into Q over each head's head_dim (d = h * HDIM + j).
        q_w_f = np.zeros_like(q_w)
        q_b_f = np.zeros_like(q_b)
        for h in range(HEADS):
            for j in range(HDIM):
                d = h * HDIM + j
                q_w_f[:, d] = np.clip(
                    np.round(q_w[:, d].astype(np.float64) * q_scale[j] / SCALE),
                    -(2**31), 2**31 - 1,
                ).astype(np.int32)
                q_b_f[d] = int(np.clip(
                    np.round(q_b[d] * q_scale[j] / SCALE), -(2**31), 2**31 - 1
                ))

        pad2(q_w_f, H_PAD, H_PAD).tofile(f"{out_dir}/L{L}_q_w_i32.bin")
        pad2(k_w, H_PAD, H_PAD).tofile(f"{out_dir}/L{L}_k_w_i32.bin")
        pad2(v_w, H_PAD, H_PAD).tofile(f"{out_dir}/L{L}_v_w_i32.bin")
        pad1(q_b_f, H_PAD).tofile(f"{out_dir}/L{L}_q_b_i32.bin")
        pad1(k_b, H_PAD).tofile(f"{out_dir}/L{L}_k_b_i32.bin")
        pad1(v_b, H_PAD).tofile(f"{out_dir}/L{L}_v_b_i32.bin")

    print(f"dumped per-layer Q/K/V projection weights under {out_dir}/")


if __name__ == "__main__":
    main()
