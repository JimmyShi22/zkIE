#!/usr/bin/env python3
"""Dump the TimesFM 200M weights/biases/constants for the GKR prover.

200M structure (same family as 8M, but non-degenerate attention at seq=16):
  hidden = 1280 (pad 2048), qkv fused = 3840 (pad 4096), 20 layers.
  Per layer: RMSNorm -> QKV -> QK^T -> softmax -> PV -> o_proj -> residual ->
             LayerNorm -> gate -> ReLU -> down -> residual.
The FFN intermediate equals hidden (1280), unlike the 8M (1024 > 264).

Usage: .venv-timesfm/bin/python scripts/extract_200m.py
"""
import os
import re

import numpy as np
import onnx

SCALE = 65536.0
H = 1280
H_PAD = 2048
QKV_PAD = 4096
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

    # Map each layer's four matmul weights by walking nodes and matching the
    # downstream bias names.
    layer_weights = {}
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        w = [x for x in n.input if x in inits and len(inits[x].shape) == 2]
        if not w:
            continue
        wname = w[0]
        out = n.output[0]
        for n2 in g.node:
            if n2.op_type == "Add" and out in n2.input:
                for b in n2.input:
                    mm = re.search(r"layers\.(\d+)\.", b)
                    if mm:
                        layer_weights.setdefault(int(mm.group(1)), []).append((wname, out))
                        break

    for L in range(N_LAYERS):
        ws = layer_weights[L]
        assert len(ws) == 4, f"layer {L}: expected 4 weights, got {len(ws)}"
        # node order within a layer: qkv, o_proj, gate, down.
        qkv_w, o_proj_w, gate_w, down_w = [w for w, o in ws]
        assert inits[qkv_w].shape == (H, 3 * H)
        assert inits[o_proj_w].shape == (H, H)
        assert inits[gate_w].shape == (H, H)
        assert inits[down_w].shape == (H, H)
        print(f"L{L}: qkv={qkv_w} o_proj={o_proj_w} gate={gate_w} down={down_w}")

        pad2(q(inits[qkv_w]), H_PAD, QKV_PAD).tofile(f"{out_dir}/L{L}_qkv_w_i32.bin")
        pad2(q(inits[o_proj_w]), H_PAD, H_PAD).tofile(f"{out_dir}/L{L}_o_proj_w_i32.bin")
        pad2(q(inits[gate_w]), H_PAD, H_PAD).tofile(f"{out_dir}/L{L}_gate_w_i32.bin")
        pad2(q(inits[down_w]), H_PAD, H_PAD).tofile(f"{out_dir}/L{L}_down_w_i32.bin")

        q(inits[f"stacked_transformer.layers.{L}.self_attn.qkv_proj.bias"]).tofile(f"{out_dir}/L{L}_qkv_b_i32.bin")
        pad1(q(inits[f"stacked_transformer.layers.{L}.self_attn.o_proj.bias"]), H_PAD).tofile(f"{out_dir}/L{L}_o_proj_b_i32.bin")
        pad1(q(inits[f"stacked_transformer.layers.{L}.mlp.gate_proj.bias"]), H_PAD).tofile(f"{out_dir}/L{L}_gate_b_i32.bin")
        pad1(q(inits[f"stacked_transformer.layers.{L}.mlp.down_proj.bias"]), H_PAD).tofile(f"{out_dir}/L{L}_down_b_i32.bin")
        pad1(q(inits[f"stacked_transformer.layers.{L}.input_layernorm.weight"]), H_PAD).tofile(f"{out_dir}/L{L}_lnw_i32.bin")
        pad1(q(inits[f"stacked_transformer.layers.{L}.mlp.layer_norm.weight"]), H_PAD).tofile(f"{out_dir}/L{L}_mlp_w_i32.bin")
        pad1(q(inits[f"stacked_transformer.layers.{L}.mlp.layer_norm.bias"]), H_PAD).tofile(f"{out_dir}/L{L}_mlp_b_i32.bin")

    print(f"dumped {N_LAYERS} layers under {out_dir}/")


if __name__ == "__main__":
    main()
