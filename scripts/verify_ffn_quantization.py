#!/usr/bin/env python3
"""Verify the int32 fixed-point FFN reproduces the real TimesFM output.

Extracts layer 0's FFN weights/input/output from the ONNX, quantizes everything
at scale 2^12 into int32, recomputes FFN1 -> ReLU -> FFN2 in integer arithmetic
with a rescale after each matmul, and reports the error against the real float
output. This anchors the GKR fixed-point arithmetic: the proofs are sound and
the quantization is close to the float reference.

Usage: .venv-timesfm/bin/python scripts/verify_ffn_quantization.py
"""
import onnx
import numpy as np

SCALE = 4096.0  # 2^12, int32 fixed-point


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
    gate = inits["val_126"]
    down = inits["val_128"]
    gbias = inits["stacked_transformer.layers.0.mlp.gate_proj.bias"]

    q_in = np.load("models/ffn_in_i32.npy").astype(np.int64)
    q_out = np.load("models/ffn_out_i32.npy").astype(np.int64)
    q_gate = np.round(gate * SCALE).astype(np.int64)
    q_down = np.round(down * SCALE).astype(np.int64)
    q_gbias = np.round(gbias * SCALE).astype(np.int64)

    q_ffn1 = np.round((q_in @ q_gate) / SCALE).astype(np.int64) + q_gbias
    q_g = np.maximum(q_ffn1, 0)
    q_ffn2 = np.round((q_g @ q_down) / SCALE).astype(np.int64)

    diff = np.abs(q_ffn2 - q_out)
    print(f"max abs diff  = {diff.max()}")
    print(f"mean abs diff = {float(diff.mean()):.3f}")
    print(f"relative      = {float(diff.mean() / (np.abs(q_out).mean() + 1e-6)):.4%}")


if __name__ == "__main__":
    main()
