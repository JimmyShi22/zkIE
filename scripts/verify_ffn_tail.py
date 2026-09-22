#!/usr/bin/env python3
"""Verify the complete FFN tail (LayerNorm -> gate -> ReLU -> down -> residual)
quantized computation matches the real activations, with the per-matmul rescale.

Usage: .venv-timesfm/bin/python scripts/verify_ffn_tail.py
"""
import numpy as np
import onnx

SCALE = 65536.0


def load(path, full_shape, keep):
    return np.fromfile(path, dtype=np.int32).astype(np.int64).reshape(full_shape)[
        tuple(slice(0, k) for k in keep)
    ]


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
    gbias = np.round(inits["stacked_transformer.layers.0.mlp.gate_proj.bias"] * SCALE).astype(np.int64)
    dbias = np.round(inits["stacked_transformer.layers.0.mlp.down_proj.bias"] * SCALE).astype(np.int64)

    add5 = load("models/norm_in.bin", (512,), (264,))
    lnout = load("models/activations/layer_norm_512_i32.bin", (512,), (264,))
    ln_w = load("models/norm_w.bin", (512,), (264,))
    ln_b = load("models/norm_b.bin", (512,), (264,))
    gate = load("models/weights/val_126_512x1024_i32.bin", (512, 1024), (264, 1024))
    down = load("models/weights/val_128_1024x512_i32.bin", (1024, 512), (1024, 264))
    v127 = load("models/activations/val_127_1024_i32.bin", (1024,), (1024,))
    relu = load("models/activations/relu_1024_i32.bin", (1024,), (1024,))
    v129 = load("models/activations/val_129_512_i32.bin", (512,), (264,))

    # recompute mean/rstd from the float (via the scalar file)
    scalars = np.fromfile("models/norm_scalars_f64.bin", dtype=np.float64)
    mean = np.round(scalars[0] * SCALE).astype(np.int64)
    rstd = np.round(scalars[1] * SCALE).astype(np.int64)

    # LayerNorm with two rescales
    t1 = np.round((add5 - mean) * rstd // SCALE).astype(np.int64)
    t2 = np.round(t1 * ln_w // SCALE).astype(np.int64)
    ln_recon = t2 + ln_b
    d_ln = np.abs(ln_recon - lnout)
    print(f"layernorm  max={d_ln.max()} rel={d_ln.mean()/(np.abs(lnout).mean()+1e-6):.4%}")

    # gate matmul (raw 2^32, rescale 2^16) + bias + ReLU
    v127_raw = lnout @ gate
    v127_recon = np.round(v127_raw / SCALE).astype(np.int64)
    d_g = np.abs(v127_recon - v127)
    print(f"gate       max={d_g.max()} rel={d_g.mean()/(np.abs(v127).mean()+1e-6):.4%}")
    lin5 = v127 + gbias
    relu_recon = np.maximum(lin5, 0)
    d_r = np.abs(relu_recon - relu)
    print(f"relu       max={d_r.max()} rel={d_r.mean()/(np.abs(relu).mean()+1e-6):.4%}")

    # down matmul + bias
    v129_raw = relu @ down
    v129_recon = np.round(v129_raw / SCALE).astype(np.int64)
    d_d = np.abs(v129_recon - v129)
    print(f"down       max={d_d.max()} rel={d_d.mean()/(np.abs(v129).mean()+1e-6):.4%}")
    lin6 = v129 + dbias
    print("final (linear_6) = down + down_bias =", lin6[:3], "(residual add is host-side +add5)")


if __name__ == "__main__":
    main()
