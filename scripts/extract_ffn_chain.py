#!/usr/bin/env python3
"""Run the quantized fixed-point FFN for layer 0 and dump consistent tensors.

The Rust proof proves the *quantized* computation, so the intermediate tensors
it commits must be produced by the same fixed-point rules (round half-to-even,
the rsqrt table, padding to powers of two) — not by independently quantizing
each ONNX node from floats (which drifts by several LSBs across a matmul).

This replay of layer 0's FFN tail produces exactly that consistent chain:

    norm  -> gate -> rescale+bias -> ReLU -> down -> rescale+bias -> residual

and verifies the simulated residual stays close to the ONNX float reference.

Usage: .venv-timesfm/bin/python scripts/extract_ffn_chain.py
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0
INDEX_SCALE = 1 << 14


def div_round(a: int, b: int) -> int:
    """Round a/b to nearest, ties toward +inf (matches the Rust div_round)."""
    q, r = divmod(a, b)
    return q + 1 if r * 2 >= b else q


def dump_i32(path: str, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr, dtype=np.int32).tofile(path)


def dump_i64(path: str, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr, dtype=np.int64).tofile(path)


def main() -> None:
    base = "models"
    out_dir = f"{base}/ffn_chain"
    os.makedirs(out_dir, exist_ok=True)

    rsqrt = np.fromfile(f"{base}/rsqrt_table_i32.bin", dtype=np.int32).astype(np.int64)
    x = np.fromfile(f"{base}/norms/L0_in.bin", dtype=np.int32).astype(np.int64)
    nw = np.fromfile(f"{base}/norms/L0_w.bin", dtype=np.int32).astype(np.int64)
    nb = np.fromfile(f"{base}/norms/L0_b.bin", dtype=np.int32).astype(np.int64)
    gate = np.fromfile(f"{base}/weights/val_126_512x1024_i32.bin", dtype=np.int32).astype(np.int64)
    down = np.fromfile(f"{base}/weights/val_128_1024x512_i32.bin", dtype=np.int32).astype(np.int64)
    gate = gate.reshape(512, 1024)
    down = down.reshape(1024, 512)

    m = onnx.load(f"{base}/timesfm_8m_fintext_ctx32.onnx")
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
    gate_bias = np.round(inits["stacked_transformer.layers.0.mlp.gate_proj.bias"] * SCALE).astype(np.int64)
    down_bias = np.zeros(512, dtype=np.int64)
    down_bias[:264] = np.round(inits["stacked_transformer.layers.0.mlp.down_proj.bias"] * SCALE).astype(np.int64)

    n_real = int((nw != 0).sum())
    xr = x[:n_real]
    mean = div_round(int(xr.sum()), n_real)
    var = div_round(int(((xr - mean) ** 2).sum()), n_real)
    s_index = div_round(var, 1 << 18)
    rstd = int(rsqrt[s_index])
    norm_raw = (x - mean) * rstd * nw  # scale 2^48, padding yields 0
    norm_out = np.array([div_round(int(v), 1 << 32) for v in norm_raw]) + nb

    gate_raw = (norm_out.reshape(1, 512) @ gate).reshape(-1)  # scale 2^32
    linear5 = np.array([div_round(int(v), 1 << 16) for v in gate_raw]) + gate_bias
    relu = np.maximum(linear5, 0)

    down_raw = (relu.reshape(1, 1024) @ down).reshape(-1)  # scale 2^32
    ffn_out = np.array([div_round(int(v), 1 << 16) for v in down_raw]) + down_bias
    residual = ffn_out + x  # add_6 = linear_6 + add_5

    # Reference float outputs from ONNX (layer 0 only).
    g = m.graph
    for name in ("add_6",):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_ffn_chain.onnx")
    sess = ort.InferenceSession("/tmp/tsfm_ffn_chain.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad = np.zeros((1, 32), dtype=np.float32)
    out_names = [o.name for o in m2.graph.output]
    res = sess.run(out_names, {
        "input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64),
    })
    add6_float = res[out_names.index("add_6")].reshape(-1)
    sim_float = residual[: add6_float.size] / SCALE
    err = float(np.abs(sim_float - add6_float).max())

    # Dump everything the proof commits.
    dump_i32(f"{out_dir}/norm_out_i32.bin", norm_out)
    dump_i64(f"{out_dir}/norm_raw_i64.bin", norm_raw)
    dump_i32(f"{out_dir}/gate_bias_i32.bin", gate_bias)
    dump_i64(f"{out_dir}/gate_raw_i64.bin", gate_raw)
    dump_i32(f"{out_dir}/relu_i32.bin", relu)
    dump_i32(f"{out_dir}/down_bias_i32.bin", down_bias)
    dump_i64(f"{out_dir}/down_raw_i64.bin", down_raw)
    dump_i32(f"{out_dir}/ffn_out_i32.bin", ffn_out)
    dump_i32(f"{out_dir}/residual_i32.bin", residual)

    print(f"layer-0 FFN chain dumped under {out_dir}/")
    print(f"n_real={n_real} mean={mean} var={var} s_index={s_index} rstd={rstd}")
    print(f"simulated residual max abs error vs ONNX float = {err:.3e}")
    assert err < 2e-3, f"quantized FFN drifted too far from float: {err:.3e}"


if __name__ == "__main__":
    main()
