#!/usr/bin/env python3
"""Run the quantized fixed-point attention block (layer 0) and dump tensors.

At seq=1 the attention degenerates (softmax = 1, so `softmax @ V = V`), which
means the attention output is exactly the V chunk of the fused QKV projection.
This replays the block with the same fixed-point rules as the Rust prover:

    RMSNorm -> V matmul -> rescale+bias -> o_proj matmul -> rescale+bias -> residual

and verifies the simulated residual stays close to the ONNX float reference.

Usage: .venv-timesfm/bin/python scripts/extract_attention_chain.py
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0


def div_round(a: int, b: int) -> int:
    q, r = divmod(a, b)
    return q + 1 if r * 2 >= b else q


def dump_i32(path: str, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr, dtype=np.int32).tofile(path)


def dump_i64(path: str, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr, dtype=np.int64).tofile(path)


def pad(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    out[: arr.size] = arr
    return out


def pad2(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.int64)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def main() -> None:
    base = "models"
    out_dir = f"{base}/attention_chain"
    os.makedirs(out_dir, exist_ok=True)

    m = onnx.load(f"{base}/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}

    ln_w = inits["stacked_transformer.layers.0.input_layernorm.weight"]
    qkv = inits["val_94"]            # [264, 792] fused Q,K,V
    qkv_bias = inits["stacked_transformer.layers.0.self_attn.qkv_proj.bias"]
    op_w = inits["val_122"]          # [264, 264]
    op_b = inits["stacked_transformer.layers.0.self_attn.o_proj.bias"]

    v_w = qkv[:, 528:792]            # V chunk of the fused QKV weight
    v_b = qkv_bias[528:792]

    rsqrt = np.fromfile(f"{base}/rsqrt_table_i32.bin", dtype=np.int32).astype(np.int64)

    # Reference floats for add_2 (input) and add_5 (residual output).
    for name in ("add_2", "add_5"):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_attention.onnx")
    sess = ort.InferenceSession("/tmp/tsfm_attention.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad_in = np.zeros((1, 32), dtype=np.float32)
    res = sess.run(["add_2", "add_5"], {
        "input_ts": inp, "input_padding": pad_in, "freq": np.array([[0]], dtype=np.int64),
    })
    add2_float = res[0].reshape(-1)
    add5_float = res[1].reshape(-1)

    # Quantize the fixed weights and the input.
    add2_q = pad(np.round(add2_float * SCALE), 512)
    ln_w_q = pad(np.round(ln_w * SCALE), 512)
    v_w_q = pad2(np.round(v_w * SCALE), 512, 512)
    v_b_q = pad(np.round(v_b * SCALE), 512)
    op_w_q = pad2(np.round(op_w * SCALE), 512, 512)
    op_b_q = pad(np.round(op_b * SCALE), 512)

    n_real = 264
    # RMSNorm: mul_9 = add_2 * rsqrt(mean(add_2^2)+eps) * w.
    sqsum = int((add2_q[:n_real] ** 2).sum())
    s_index = div_round(div_round(sqsum, n_real), 1 << 18)
    rstd = int(rsqrt[s_index])
    rms_raw = add2_q * rstd * ln_w_q          # scale 2^48, padding yields 0
    mul9_q = np.array([div_round(int(v), 1 << 32) for v in rms_raw])

    # V matmul: V = mul_9 @ v_w (scale 2^32), then rescale + bias.
    v_raw = (mul9_q.reshape(1, 512) @ v_w_q).reshape(-1)
    lin3v_q = np.array([div_round(int(v), 1 << 16) for v in v_raw]) + v_b_q

    # o_proj matmul: val_123 = lin3v @ op_w, then rescale + bias.
    op_raw = (lin3v_q.reshape(1, 512) @ op_w_q).reshape(-1)
    lin4_q = np.array([div_round(int(v), 1 << 16) for v in op_raw]) + op_b_q

    # residual: add_5 = add_2 + linear_4.
    add5_q = add2_q + lin4_q

    err = float(np.abs(add5_q[: add5_float.size] / SCALE - add5_float).max())

    dump_i32(f"{out_dir}/add2_i32.bin", add2_q)
    dump_i32(f"{out_dir}/ln_w_i32.bin", ln_w_q)
    dump_i64(f"{out_dir}/rms_raw_i64.bin", rms_raw)
    dump_i32(f"{out_dir}/mul9_i32.bin", mul9_q)
    dump_i32(f"{out_dir}/v_w_i32.bin", v_w_q)
    dump_i32(f"{out_dir}/v_b_i32.bin", v_b_q)
    dump_i64(f"{out_dir}/v_raw_i64.bin", v_raw)
    dump_i32(f"{out_dir}/lin3v_i32.bin", lin3v_q)
    dump_i32(f"{out_dir}/op_w_i32.bin", op_w_q)
    dump_i32(f"{out_dir}/op_b_i32.bin", op_b_q)
    dump_i64(f"{out_dir}/op_raw_i64.bin", op_raw)
    dump_i32(f"{out_dir}/lin4_i32.bin", lin4_q)
    dump_i32(f"{out_dir}/add5_i32.bin", add5_q)

    print(f"layer-0 attention chain dumped under {out_dir}/")
    print(f"s_index={s_index} rstd={rstd}")
    print(f"simulated residual max abs error vs ONNX float = {err:.3e}")
    assert err < 2e-3, f"quantized attention drifted too far from float: {err:.3e}"


if __name__ == "__main__":
    main()
