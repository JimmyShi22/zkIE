#!/usr/bin/env python3
"""Dump per-layer weights, biases and the input residual for the full-stack proof.

The Rust `prove_full_stack` example recomputes every intermediate tensor in
fixed-point (using `rms_norm_raw` / `layer_norm_raw` / `affine_raw` / dense
matmul), so it only needs the quantized weights, biases, and the layer-0 input.
This script extracts and quantizes those, plus the V submatrix of each fused
QKV weight (columns 528..792), padded to powers of two.

Usage: .venv-timesfm/bin/python scripts/extract_full_stack.py
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0


def pad1(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int32)
    out[: arr.size] = arr
    return out


def pad2(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.int32)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def q(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr * SCALE), -(2**31), 2**31 - 1).astype(np.int32)


def main() -> None:
    base = "models"
    out_dir = f"{base}/full_stack"
    os.makedirs(out_dir, exist_ok=True)

    m = onnx.load(f"{base}/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}

    weights = [
        ["val_94", "val_122", "val_126", "val_128"],
        ["val_134", "val_159", "val_163", "val_165"],
        ["val_171", "val_196", "val_200", "val_202"],
        ["val_208", "val_233", "val_237", "val_239"],
        ["val_245", "val_270", "val_274", "val_276"],
        ["val_282", "val_307", "val_311", "val_313"],
        ["val_319", "val_344", "val_348", "val_350"],
    ]

    for li, (qw_name, ow_name, gw_name, dw_name) in enumerate(weights):
        qkv = inits[qw_name]
        v_w = pad2(q(qkv[:, 528:792]), 512, 512)
        op_w = pad2(q(inits[ow_name]), 512, 512)
        gate_w = pad2(q(inits[gw_name]), 512, 1024)
        down_w = pad2(q(inits[dw_name]), 1024, 512)
        v_w.tofile(f"{out_dir}/L{li}_v_w_i32.bin")
        op_w.tofile(f"{out_dir}/L{li}_op_w_i32.bin")
        gate_w.tofile(f"{out_dir}/L{li}_gate_w_i32.bin")
        down_w.tofile(f"{out_dir}/L{li}_down_w_i32.bin")

        qb = inits[f"stacked_transformer.layers.{li}.self_attn.qkv_proj.bias"]
        ob = inits[f"stacked_transformer.layers.{li}.self_attn.o_proj.bias"]
        gb = inits[f"stacked_transformer.layers.{li}.mlp.gate_proj.bias"]
        db = inits[f"stacked_transformer.layers.{li}.mlp.down_proj.bias"]
        pad1(q(qb[528:792]), 512).tofile(f"{out_dir}/L{li}_v_b_i32.bin")
        pad1(q(ob), 512).tofile(f"{out_dir}/L{li}_op_b_i32.bin")
        q(gb).tofile(f"{out_dir}/L{li}_gate_b_i32.bin")
        pad1(q(db), 512).tofile(f"{out_dir}/L{li}_down_b_i32.bin")

        lnw = inits[f"stacked_transformer.layers.{li}.input_layernorm.weight"]
        mlp_w = inits[f"stacked_transformer.layers.{li}.mlp.layer_norm.weight"]
        mlp_b = inits[f"stacked_transformer.layers.{li}.mlp.layer_norm.bias"]
        pad1(q(lnw), 512).tofile(f"{out_dir}/L{li}_lnw_i32.bin")
        pad1(q(mlp_w), 512).tofile(f"{out_dir}/L{li}_mlp_w_i32.bin")
        pad1(q(mlp_b), 512).tofile(f"{out_dir}/L{li}_mlp_b_i32.bin")

    # Layer-0 input residual (add_2), quantized and padded.
    g.output.append(helper.make_tensor_value_info("add_2", TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_extract.onnx")
    sess = ort.InferenceSession("/tmp/tsfm_extract.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad_in = np.zeros((1, 32), dtype=np.float32)
    add2 = sess.run(["add_2"], {
        "input_ts": inp, "input_padding": pad_in, "freq": np.array([[0]], dtype=np.int64),
    })[0].reshape(-1)
    pad1(q(add2), 512).tofile(f"{out_dir}/x_i32.bin")

    print(f"dumped full-stack weights/biases/input under {out_dir}/")


if __name__ == "__main__":
    main()
