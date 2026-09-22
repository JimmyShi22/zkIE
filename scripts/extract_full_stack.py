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

    # Epilogue (horizon FFN output head): hidden -> SiLU -> output + residual ->
    # rescale. Dump the weights/biases and the scalar rescale factors.
    pad2(q(inits["val_352"]), 512, 1024).tofile(f"{out_dir}/head_hid_w_i32.bin")
    pad2(q(inits["val_355"]), 1024, 2048).tofile(f"{out_dir}/head_out_w_i32.bin")
    pad2(q(inits["val_357"]), 512, 2048).tofile(f"{out_dir}/head_res_w_i32.bin")
    q(inits["horizon_ff_layer.hidden_layer.0.bias"]).tofile(f"{out_dir}/head_hid_b_i32.bin")
    pad1(q(inits["horizon_ff_layer.output_layer.bias"]), 2048).tofile(f"{out_dir}/head_out_b_i32.bin")
    pad1(q(inits["horizon_ff_layer.residual_layer.bias"]), 2048).tofile(f"{out_dir}/head_res_b_i32.bin")

    # The rescale scalars (unsqueeze_4, unsqueeze_2) depend on the input
    # normalization; extract them for this fixed input.
    g2 = m.graph
    for name in ("unsqueeze_4", "unsqueeze_2"):
        g2.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m3 = helper.make_model(g2, opset_imports=m.opset_import)
    onnx.save(m3, "/tmp/tsfm_scalars.onnx")
    sess3 = ort.InferenceSession("/tmp/tsfm_scalars.onnx", providers=["CPUExecutionProvider"])
    names3 = [o.name for o in m3.graph.output]
    r3 = sess3.run(names3, {"input_ts": inp, "input_padding": pad_in, "freq": np.array([[0]], dtype=np.int64)})
    u4 = float(np.asarray(r3[names3.index("unsqueeze_4")]).reshape(-1)[0])
    u2 = float(np.asarray(r3[names3.index("unsqueeze_2")]).reshape(-1)[0])
    np.array([round(u4 * SCALE), round(u2 * SCALE)], dtype=np.int64).tofile(f"{out_dir}/scale_i64.bin")

    # Prologue: cat (embedded input, 64) -> SiLU input FFN -> add, then +gather
    # +embedding (freq) -> add_2. `cat`, `gather`, `embedding` are the fixed
    # embedded inputs for this input/freq; the SiLU FFN and additions are proven.
    g3 = m.graph
    for name in ("cat", "gather", "embedding"):
        g3.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m4 = helper.make_model(g3, opset_imports=m.opset_import)
    onnx.save(m4, "/tmp/tsfm_prologue.onnx")
    sess4 = ort.InferenceSession("/tmp/tsfm_prologue.onnx", providers=["CPUExecutionProvider"])
    names4 = [o.name for o in m4.graph.output]
    r4 = sess4.run(names4, {"input_ts": inp, "input_padding": pad_in, "freq": np.array([[0]], dtype=np.int64)})
    cat = r4[names4.index("cat")].reshape(-1)
    gather = r4[names4.index("gather")].reshape(-1)
    embedding = r4[names4.index("embedding")].reshape(-1)

    # Raw input (input_ts) and the padding mask (where_1, all zeros for the
    # fixed input). The input embedding is a LayerNorm over the 32 timestamps
    # plus a concat with the mask; the Rust proof computes it from these.
    pad1(q(inp.reshape(-1)), 32).tofile(f"{out_dir}/input_ts_i32.bin")
    pad1(np.zeros(32, dtype=np.int32), 32).tofile(f"{out_dir}/input_pad_i32.bin")
    pad1(q(cat), 64).tofile(f"{out_dir}/cat_i32.bin")
    pad2(q(inits["val_49"]), 64, 1024).tofile(f"{out_dir}/pro_hid_w_i32.bin")
    q(inits["input_ff_layer.hidden_layer.0.bias"]).tofile(f"{out_dir}/pro_hid_b_i32.bin")
    pad2(q(inits["val_52"]), 1024, 512).tofile(f"{out_dir}/pro_out_w_i32.bin")
    pad1(q(inits["input_ff_layer.output_layer.bias"]), 512).tofile(f"{out_dir}/pro_out_b_i32.bin")
    pad2(q(inits["val_54"]), 64, 512).tofile(f"{out_dir}/pro_res_w_i32.bin")
    pad1(q(inits["input_ff_layer.residual_layer.bias"]), 512).tofile(f"{out_dir}/pro_res_b_i32.bin")
    pad1(q(gather), 512).tofile(f"{out_dir}/gather_i32.bin")
    pad1(q(embedding), 512).tofile(f"{out_dir}/embedding_i32.bin")

    print(f"dumped full-stack + epilogue + prologue tensors under {out_dir}/")


if __name__ == "__main__":
    main()
