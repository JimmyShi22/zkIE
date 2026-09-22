#!/usr/bin/env python3
"""Dump the TimesFM 200M prologue/epilogue/attention constants for the GKR prover.

`extract_200m.py` already dumps the 20 per-layer weights/biases/norms. This
script dumps the remaining fixed tensors the full-stack proof needs:
  - attention constants: q_scale (per-head_dim), causal mask
  - input FFN weights/biases (val_50/53/55) and the embedded `cat` input
  - freq embedding (gather/embedding for freq=0) and the input_ts/RevIN inputs
  - output head (horizon FFN) weights/biases and the rescale scalars u2/u4

Usage: .venv-timesfm/bin/python scripts/extract_200m_full.py
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0
H = 1280
H_PAD = 2048
SEQ = 16
HEADS = 16
HDIM = 80
HDIM_PAD = 128


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

    # Attention constants: per-head_dim Q scale and the causal mask.
    q_scale = pad1(q(inits["unsqueeze_29"].reshape(HDIM)), HDIM_PAD)
    q_scale.tofile(f"{out_dir}/q_scale_i32.bin")
    # Masked (future) positions: use -2^21 (= -32 in the exp table's scale) so
    # softmax -> ~0 without overflowing i32 when a negative logit is added.
    mask = np.triu(np.full((SEQ, SEQ), -(1 << 21)), k=1).astype(np.int32)
    mask.tofile(f"{out_dir}/mask_q_i32.bin")

    # Input FFN: cat [SEQ, 64] -> SiLU FFN -> add. The cat input is the RevIN
    # embedding (computed in Rust), so only the weights/biases are dumped here.
    pad2(q(inits["val_50"]), 64, H_PAD).tofile(f"{out_dir}/pro_hid_w_i32.bin")
    pad1(q(inits["input_ff_layer.hidden_layer.0.bias"]), H_PAD).tofile(f"{out_dir}/pro_hid_b_i32.bin")
    pad2(q(inits["val_53"]), H_PAD, H_PAD).tofile(f"{out_dir}/pro_out_w_i32.bin")
    pad1(q(inits["input_ff_layer.output_layer.bias"]), H_PAD).tofile(f"{out_dir}/pro_out_b_i32.bin")
    pad2(q(inits["val_55"]), 64, H_PAD).tofile(f"{out_dir}/pro_res_w_i32.bin")
    pad1(q(inits["input_ff_layer.residual_layer.bias"]), H_PAD).tofile(f"{out_dir}/pro_res_b_i32.bin")

    # Output head (horizon FFN) + rescale scalars.
    pad2(q(inits["val_850"]), H_PAD, H_PAD).tofile(f"{out_dir}/head_hid_w_i32.bin")
    pad1(q(inits["horizon_ff_layer.hidden_layer.0.bias"]), H_PAD).tofile(f"{out_dir}/head_hid_b_i32.bin")
    pad2(q(inits["val_853"]), H_PAD, H_PAD).tofile(f"{out_dir}/head_out_w_i32.bin")
    pad1(q(inits["horizon_ff_layer.output_layer.bias"]), H_PAD).tofile(f"{out_dir}/head_out_b_i32.bin")
    pad2(q(inits["val_855"]), H_PAD, H_PAD).tofile(f"{out_dir}/head_res_w_i32.bin")
    pad1(q(inits["horizon_ff_layer.residual_layer.bias"]), H_PAD).tofile(f"{out_dir}/head_res_b_i32.bin")

    # Runtime tensors for a fixed input (freq=0): cat, gather, embedding, and
    # the rescale scalars unsqueeze_2/unsqueeze_4.
    inp = np.random.RandomState(0).randn(1, 512).astype(np.float32)
    pad_in = np.zeros((1, 512), dtype=np.float32)
    g2 = m.graph
    for name in ("cat", "gather", "embedding", "unsqueeze_2", "unsqueeze_4"):
        g2.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g2, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/s200_full.onnx")
    sess = ort.InferenceSession("/tmp/s200_full.onnx", providers=["CPUExecutionProvider"])
    names = [o.name for o in m2.graph.output]
    r = sess.run(names, {"input_ts": inp, "input_padding": pad_in, "freq": np.array([[0]], dtype=np.int64)})

    cat = r[names.index("cat")].reshape(-1)
    gather = r[names.index("gather")].reshape(SEQ, H)
    embedding = r[names.index("embedding")].reshape(H)
    u2 = float(np.asarray(r[names.index("unsqueeze_2")]).reshape(-1)[0])
    u4 = float(np.asarray(r[names.index("unsqueeze_4")]).reshape(-1)[0])

    pad1(q(inp.reshape(-1)), 512).tofile(f"{out_dir}/input_ts_i32.bin")
    pad1(q(cat), 64 * SEQ).tofile(f"{out_dir}/cat_i32.bin")
    pad2(q(gather), SEQ, H_PAD).tofile(f"{out_dir}/gather_i32.bin")
    pad1(q(embedding), H_PAD).tofile(f"{out_dir}/embedding_i32.bin")
    np.array([round(u4 * SCALE), round(u2 * SCALE)], dtype=np.int64).tofile(f"{out_dir}/scale_i64.bin")

    print(f"dumped prologue/epilogue/attention constants under {out_dir}/")


if __name__ == "__main__":
    main()
