#!/usr/bin/env python3
"""Verify the attention chain (QK^T -> softmax -> PV) for the ctx32 model.

At seq=1 the attention is degenerate: QK^T yields one score per head, softmax of
a single value is 1, so PV = V. Crucially, QK^T is a *batched dot product*
(element-wise multiply then per-head sum), not a GKR matmul, so it is covered by
plain field arithmetic rather than a sum-check. A longer context would make the
attention non-degenerate (and head_dim=66 would need padding to a power of two).

Usage: .venv-timesfm/bin/python scripts/verify_attention.py
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    for name in ("transpose", "transpose_3", "matmul", "softmax", "transpose_2", "matmul_1"):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_attn.onnx")
    sess = ort.InferenceSession("/tmp/tsfm_attn.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad = np.zeros((1, 32), dtype=np.float32)
    res = sess.run([o.name for o in m2.graph.output], {
        "input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64),
    })
    names = [o.name for o in m2.graph.output]
    q = res[names.index("transpose")].reshape(-1)
    kt = res[names.index("transpose_3")].reshape(-1)
    scores = res[names.index("matmul")].reshape(-1)
    sm = res[names.index("softmax")].reshape(-1)
    v = res[names.index("transpose_2")].reshape(-1)
    pv = res[names.index("matmul_1")].reshape(-1)
    # per-head dot product (4 heads x 66 head_dim): element-wise mul + reduce sum
    qh = q.reshape(4, 66)
    kh = kt.reshape(4, 66)
    scores_recon = np.sum(qh * kh, axis=1)
    print(f"QK^T  max err = {np.abs(scores_recon - scores).max():.2e}")
    print(f"softmax == 1 : {np.allclose(sm, 1.0)}")
    print(f"PV == V      : {np.allclose(pv, v)}")


if __name__ == "__main__":
    main()
