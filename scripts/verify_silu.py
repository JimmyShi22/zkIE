#!/usr/bin/env python3
"""Verify SiLU = x*sigmoid(x) (the input/horizon FFN activation).

SiLU is structurally identical to GELU (`x * Phi(x)`), just with a sigmoid table
instead of the Gaussian CDF, so the existing gelu.rs lookup machinery covers it
with no new code. This script confirms the float relationship for completeness.

Usage: .venv-timesfm/bin/python scripts/verify_silu.py
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    for name in ("linear", "val_51", "silu"):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_silu.onnx")
    sess = ort.InferenceSession("/tmp/tsfm_silu.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad = np.zeros((1, 32), dtype=np.float32)
    res = sess.run([o.name for o in m2.graph.output], {
        "input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64),
    })
    names = [o.name for o in m2.graph.output]
    x = res[names.index("linear")].reshape(-1)
    sig = res[names.index("val_51")].reshape(-1)
    silu = res[names.index("silu")].reshape(-1)
    print(f"sigmoid max err = {np.abs(sig - 1/(1+np.exp(-x))).max():.2e}")
    print(f"silu    max err = {np.abs(silu - x*sig).max():.2e}")


if __name__ == "__main__":
    main()
