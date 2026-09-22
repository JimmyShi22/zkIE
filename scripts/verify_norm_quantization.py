#!/usr/bin/env python3
"""Verify the quantized LayerNorm (mlp.layer_norm) matches the real output.

The only non-arithmetic piece of standard LayerNorm is rstd = 1/sqrt(var+eps), a
single-scalar lookup. This recomputes the normalization at scale 2^16 and reports
the error against the extracted activation, confirming the norm's fixed-point
arithmetic is sound.

Usage: .venv-timesfm/bin/python scripts/verify_norm_quantization.py
"""
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto

SCALE = 65536.0


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in g.initializer}
    ln_w = inits["stacked_transformer.layers.0.mlp.layer_norm.weight"]
    ln_b = inits["stacked_transformer.layers.0.mlp.layer_norm.bias"]
    for name in ("add_5", "layer_norm"):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_norm.onnx")

    sess = ort.InferenceSession("/tmp/tsfm_norm.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad = np.zeros((1, 32), dtype=np.float32)
    res = sess.run([o.name for o in m2.graph.output], {
        "input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64),
    })
    names = [o.name for o in m2.graph.output]
    x = res[names.index("add_5")].reshape(-1)
    y = res[names.index("layer_norm")].reshape(-1)

    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    rstd = 1.0 / np.sqrt(var + 1e-6)
    q_x = np.round(x * SCALE).astype(np.int64)
    q_mean = np.round(mean * SCALE).astype(np.int64)
    q_rstd = np.round(rstd * SCALE).astype(np.int64)
    q_w = np.round(ln_w * SCALE).astype(np.int64)
    q_b = np.round(ln_b * SCALE).astype(np.int64)
    q_y = np.round(y * SCALE).astype(np.int64)
    recon = np.round((q_x - q_mean) * q_rstd // SCALE * q_w // SCALE + q_b).astype(np.int64)
    d = np.abs(recon - q_y)
    print(f"max abs diff = {d.max()} LSB, mean = {float(d.mean()):.3f}, rel = {float(d.mean()/(np.abs(q_y).mean()+1e-6)):.4%}")


if __name__ == "__main__":
    main()
