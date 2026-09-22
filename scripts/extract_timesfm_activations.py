#!/usr/bin/env python3
"""Extract TimesFM layer-0 activations from one forward pass.

Adds the key intermediate tensors as ONNX outputs, runs one forward pass,
reports each tensor's float range, and dumps them quantized at scale 2^16 into
int32 (padded to a power of two). The full activation range spans ~[-4.3, 4.3],
so int16 cannot cover it at fine scale; int32 at 2^16 keeps both the tiny inputs
and the large ReLU output precise.
Usage: .venv-timesfm/bin/python scripts/extract_timesfm_activations.py
"""

import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    key = ["mul_9", "val_95", "matmul", "softmax", "matmul_1", "view_5", "val_123", "layer_norm", "val_127", "relu", "val_129"]
    for name in key:
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_all_acts.onnx")

    sess = ort.InferenceSession("/tmp/tsfm_all_acts.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad = np.zeros((1, 32), dtype=np.float32)
    outs = sess.run([o.name for o in m2.graph.output], {
        "input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64),
    })

    names = [o.name for o in m2.graph.output]
    import os
    os.makedirs("models/activations", exist_ok=True)
    SCALE = 65536.0  # 2^16
    print(f"{'tensor':12} {'shape':18} {'min':>10} {'max':>10}")
    for name in key:
        if name in names:
            v = outs[names.index(name)]
            q = np.clip(np.round(v.reshape(-1) * SCALE), -(2**31), 2**31 - 1).astype(np.int32)
            n = q.size
            pn = 1 << (n - 1).bit_length()
            pad = np.zeros(pn, dtype=np.int32)
            pad[:n] = q
            pad.tofile(f"models/activations/{name}_{pn}_i32.bin")
            print(f"{name:12} {str(v.shape):18} {float(v.min()):>10.4f} {float(v.max()):>10.4f}")


if __name__ == "__main__":
    main()
