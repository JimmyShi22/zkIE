#!/usr/bin/env python3
"""Extract the 7 LayerNorm inputs and mean/rstd scalars for the GKR prover.

Walks the LayerNormalization nodes (one per decoder layer), extracts each norm's
input (the residual), weight and bias, computes mean and rstd = 1/sqrt(var+eps),
and dumps everything quantized at scale 2^16 under models/norms/.

Usage: .venv-timesfm/bin/python scripts/extract_all_norms.py
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def dump(path, arr, padn):
    q = np.clip(np.round(arr * SCALE), -(2**31), 2**31 - 1).astype(np.int32)
    p = np.zeros(padn, dtype=np.int32)
    p[: q.size] = q
    p.tofile(path)


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in g.initializer}
    ln_nodes = [n for n in g.node if n.op_type == "LayerNormalization"]

    names = set()
    for n in ln_nodes:
        names.add(n.input[0])
        names.add(n.output[0])
    for name in names:
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_norms.onnx")
    sess = ort.InferenceSession("/tmp/tsfm_norms.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad = np.zeros((1, 32), dtype=np.float32)
    res = sess.run([o.name for o in m2.graph.output], {
        "input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64),
    })
    nmap = {n: i for i, n in enumerate([o.name for o in m2.graph.output])}

    out_dir = "models/norms"
    os.makedirs(out_dir, exist_ok=True)
    for i, n in enumerate(ln_nodes):
        x = res[nmap[n.input[0]]].reshape(-1).astype(np.float64)
        w = inits[n.input[1]]
        b = inits[n.input[2]]
        mean = x.mean()
        var = ((x - mean) ** 2).mean()
        rstd = 1.0 / np.sqrt(var + 1e-6)
        dump(f"{out_dir}/L{i}_in.bin", x, next_pow2(x.size))
        dump(f"{out_dir}/L{i}_w.bin", w, next_pow2(w.size))
        dump(f"{out_dir}/L{i}_b.bin", b, next_pow2(b.size))
        np.array([mean, rstd], dtype=np.float64).tofile(f"{out_dir}/L{i}_scalars_f64.bin")

    print(f"dumped {len(ln_nodes)} LayerNorm inputs/scalars under {out_dir}/")


if __name__ == "__main__":
    main()
