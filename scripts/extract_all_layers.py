#!/usr/bin/env python3
"""Extract the 7 transformer layers' matmul activations for the GKR prover.

Walks the ONNX MatMul nodes, groups them into the 7 decoder layers by the four
2D weight shapes (qkv 264x792, o_proj 264x264, gate 264x1024, down 1024x264),
and dumps each layer's activation inputs/outputs quantized at scale 2^16 into
padded int32 under models/layers/. Attention QK^T/PV are activation-matmuls
(degenerate at seq=1), so only the four weight matmuls are materialized here.

Usage: .venv-timesfm/bin/python scripts/extract_all_layers.py
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    inits = {i.name: [d for d in i.dims] for i in g.initializer}

    # Collect the four weight-matmuls per layer, in layer order.
    layers = []
    started = False
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        ws = [i for i in n.input if i in inits and len(inits[i]) == 2]
        if not ws:
            continue
        w = ws[0]
        act = [i for i in n.input if i not in inits][0]
        shape = tuple(inits[w])
        if shape == (264, 792):  # QKV starts a new layer
            started = True
            layers.append({"qkv": (act, n.output[0], w)})
        elif not started:
            continue  # input/horizon FFN matmuls before the first transformer layer
        elif shape == (264, 264):
            layers[-1]["o_proj"] = (act, n.output[0], w)
        elif shape == (264, 1024):
            layers[-1]["gate"] = (act, n.output[0], w)
        elif shape == (1024, 264):
            layers[-1]["down"] = (act, n.output[0], w)

    assert len(layers) == 7, f"expected 7 layers, got {len(layers)}"

    # Add every activation/output as an output and run once.
    names = set()
    for L in layers:
        for key in ("qkv", "o_proj", "gate", "down"):
            act, out, _ = L[key]
            names.add(act)
            names.add(out)
    for name in names:
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_layers.onnx")

    sess = ort.InferenceSession("/tmp/tsfm_layers.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad = np.zeros((1, 32), dtype=np.float32)
    res = sess.run([o.name for o in m2.graph.output], {
        "input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64),
    })
    nmap = {n: i for i, n in enumerate([o.name for o in m2.graph.output])}

    out_dir = "models/layers"
    os.makedirs(out_dir, exist_ok=True)
    for i, L in enumerate(layers):
        for key in ("qkv", "o_proj", "gate", "down"):
            act, out, _ = L[key]
            for tag, name in (("in", act), ("out", out)):
                v = res[nmap[name]].reshape(-1).astype(np.float64)
                q = np.clip(np.round(v * SCALE), -(2**31), 2**31 - 1).astype(np.int32)
                pn = next_pow2(q.size)
                p = np.zeros(pn, dtype=np.int32)
                p[: q.size] = q
                p.tofile(os.path.join(out_dir, f"L{i}_{key}_{tag}_{pn}_i32.bin"))

    print(f"dumped {len(layers)} layers under {out_dir}/")


if __name__ == "__main__":
    main()
