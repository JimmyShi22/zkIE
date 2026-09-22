#!/usr/bin/env python3
"""Verify the int32 quantization for every weight-matmuls in TimesFM.

Walks all MatMul nodes with a 2D weight, quantizes the inputs at scale 2^16,
recomputes the matmul, rescales, and reports the absolute error (in quantized
LSB units) against the real output. Absolute error is the meaningful metric:
several later-layer gates and the horizon FFN have near-zero weights, so their
relative error is misleadingly large while their absolute error is ~1 LSB.

Usage: .venv-timesfm/bin/python scripts/verify_full_quantization.py
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

    matmuls = []
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        ws = [i for i in n.input if i in inits and len(inits[i].shape) == 2]
        if ws:
            act = [i for i in n.input if i not in inits][0]
            matmuls.append((n.output[0], ws[0], act))

    names = set()
    for out, _, act in matmuls:
        names.add(out)
        names.add(act)
    for name in names:
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_verify.onnx")

    sess = ort.InferenceSession("/tmp/tsfm_verify.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad = np.zeros((1, 32), dtype=np.float32)
    res = sess.run([o.name for o in m2.graph.output], {
        "input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64),
    })
    nmap = {n: i for i, n in enumerate([o.name for o in m2.graph.output])}

    errs = []
    for out, wname, act in matmuls:
        A = res[nmap[act]].reshape(-1)
        O = res[nmap[out]].reshape(-1)
        W = inits[wname]
        qA = np.round(A * SCALE).astype(np.int64)
        qW = np.round(W * SCALE).astype(np.int64)
        qO = np.round(O * SCALE).astype(np.int64)
        recon = np.round((qA @ qW) / SCALE).astype(np.int64)
        errs.append((int(np.abs(recon - qO).max()), out))

    errs.sort(reverse=True)
    print(f"{'output':12} {'max abs err (LSB)':>18}")
    for e, out in errs[:8]:
        print(f"{out:12} {e:>18}")
    print(f"worst absolute error: {errs[0][0]} LSB (~{errs[0][0]/SCALE:.2e} float)")
    print(f"matmuls under 20 LSB: {sum(1 for e, _ in errs if e < 20)} of {len(errs)}")


if __name__ == "__main__":
    main()
