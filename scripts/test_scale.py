#!/usr/bin/env python3
"""Measure final-output fidelity at several fixed-point scales to pick the
smallest scale that brings the error to an acceptable level."""
import re

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

H, SEQ, HEADS, HDIM = 1280, 16, 16, 80


def dr(a, b):
    a = np.asarray(a, dtype=np.int64)
    qq, r = np.divmod(a, b)
    return np.where(2 * r >= b, qq + 1, qq).astype(np.int64)


def build_exp(s):
    # x in [-32, 0] at scale 2^s, exp value at scale 2^s.
    n = 32 << s
    idx = np.arange(n)
    x = (idx - n) / (1 << s)
    return np.clip(np.round(np.exp(x) * (1 << s)), -(2**31), 2**31 - 1).astype(np.int32), n


def build_silu(s):
    # x in [-8, 8] at scale 2^s.
    n = 8 << s
    idx = np.arange(2 * n)
    x = (idx - n) / (1 << s)
    return np.clip(np.round((x / (1 + np.exp(-x))) * (1 << s)), -(2**31), 2**31 - 1).astype(np.int32), n


def build_rsqrt(s, index_bits):
    # index at scale 2^index_bits, value at scale 2^s.
    n = 32 << index_bits
    idx = np.arange(n)
    v = idx / (1 << index_bits) + 1e-6
    return np.clip(np.round((1 / np.sqrt(v)) * (1 << s)), -(2**31), 2**31 - 1).astype(np.int32)


def run(s, index_bits=14):
    SCALE = 1 << s
    def q(x):
        return np.clip(np.round(x * SCALE), -(2**31), 2**31 - 1).astype(np.int64)
    def rs(raw, sh):
        return dr(raw, 1 << sh).astype(np.int64)

    exp_t, exp_off = build_exp(s)
    silu_t, silu_off = build_silu(s)
    rsqrt = build_rsqrt(s, index_bits)

    base = "models"
    m = onnx.load(f"{base}/timesfm_1_0_200m.onnx")
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
    q_scale = q(inits["unsqueeze_29"].reshape(HDIM))
    mask_q = q(np.triu(np.full((SEQ, SEQ), -1e30), k=1))

    layer_weights = {}
    for n in m.graph.node:
        if n.op_type != "MatMul":
            continue
        w = [x for x in n.input if x in inits and len(inits[x].shape) == 2]
        if not w:
            continue
        out = n.output[0]
        for n2 in m.graph.node:
            if n2.op_type == "Add" and out in n2.input:
                for b in n2.input:
                    mm = re.search(r"layers\.(\d+)\.", b)
                    if mm:
                        layer_weights.setdefault(int(mm.group(1)), []).append(w[0])
                        break

    def rms_norm(x, w):
        sq = (x**2).sum(axis=-1)
        s2 = dr(sq, H)
        rstd = rsqrt[dr(s2, 1 << (2 * s - index_bits))][:, None]
        return rs(x * rstd * w, 2 * s)

    def layer_norm(x, w, b):
        mean = dr(x.sum(axis=-1), H)
        var = dr(((x - mean[:, None]) ** 2).sum(axis=-1), H)
        rstd = rsqrt[dr(var, 1 << (2 * s - index_bits))][:, None]
        return rs((x - mean[:, None]) * rstd * w, 2 * s) + b

    def attention(x, qkv_w, qkv_b, o_w, o_b):
        qkv = rs(x @ qkv_w, s) + qkv_b
        q_, k, v = qkv[:, :H], qkv[:, H:2 * H], qkv[:, 2 * H:]
        qh = q_.reshape(SEQ, HEADS, HDIM).transpose(1, 0, 2)
        kh = k.reshape(SEQ, HEADS, HDIM).transpose(1, 2, 0)
        vh = v.reshape(SEQ, HEADS, HDIM).transpose(1, 0, 2)
        qh = rs(qh * q_scale, s)
        scores = rs(np.einsum('hqd,hdk->hqk', qh, kh), s) + mask_q
        c = scores.max(axis=-1, keepdims=True)
        e = exp_t[np.clip(scores - c + exp_off, 0, len(exp_t) - 1)]
        sm = dr(e.astype(np.int64) * SCALE, e.sum(axis=-1, keepdims=True).astype(np.int64))
        attn = rs(np.einsum('hqk,hkd->hqd', sm, vh), s)
        attn = attn.transpose(1, 0, 2).reshape(SEQ, H)
        return rs(attn @ o_w, s) + o_b

    inp = np.random.RandomState(0).randn(1, 512).astype(np.float32)
    pad = np.zeros((1, 512), dtype=np.float32)
    freq = np.array([[0]], dtype=np.int64)
    g = m.graph
    for name in ("output_ts", "unsqueeze_2", "unsqueeze_4"):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m4 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m4, "/tmp/ref_scale.onnx")
    s4 = ort.InferenceSession("/tmp/ref_scale.onnx", providers=["CPUExecutionProvider"])
    ref = s4.run(["output_ts"], {"input_ts": inp, "input_padding": pad, "freq": freq})[0].reshape(SEQ, -1)
    u2 = float(np.asarray(s4.run(["unsqueeze_2"], {"input_ts": inp, "input_padding": pad, "freq": freq})[0]).reshape(-1)[0])
    u4 = float(np.asarray(s4.run(["unsqueeze_4"], {"input_ts": inp, "input_padding": pad, "freq": freq})[0]).reshape(-1)[0])

    ts = inp.reshape(SEQ, 32)
    refq = q(ts[0])
    mean = dr(refq.sum(), 32)
    rstd = int(rsqrt[dr(dr(((refq - mean) ** 2).sum(), 32), 1 << (2 * s - index_bits))])
    norm = rs((q(ts) - mean) * rstd, s)
    cat = np.concatenate([norm, np.zeros((SEQ, 32), dtype=np.int64)], axis=1)
    hid = rs(cat @ q(inits["val_50"]), s) + q(inits["input_ff_layer.hidden_layer.0.bias"])
    silu = silu_t[np.clip(hid + silu_off, 0, len(silu_t) - 1)]
    o = rs(silu @ q(inits["val_53"]), s) + q(inits["input_ff_layer.output_layer.bias"])
    rr = rs(cat @ q(inits["val_55"]), s) + q(inits["input_ff_layer.residual_layer.bias"])
    add = o + rr
    g3 = m.graph
    for name in ("gather", "embedding"):
        g3.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m5 = helper.make_model(g3, opset_imports=m.opset_import)
    onnx.save(m5, "/tmp/freq_scale.onnx")
    s5 = ort.InferenceSession("/tmp/freq_scale.onnx", providers=["CPUExecutionProvider"])
    r3 = s5.run(["gather", "embedding"], {"input_ts": inp, "input_padding": pad, "freq": freq})
    x = add + q(r3[0].reshape(SEQ, H)) + q(r3[1].reshape(1, H))

    for L in range(20):
        p = f"stacked_transformer.layers.{L}"
        lnw = q(inits[f"{p}.input_layernorm.weight"])
        residual = x
        x = rms_norm(x, lnw)
        qkv_w, o_w, g_w, d_w = [q(inits[w]) for w in layer_weights[L]]
        qkv_b = q(inits[f"{p}.self_attn.qkv_proj.bias"])
        o_b = q(inits[f"{p}.self_attn.o_proj.bias"])
        g_b = q(inits[f"{p}.mlp.gate_proj.bias"])
        d_b = q(inits[f"{p}.mlp.down_proj.bias"])
        x = residual + attention(x, qkv_w, qkv_b, o_w, o_b)
        mlp_w = q(inits[f"{p}.mlp.layer_norm.weight"])
        mlp_b = q(inits[f"{p}.mlp.layer_norm.bias"])
        ln = layer_norm(x, mlp_w, mlp_b)
        gate = rs(ln @ g_w, s) + g_b
        relu = np.maximum(gate, 0)
        x = x + rs(relu @ d_w, s) + d_b
        if L == 0:
            print(f"  [s={s}] L0 x residual magnitude:", int(np.abs(x).max()), "lsb")

    hid = rs(x @ q(inits["val_837"]), s) + q(inits["horizon_ff_layer.hidden_layer.0.bias"])
    silu = silu_t[np.clip(hid + silu_off, 0, len(silu_t) - 1)]
    o = rs(silu @ q(inits["val_840"]), s) + q(inits["horizon_ff_layer.output_layer.bias"])
    rr = rs(x @ q(inits["val_842"]), s) + q(inits["horizon_ff_layer.residual_layer.bias"])
    add31 = o + rr
    out = rs(add31 * q(np.array([u4])), s) + q(np.array([u2]))
    return np.abs(out / SCALE - ref).max()


if __name__ == "__main__":
    for s, ib in ((16, 14), (18, 14), (18, 16), (18, 18)):
        err = run(s, ib)
        print(f"scale 2^{s} (rsqrt index 2^{ib}): final max err = {err:.4e}")
