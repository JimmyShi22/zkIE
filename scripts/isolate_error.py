#!/usr/bin/env python3
"""Isolate the fixed-point error source: attention vs FFN vs norms."""
import re

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0
H, SEQ, HEADS, HDIM = 1280, 16, 16, 80
EXP_OFF, SILU_OFF = 1 << 21, 1 << 19


def q(x):
    return np.clip(np.round(x * SCALE), -(2**31), 2**31 - 1).astype(np.int64)


def dr(a, b):
    a = np.asarray(a, dtype=np.int64)
    qq, r = np.divmod(a, b)
    return np.where(2 * r >= b, qq + 1, qq).astype(np.int64)


def rs(raw, sh):
    return dr(raw, 1 << sh).astype(np.int64)


def rms_norm_q(x, w, rsqrt):
    sq = (x**2).sum(axis=-1)
    s = dr(sq, H)
    rstd = rsqrt[dr(s, 1 << 18)][:, None]
    return rs(x * rstd * w, 32)


def layer_norm_q(x, w, b, rsqrt):
    mean = dr(x.sum(axis=-1), H)
    var = dr(((x - mean[:, None]) ** 2).sum(axis=-1), H)
    rstd = rsqrt[dr(var, 1 << 18)][:, None]
    return rs((x - mean[:, None]) * rstd * w, 32) + b


def rms_norm_f(x, w):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) / H + 1e-6) * w


def layer_norm_f(x, w, b):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-6) * w + b


def attention_q(x, qkv_w, qkv_b, o_w, o_b, q_scale, mask_q, exp_t):
    qkv = rs(x @ qkv_w, 16) + qkv_b
    q_, k, v = qkv[:, :H], qkv[:, H:2 * H], qkv[:, 2 * H:]
    qh = q_.reshape(SEQ, HEADS, HDIM).transpose(1, 0, 2)
    kh = k.reshape(SEQ, HEADS, HDIM).transpose(1, 2, 0)
    vh = v.reshape(SEQ, HEADS, HDIM).transpose(1, 0, 2)
    qh = rs(qh * q_scale, 16)
    scores = rs(np.einsum('hqd,hdk->hqk', qh, kh), 16) + mask_q
    c = scores.max(axis=-1, keepdims=True)
    e = exp_t[np.clip(scores - c + EXP_OFF, 0, len(exp_t) - 1)]
    s = e.sum(axis=-1, keepdims=True)
    sm = dr(e * SCALE, s)
    attn = rs(np.einsum('hqk,hkd->hqd', sm, vh), 16)
    attn = attn.transpose(1, 0, 2).reshape(SEQ, H)
    return rs(attn @ o_w, 16) + o_b


def attention_f(x, qkv_wf, qkv_bf, o_wf, o_bf, q_scalef, mask_f):
    qkv = x @ qkv_wf + qkv_bf
    q_, k, v = qkv[:, :H], qkv[:, H:2 * H], qkv[:, 2 * H:]
    qh = q_.reshape(SEQ, HEADS, HDIM).transpose(1, 0, 2)
    kh = k.reshape(SEQ, HEADS, HDIM).transpose(1, 2, 0)
    vh = v.reshape(SEQ, HEADS, HDIM).transpose(1, 0, 2)
    qh = qh * q_scalef[None, None, :]
    scores = np.einsum('hqd,hdk->hqk', qh, kh) + mask_f
    sm = np.exp(scores - scores.max(-1, keepdims=True))
    sm = sm / sm.sum(-1, keepdims=True)
    attn = np.einsum('hqk,hkd->hqd', sm, vh).transpose(1, 0, 2).reshape(SEQ, H)
    return attn @ o_wf + o_bf


def main():
    base = "models"
    m = onnx.load(f"{base}/timesfm_1_0_200m.onnx")
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
    rsqrt = np.fromfile(f"{base}/rsqrt_table_i32.bin", dtype=np.int32).astype(np.int64)
    exp_t = np.fromfile(f"{base}/exp_table_i32.bin", dtype=np.int32).astype(np.int64)
    silu_t = np.fromfile(f"{base}/silu_table_i32.bin", dtype=np.int32).astype(np.int64)
    q_scale = q(inits["unsqueeze_29"].reshape(HDIM))
    q_scale_f = inits["unsqueeze_29"].reshape(HDIM)
    mask_q = q(np.triu(np.full((SEQ, SEQ), -1e30), k=1))
    mask_f = np.triu(np.full((SEQ, SEQ), -1e30), k=1)

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

    inp = np.random.RandomState(0).randn(1, 512).astype(np.float32)
    pad = np.zeros((1, 512), dtype=np.float32)
    freq = np.array([[0]], dtype=np.int64)

    # float reference output
    g2 = m.graph
    for name in ("output_ts", "unsqueeze_2", "unsqueeze_4"):
        g2.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m4 = helper.make_model(g2, opset_imports=m.opset_import)
    onnx.save(m4, "/tmp/ref.onnx")
    s4 = ort.InferenceSession("/tmp/ref.onnx", providers=["CPUExecutionProvider"])
    ref = s4.run(["output_ts"], {"input_ts": inp, "input_padding": pad, "freq": freq})[0].reshape(SEQ, -1)
    u2 = float(np.asarray(s4.run(["unsqueeze_2"], {"input_ts": inp, "input_padding": pad, "freq": freq})[0]).reshape(-1)[0])
    u4 = float(np.asarray(s4.run(["unsqueeze_4"], {"input_ts": inp, "input_padding": pad, "freq": freq})[0]).reshape(-1)[0])

    # prologue (fixed)
    ts = inp.reshape(SEQ, 32)
    refq = q(ts[0])
    mean = dr(refq.sum(), 32)
    rstd = int(rsqrt[dr(dr(((refq - mean) ** 2).sum(), 32), 1 << 18)])
    norm = rs((q(ts) - mean) * rstd, 16)
    cat = np.concatenate([norm, np.zeros((SEQ, 32), dtype=np.int64)], axis=1)
    hid = rs(cat @ q(inits["val_50"]), 16) + q(inits["input_ff_layer.hidden_layer.0.bias"])
    silu = silu_t[np.clip(hid + SILU_OFF, 0, len(silu_t) - 1)]
    o = rs(silu @ q(inits["val_53"]), 16) + q(inits["input_ff_layer.output_layer.bias"])
    rr = rs(cat @ q(inits["val_55"]), 16) + q(inits["input_ff_layer.residual_layer.bias"])
    add = o + rr
    g3 = m.graph
    for name in ("gather", "embedding"):
        g3.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m5 = helper.make_model(g3, opset_imports=m.opset_import)
    onnx.save(m5, "/tmp/freq.onnx")
    s5 = ort.InferenceSession("/tmp/freq.onnx", providers=["CPUExecutionProvider"])
    r3 = s5.run(["gather", "embedding"], {"input_ts": inp, "input_padding": pad, "freq": freq})
    x0 = add + q(r3[0].reshape(SEQ, H)) + q(r3[1].reshape(1, H))

    # Float x0 from the ONNX reference (the add_2 residual).
    g4 = m.graph
    for name in ("add_2",):
        g4.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m6 = helper.make_model(g4, opset_imports=m.opset_import)
    onnx.save(m6, "/tmp/x0.onnx")
    s6 = ort.InferenceSession("/tmp/x0.onnx", providers=["CPUExecutionProvider"])
    x0_float = s6.run(["add_2"], {"input_ts": inp, "input_padding": pad, "freq": freq})[0].reshape(SEQ, H)
    x0_float_q = q(x0_float)

    def run(attn_q, ffn_q, norm_q=True, float_prologue=False, float_head=False):
        x = x0.copy()
        if float_prologue:
            x = x0_float_q.copy()
        for L in range(20):
            p = f"stacked_transformer.layers.{L}"
            residual = x
            if norm_q:
                lnw = q(inits[f"{p}.input_layernorm.weight"])
                xn = rms_norm_q(x, lnw, rsqrt)
            else:
                lnw = inits[f"{p}.input_layernorm.weight"]
                xn = rms_norm_f(x.astype(np.float64) / SCALE, lnw)
                xn = q(xn)
            qkv_w, o_w, g_w, d_w = [q(inits[w]) for w in layer_weights[L]]
            qkv_b = q(inits[f"{p}.self_attn.qkv_proj.bias"])
            o_b = q(inits[f"{p}.self_attn.o_proj.bias"])
            g_b = q(inits[f"{p}.mlp.gate_proj.bias"])
            d_b = q(inits[f"{p}.mlp.down_proj.bias"])
            if attn_q:
                att = attention_q(xn, qkv_w, qkv_b, o_w, o_b, q_scale, mask_q, exp_t)
            else:
                att = attention_f(xn.astype(np.float64) / SCALE,
                                  inits[layer_weights[L][0]], inits[f"{p}.self_attn.qkv_proj.bias"],
                                  inits[layer_weights[L][1]], inits[f"{p}.self_attn.o_proj.bias"],
                                  q_scale_f, mask_f)
                att = q(att)
            x = residual + att
            if ffn_q:
                mlp_w = q(inits[f"{p}.mlp.layer_norm.weight"])
                mlp_b = q(inits[f"{p}.mlp.layer_norm.bias"])
                ln = layer_norm_q(x, mlp_w, mlp_b, rsqrt)
                gate = rs(ln @ g_w, 16) + g_b
                relu = np.maximum(gate, 0)
                x = x + rs(relu @ d_w, 16) + d_b
            else:
                mlp_w = inits[f"{p}.mlp.layer_norm.weight"]
                mlp_b = inits[f"{p}.mlp.layer_norm.bias"]
                ln = layer_norm_f(x.astype(np.float64) / SCALE, mlp_w, mlp_b)
                gate = ln @ inits[layer_weights[L][2]] + inits[f"{p}.mlp.gate_proj.bias"]
                relu = np.maximum(gate, 0)
                out = (relu @ inits[layer_weights[L][3]] + inits[f"{p}.mlp.down_proj.bias"])
                x = x + q(out)
        # head
        if float_head:
            xf = x.astype(np.float64) / SCALE
            hidf = xf @ inits["val_837"] + inits["horizon_ff_layer.hidden_layer.0.bias"]
            siluf = hidf / (1 + np.exp(-hidf))
            of = siluf @ inits["val_840"] + inits["horizon_ff_layer.output_layer.bias"]
            rf = xf @ inits["val_842"] + inits["horizon_ff_layer.residual_layer.bias"]
            out = (of + rf) * u4 + u2
            return np.abs(out - ref).max()
        hid = rs(x @ q(inits["val_837"]), 16) + q(inits["horizon_ff_layer.hidden_layer.0.bias"])
        silu = silu_t[np.clip(hid + SILU_OFF, 0, len(silu_t) - 1)]
        o = rs(silu @ q(inits["val_840"]), 16) + q(inits["horizon_ff_layer.output_layer.bias"])
        rr = rs(x @ q(inits["val_842"]), 16) + q(inits["horizon_ff_layer.residual_layer.bias"])
        add31 = o + rr
        out = rs(add31 * q(np.array([u4])), 16) + q(np.array([u2]))
        return np.abs(out / SCALE - ref).max()

    print("full fixed  err =", run(True, True))
    print("float attn  err =", run(False, True))
    print("float ffn   err =", run(True, False))
    print("float rms   err =", run(True, True, norm_q=False))
    print("float prologue err =", run(True, True, norm_q=True, float_prologue=True))
    print("float head  err =", run(True, True, norm_q=True, float_head=True))
    print("float all   err =", run(False, False, norm_q=False, float_prologue=True, float_head=True))


if __name__ == "__main__":
    main()
