#!/usr/bin/env python3
"""Verify the fixed-point TimesFM 200M forward pass matches the ONNX float."""
import re
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0
H, SEQ, HEADS, HDIM = 1280, 16, 16, 80
EXP_OFF, SILU_OFF = 1 << 21, 1 << 19


def dr(a, b):
    a = np.asarray(a, dtype=np.int64)
    q, r = np.divmod(a, b)
    return np.where(2 * r >= b, q + 1, q).astype(np.int64)


def q(x):
    return np.clip(np.round(x * SCALE), -(2**31), 2**31 - 1).astype(np.int64)


def rs(raw, sh):
    return np.vectorize(lambda v: dr(v, 1 << sh))(raw).astype(np.int64)


def rms_norm(x, w, rsqrt):
    sq = (x**2).sum(axis=-1)
    s = dr(sq, H)
    rstd = rsqrt[dr(s, 1 << 18)][:, None]
    return rs(x * rstd * w, 32)


def layer_norm(x, w, b, rsqrt):
    mean = dr(x.sum(axis=-1), H)
    var = dr(((x - mean[:, None]) ** 2).sum(axis=-1), H)
    rstd = rsqrt[dr(var, 1 << 18)][:, None]
    return rs((x - mean[:, None]) * rstd * w, 32) + b


def attention(x, qkv_w, qkv_b, o_w, o_b, q_scale, mask_q, exp_t):
    qkv = rs(x @ qkv_w, 16) + qkv_b                    # [SEQ, 3H]
    q_, k, v = qkv[:, :H], qkv[:, H:2*H], qkv[:, 2*H:]
    qh = q_.reshape(SEQ, HEADS, HDIM).transpose(1, 0, 2)   # [heads, seq, hdim]
    kh = k.reshape(SEQ, HEADS, HDIM).transpose(1, 2, 0)     # [heads, hdim, seq]
    vh = v.reshape(SEQ, HEADS, HDIM).transpose(1, 0, 2)
    qh = rs(qh * q_scale, 16)
    scores = rs(np.einsum('hqd,hdk->hqk', qh, kh), 16) + mask_q  # [heads, seq, seq]
    c = scores.max(axis=-1, keepdims=True)
    e = exp_t[np.clip(scores - c + EXP_OFF, 0, len(exp_t) - 1)]
    s = e.sum(axis=-1, keepdims=True)
    sm = dr(e * SCALE, s)
    attn = rs(np.einsum('hqk,hkd->hqd', sm, vh), 16)
    attn = attn.transpose(1, 0, 2).reshape(SEQ, H)
    return rs(attn @ o_w, 16) + o_b


def main():
    m = onnx.load("models/timesfm_1_0_200m.onnx")
    g = m.graph
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in g.initializer}
    rsqrt = np.fromfile("models/rsqrt_table_i32.bin", dtype=np.int32).astype(np.int64)
    exp_t = np.fromfile("models/exp_table_i32.bin", dtype=np.int32).astype(np.int64)
    silu_t = np.fromfile("models/silu_table_i32.bin", dtype=np.int32).astype(np.int64)
    q_scale = q(inits["unsqueeze_29"].reshape(HDIM))
    mask_q = q(np.triu(np.full((SEQ, SEQ), -1e30), k=1))

    # build per-layer weight map (node order: qkv, o_proj, gate, down)
    layer_weights = {}
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        w = [x for x in n.input if x in inits and len(inits[x].shape) == 2]
        if not w:
            continue
        out = n.output[0]
        for n2 in g.node:
            if n2.op_type == "Add" and out in n2.input:
                for b in n2.input:
                    mm = re.search(r"layers\.(\d+)\.", b)
                    if mm:
                        layer_weights.setdefault(int(mm.group(1)), []).append(w[0])
                        break

    for name in ("output_ts", "unsqueeze_2", "unsqueeze_4"):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    g.output.append(helper.make_tensor_value_info("add_5", TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/s200.onnx")
    sess = ort.InferenceSession("/tmp/s200.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 512).astype(np.float32)
    pad = np.zeros((1, 512), dtype=np.float32)
    out_ref, u2f, u4f = sess.run(["output_ts", "unsqueeze_2", "unsqueeze_4"],
                                 {"input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64)})
    add5_ref = sess.run(["add_5"], {"input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64)})[0]

    # input embedding (RevIN over first 32)
    ts = inp.reshape(SEQ, 32)
    ref = q(ts[0])
    mean = dr(ref.sum(), 32)
    rstd = int(rsqrt[dr(dr(((ref - mean) ** 2).sum(), 32), 1 << 18)])
    norm = rs((q(ts) - mean) * rstd, 16)
    cat = np.concatenate([norm, np.zeros((SEQ, 32), dtype=np.int64)], axis=1)

    # input FFN
    hid = rs(cat @ q(inits["val_50"]), 16) + q(inits["input_ff_layer.hidden_layer.0.bias"])
    silu = silu_t[np.clip(hid + SILU_OFF, 0, len(silu_t) - 1)]
    o = rs(silu @ q(inits["val_53"]), 16) + q(inits["input_ff_layer.output_layer.bias"])
    r = rs(cat @ q(inits["val_55"]), 16) + q(inits["input_ff_layer.residual_layer.bias"])
    add = o + r

    # freq embedding
    g2 = m.graph
    for name in ("gather", "embedding"):
        g2.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m3 = helper.make_model(g2, opset_imports=m.opset_import)
    onnx.save(m3, "/tmp/s200b.onnx")
    s3 = ort.InferenceSession("/tmp/s200b.onnx", providers=["CPUExecutionProvider"])
    r3 = s3.run(["gather", "embedding"], {"input_ts": inp, "input_padding": pad, "freq": np.array([[0]], dtype=np.int64)})
    x = add + q(r3[0].reshape(SEQ, H)) + q(r3[1].reshape(1, H))

    for L in range(20):
        p = f"stacked_transformer.layers.{L}"
        lnw = q(inits[f"{p}.input_layernorm.weight"])
        residual = x
        x = rms_norm(x, lnw, rsqrt)
        qkv_w, o_w, g_w, d_w = [q(inits[w]) for w in layer_weights[L]]
        qkv_b = q(inits[f"{p}.self_attn.qkv_proj.bias"])
        o_b = q(inits[f"{p}.self_attn.o_proj.bias"])
        g_b = q(inits[f"{p}.mlp.gate_proj.bias"])
        d_b = q(inits[f"{p}.mlp.down_proj.bias"])
        x = residual + attention(x, qkv_w, qkv_b, o_w, o_b, q_scale, mask_q, exp_t)
        if L == 0:
            e0 = np.abs(x.astype(np.float64) / SCALE - add5_ref.reshape(SEQ, -1)).max()
            print(f"layer 0 attention output err = {e0:.3e}")
        mlp_w = q(inits[f"{p}.mlp.layer_norm.weight"])
        mlp_b = q(inits[f"{p}.mlp.layer_norm.bias"])
        ln = layer_norm(x, mlp_w, mlp_b, rsqrt)
        gate = rs(ln @ g_w, 16) + g_b
        relu = np.maximum(gate, 0)
        x = x + rs(relu @ d_w, 16) + d_b

    # output head (horizon FFN): hidden -> SiLU -> output + residual -> rescale
    hid = rs(x @ q(inits["val_837"]), 16) + q(inits["horizon_ff_layer.hidden_layer.0.bias"])
    silu = silu_t[np.clip(hid + SILU_OFF, 0, len(silu_t) - 1)]
    o = rs(silu @ q(inits["val_840"]), 16) + q(inits["horizon_ff_layer.output_layer.bias"])
    r = rs(x @ q(inits["val_842"]), 16) + q(inits["horizon_ff_layer.residual_layer.bias"])
    add31 = o + r
    # rescale: out = add31 * u4 + u2 (scalars)
    u4 = q(np.array([float(np.asarray(u4f).reshape(-1)[0])]))
    u2 = q(np.array([float(np.asarray(u2f).reshape(-1)[0])]))
    out = rs(add31 * u4, 16) + u2
    err = np.abs(out.astype(np.float64) / SCALE - out_ref.reshape(SEQ, -1)).max()
    print(f"200M full forward pass: final output max abs err = {err:.3e}")


if __name__ == "__main__":
    main()
