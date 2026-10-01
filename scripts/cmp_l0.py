#!/usr/bin/env python3
"""Compare fp vs float for layer 0 attention + FFN to find the divergence."""
import numpy as np
from extract_deepseek import Sharded

DS = "models/deepseek-v2-lite/weights"
H = 2048
HEADS = 16
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
KV_LORA = 512
DENSE_PAD = 16384


def li(name):
    return np.fromfile("%s/%s" % (DS, name), dtype="<i4").astype(np.float64)


def lgp(name):
    return np.fromfile("models/gpt2/weights/%s" % name, dtype="<i4").astype(np.float64)


def rd(a, b):
    q = np.floor(a / b)
    r = a - q * b
    return q + (r * 2 >= b)


def main():
    S = Sharded()
    # float weights
    p = "model.layers.0."
    q_w = S.arr(p + "self_attn.q_proj.weight").astype(np.float64)
    kva_w = S.arr(p + "self_attn.kv_a_proj_with_mqa.weight").astype(np.float64)
    kva_ln = S.arr(p + "self_attn.kv_a_layernorm.weight").astype(np.float64)
    kvb_w = S.arr(p + "self_attn.kv_b_proj.weight").astype(np.float64)
    o_w = S.arr(p + "self_attn.o_proj.weight").astype(np.float64)
    in_norm = S.arr(p + "input_layernorm.weight").astype(np.float64)
    emb = S.arr("model.embed_tokens.weight").astype(np.float64)
    rng = np.random.default_rng(0)
    ids = rng.integers(1, 102400, (16,)).astype(np.int64)

    # float forward (x0 + layer 0)
    xf = emb[ids]
    hf = xf * (1 / np.sqrt((xf * xf).mean(-1, keepdims=True) + 1e-6)) * in_norm
    q = (hf @ q_w.T).reshape(16, HEADS, QK_NOPE + QK_ROPE)
    q_nope, q_pe = q[..., :QK_NOPE], q[..., QK_NOPE:]
    ckv = hf @ kva_w.T
    compressed, k_pe = ckv[..., :KV_LORA], ckv[..., KV_LORA:KV_LORA + QK_ROPE]
    kv = (kvb_w @ (compressed * (1 / np.sqrt((compressed * compressed).mean(-1, keepdims=True) + 1e-6)) * kva_ln).T).T
    kv = kv.reshape(16, HEADS, QK_NOPE + V_HEAD)
    k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
    pos = np.arange(16)[:, None]
    inv = np.arange(QK_ROPE // 2)[None, :]
    theta = pos * np.power(10000.0, -2 * inv / QK_ROPE)
    cos, sin = np.cos(theta), np.sin(theta)
    def rope(x):
        h = x.shape[-1] // 2
        return np.concatenate([x[..., :h] * cos[:, None, :] - x[..., h:] * sin[:, None, :],
                               x[..., h:] * cos[:, None, :] + x[..., :h] * sin[:, None, :]], -1)
    q_pe = rope(q_pe)
    k_pe = rope(k_pe[:, None, :])
    query = np.concatenate([q_nope, q_pe], -1)
    key = np.concatenate([k_nope, np.broadcast_to(k_pe, (16, HEADS, QK_ROPE))], -1)
    mscale = 0.1 * 0.707 * np.log(40.0) + 1.0
    ss = 192.0 ** -0.5 * mscale * mscale
    scores = np.einsum("ihd,jhd->ihj", query, key) * ss
    mask = np.triu(np.full((16, 16), -1e30), 1)
    probs = np.exp(scores + mask[:, None, :] - (scores + mask[:, None, :]).max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    attn_float = (probs @ v).reshape(16, HEADS * V_HEAD) @ o_w.T

    # fp forward (same, in Q16)
    rsqrt_t = lgp("rsqrt_table_i32.bin")
    exp_t = lgp("exp_table_i32.bin")
    silu_t = li("silu_table_i32.bin")
    x = li("x0_i32.bin").reshape(16, H)
    def rms(x, w):
        ms = rd((x * x).sum(-1), x.shape[-1])
        rstd = np.round((1.0 / np.sqrt(ms / (1 << 32) + 1e-6)) * (1 << 16))
        return rd(x * rstd[:, None] * w[None, :], 1 << 32)
    def proj(x, W):
        return rd(x @ W, 1 << 16)
    h = rms(x, li("L0_in_norm_i32.bin"))
    qnope_w = li("L0_qnope_w_i32.bin").reshape(H, HEADS * QK_NOPE)
    qpe_w = li("L0_qpe_w_i32.bin").reshape(H, HEADS * QK_ROPE)
    kvlora_w = li("L0_kvlora_w_i32.bin").reshape(H, KV_LORA)
    kpe_w = li("L0_kpe_w_i32.bin").reshape(H, QK_ROPE)
    knope_w = li("L0_knope_w_i32.bin").reshape(KV_LORA, HEADS * QK_NOPE)
    v_w = li("L0_v_w_i32.bin").reshape(KV_LORA, HEADS * V_HEAD)
    o_w2 = li("L0_o_w_i32.bin").reshape(HEADS * V_HEAD, H)
    kvlora = proj(h, kvlora_w)
    kpe = proj(h, kpe_w)
    kvlora_n = rms(kvlora, li("L0_kva_ln_i32.bin"))
    cosf = li("rope_cos_i32.bin").reshape(16, QK_ROPE // 2)
    sinf = li("rope_sin_i32.bin").reshape(16, QK_ROPE // 2)
    def rope_fp(x):
        hh = x.shape[-1] // 2
        return np.concatenate([rd(x[..., :hh] * cosf - x[..., hh:] * sinf, 1 << 16),
                               rd(x[..., hh:] * cosf + x[..., :hh] * sinf, 1 << 16)], -1)
    kpe_rot = rope_fp(kpe)
    sf = round(ss * (1 << 16))
    head_outs = []
    for hd in range(HEADS):
        qn = proj(h, qnope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
        qp = proj(h, qpe_w[:, hd * QK_ROPE:(hd + 1) * QK_ROPE])
        kn = proj(kvlora_n, knope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
        vh = proj(kvlora_n, v_w[:, hd * V_HEAD:(hd + 1) * V_HEAD])
        qp_rot = rope_fp(qp)
        s = qn @ kn.T + qp_rot @ kpe_rot.T
        s16 = rd(s * sf, 1 << 32)
        masked = s16 + np.triu(np.full((16, 16), -(1 << 30)), 1)
        rm = masked.max(-1, keepdims=True)
        shifted = np.clip((masked - rm) + (1 << 21), 0, (1 << 21) - 1).astype(np.int64)
        ev = exp_t[shifted]
        pr = rd(ev * (1 << 16), ev.sum(-1, keepdims=True))
        a16 = rd(pr @ vh, 1 << 16)
        head_outs.append(proj(a16, o_w2[hd * V_HEAD:(hd + 1) * V_HEAD]))
    attn_fp = sum(head_outs)

    # compare
    af = attn_float.astype(np.float64)
    ap = attn_fp / (1 << 16)
    diff = np.abs(af - ap)
    rel = diff / (np.abs(af) + 1e-6)
    print("float score[5,0,:6]=", scores[5,0,:6])
    # fp score row5 head0
    qn5=proj(h, qnope_w[:, 0:128]); qp5=proj(h, qpe_w[:, 0:64]); kn5=proj(kvlora_n, knope_w[:, 0:128]); qp5r=rope_fp(qp5); s5=(qn5@kn5.T + qp5r@kpe_rot.T)[5,:6]; print("fp score[5,:6]/65536=", rd(s5*sf,1<<32)/65536)
    print("float attn max=%.4f  fp attn max=%.4f" % (np.abs(af).max(), np.abs(ap).max()))
    print("max abs diff=%.4f  max rel diff=%.4f" % (diff.max(), rel.max()))
    # granular: v (head 0) and probs (row 0 head 0)
    vf = v[0,0,:8]
    vp = proj(kvlora_n, v_w[:, 0:128])[0,:8]/(1<<16)
    print("float v[0,0,:8]=", vf)
    print("fp    v[0,0,:8]=", vp)
    pf = probs[0,0,:8]
    pp = None
    print("float probs[0,0,:8]=", pf)
    print("float attn[0,:4]=", af[0, :4])
    print("fp    attn[0,:4]=", ap[0, :4])


if __name__ == "__main__":
    main()
