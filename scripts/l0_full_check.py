#!/usr/bin/env python3
import numpy as np

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
    x0 = li("x0_i32.bin").reshape(16, H)
    in_norm = li("L0_in_norm_i32.bin")
    post_norm = li("L0_post_attn_norm_i32.bin")
    kva_ln = li("L0_kva_ln_i32.bin")
    qnope_w = li("L0_qnope_w_i32.bin").reshape(H, HEADS * QK_NOPE)
    qpe_w = li("L0_qpe_w_i32.bin").reshape(H, HEADS * QK_ROPE)
    kvlora_w = li("L0_kvlora_w_i32.bin").reshape(H, KV_LORA)
    kpe_w = li("L0_kpe_w_i32.bin").reshape(H, QK_ROPE)
    knope_w = li("L0_knope_w_i32.bin").reshape(KV_LORA, HEADS * QK_NOPE)
    v_w = li("L0_v_w_i32.bin").reshape(KV_LORA, HEADS * V_HEAD)
    o_w = li("L0_o_w_i32.bin").reshape(HEADS * V_HEAD, H)
    gw = li("L0_gate_w_i32.bin").reshape(H, DENSE_PAD)
    uw = li("L0_up_w_i32.bin").reshape(H, DENSE_PAD)
    dw = li("L0_down_w_i32.bin").reshape(DENSE_PAD, H)
    silu_t = li("silu_table_i32.bin")
    exp_t = lgp("exp_table_i32.bin")
    cos = li("rope_cos_i32.bin").reshape(16, QK_ROPE // 2)
    sin = li("rope_sin_i32.bin").reshape(16, QK_ROPE // 2)

    mscale = 0.1 * 0.707 * np.log(40.0) + 1.0
    ss = 192.0 ** -0.5 * mscale * mscale
    F = 1 << 16

    def rmsf(x, w):
        ms = (x * x).mean(-1, keepdims=True)
        return x * (1.0 / np.sqrt(ms + 1e-6)) * w[None, :] / F

    def projf(x, W):
        return x @ (W / F)

    def ropef(x):
        hh = x.shape[-1] // 2
        c, s = cos / F, sin / F
        return np.concatenate([x[..., :hh] * c - x[..., hh:] * s, x[..., hh:] * c + x[..., :hh] * s], -1)

    x = x0 / F
    h = rmsf(x, in_norm)
    kvlora = projf(h, kvlora_w)
    kpe = projf(h, kpe_w)
    kvlora_n = rmsf(kvlora, kva_ln)
    kpe_rot = ropef(kpe)
    attn = np.zeros_like(x)
    for hd in range(HEADS):
        qn = projf(h, qnope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
        qp = projf(h, qpe_w[:, hd * QK_ROPE:(hd + 1) * QK_ROPE])
        kn = projf(kvlora_n, knope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
        vh = projf(kvlora_n, v_w[:, hd * V_HEAD:(hd + 1) * V_HEAD])
        qp_rot = ropef(qp)
        s = (qn @ kn.T + qp_rot @ kpe_rot.T) * ss
        mask = np.triu(np.full((16, 16), -1e30), 1)
        p = np.exp(s + mask - (s + mask).max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        attn += (p @ vh) @ (o_w[hd * V_HEAD:(hd + 1) * V_HEAD] / F)
    x = x + attn
    h2 = rmsf(x, post_norm)
    g = projf(h2, gw)
    u = projf(h2, uw)
    act = (g / (1.0 + np.exp(-g))) * u
    ff_f = act @ (dw / F)

    def rms(x, w):
        ms = rd((x * x).sum(-1, keepdims=True), x.shape[-1])
        rstd = np.round((1.0 / np.sqrt(ms / (F * F) + 1e-6)) * F)
        return rd(x * rstd * w[None, :], F * F)

    def proj(x, W):
        return rd(x @ W, F)

    def rope(x):
        hh = x.shape[-1] // 2
        return np.concatenate([rd(x[..., :hh] * cos - x[..., hh:] * sin, F),
                               rd(x[..., hh:] * cos + x[..., :hh] * sin, F)], -1)

    xq = x0
    hq = rms(xq, in_norm)
    kvlora_q = proj(hq, kvlora_w)
    kpe_q = proj(hq, kpe_w)
    kvlora_nq = rms(kvlora_q, kva_ln)
    kpe_rot_q = rope(kpe_q)
    sf = round(ss * F)
    attn_q = np.zeros_like(xq, dtype=np.float64)
    for hd in range(HEADS):
        qn = proj(hq, qnope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
        qp = proj(hq, qpe_w[:, hd * QK_ROPE:(hd + 1) * QK_ROPE])
        kn = proj(kvlora_nq, knope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
        vh = proj(kvlora_nq, v_w[:, hd * V_HEAD:(hd + 1) * V_HEAD])
        qp_rot = rope(qp)
        s16 = rd((qn @ kn.T + qp_rot @ kpe_rot_q.T) * sf, 1 << 32)
        masked = s16 + np.triu(np.full((16, 16), -(1 << 30)), 1)
        rm = masked.max(-1, keepdims=True)
        shifted = np.clip((masked - rm) + (1 << 21), 0, (1 << 21) - 1).astype(np.int64)
        ev = exp_t[shifted]
        pr = rd(ev * F, ev.sum(-1, keepdims=True))
        a16 = rd(pr @ vh, F)
        attn_q += rd(a16 @ o_w[hd * V_HEAD:(hd + 1) * V_HEAD], F)
    xq = xq + attn_q
    h2q = rms(xq, post_norm)
    gq = proj(h2q, gw)
    uq = proj(h2q, uw)
    act_q = rd(silu_t[np.clip(gq + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * uq, F)
    ff_q = proj(act_q, dw)

    print("float attn max=%.5f  fp attn max=%.5f" % (np.abs(attn).max(), np.abs(attn_q / F).max()))
    print("float ff max=%.5f  fp ff max=%.5f" % (np.abs(ff_f).max(), np.abs(ff_q / F).max()))
    print("float ff[0,:4]=", ff_f[0, :4])
    print("fp    ff[0,:4]=", (ff_q / F)[0, :4])


if __name__ == "__main__":
    main()
