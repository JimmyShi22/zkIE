#!/usr/bin/env python3
"""Run the full 27-layer forward in BOTH fp (Q16+round) and float (weights/2^16),
printing the per-layer max-abs-diff of the residual x to find the divergence."""
import numpy as np

DS = "models/deepseek-v2-lite/weights"
GPT = "models/gpt2/weights"
H = 2048
VOCAB = 102400
VOCAB_PAD = 131072
HEADS = 16
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
KV_LORA = 512
MOE_PAD = 2048
DENSE_PAD = 16384
SHARED_PAD = 4096
N_ROUTED = 64


def li(name):
    return np.fromfile("%s/%s" % (DS, name), dtype="<i4").astype(np.float64)


def lgp(name):
    return np.fromfile("%s/%s" % (GPT, name), dtype="<i4").astype(np.float64)


def rd(a, b):
    q = np.floor(a / b)
    r = a - q * b
    return q + (r * 2 >= b)


def main():
    F = 1 << 16
    exp_t = lgp("exp_table_i32.bin")
    silu_t = li("silu_table_i32.bin")
    cos = li("rope_cos_i32.bin").reshape(16, QK_ROPE // 2)
    sin = li("rope_sin_i32.bin").reshape(16, QK_ROPE // 2)
    x0 = li("x0_i32.bin").reshape(16, H)
    mscale = 0.1 * 0.707 * np.log(40.0) + 1.0
    ss = 192.0 ** -0.5 * mscale * mscale

    # float helpers
    def rmsf(x, w):
        ms = (x * x).mean(-1, keepdims=True)
        return x * (1.0 / np.sqrt(ms + 1e-6)) * w[None, :] / F
    def projf(x, W):
        return x @ (W / F)
    def ropef(x):
        hh = x.shape[-1] // 2
        c, s = cos / F, sin / F
        return np.concatenate([x[..., :hh] * c - x[..., hh:] * s, x[..., hh:] * c + x[..., :hh] * s], -1)
    def softmaxf(s, mask):
        m = s + mask
        m = m - m.max(-1, keepdims=True)
        e = np.exp(m)
        return e / e.sum(-1, keepdims=True)

    # fp helpers
    def rmsq(x, w):
        ms = rd((x * x).sum(-1, keepdims=True), x.shape[-1])
        rstd = np.round((1.0 / np.sqrt(ms / (F * F) + 1e-6)) * F)
        return rd(x * rstd * w[None, :], F * F)
    def projq(x, W):
        return rd(x @ W, F)
    def ropeq(x):
        hh = x.shape[-1] // 2
        return np.concatenate([rd(x[..., :hh] * cos - x[..., hh:] * sin, F),
                               rd(x[..., hh:] * cos + x[..., :hh] * sin, F)], -1)
    def softmaxq(s16, mask):
        masked = s16 + mask
        rm = masked.max(-1, keepdims=True)
        shifted = np.clip((masked - rm) + (1 << 21), 0, (1 << 21) - 1).astype(np.int64)
        ev = exp_t[shifted]
        return rd(ev * F, ev.sum(-1, keepdims=True))

    xf = x0 / F
    xq = x0.astype(np.float64)
    sf = round(ss * F)
    for L in range(27):
        in_norm = li("L%d_in_norm_i32.bin" % L)
        post_norm = li("L%d_post_attn_norm_i32.bin" % L)
        qnope_w = li("L%d_qnope_w_i32.bin" % L).reshape(H, HEADS * QK_NOPE)
        qpe_w = li("L%d_qpe_w_i32.bin" % L).reshape(H, HEADS * QK_ROPE)
        kvlora_w = li("L%d_kvlora_w_i32.bin" % L).reshape(H, KV_LORA)
        kpe_w = li("L%d_kpe_w_i32.bin" % L).reshape(H, QK_ROPE)
        kva_ln = li("L%d_kva_ln_i32.bin" % L)
        knope_w = li("L%d_knope_w_i32.bin" % L).reshape(KV_LORA, HEADS * QK_NOPE)
        v_w = li("L%d_v_w_i32.bin" % L).reshape(KV_LORA, HEADS * V_HEAD)
        o_w = li("L%d_o_w_i32.bin" % L).reshape(HEADS * V_HEAD, H)

        # float layer
        hf = rmsf(xf, in_norm)
        kvlora_f = projf(hf, kvlora_w)
        kpe_f = projf(hf, kpe_w)
        kvlora_nf = rmsf(kvlora_f, kva_ln)
        kpe_rot_f = ropef(kpe_f)
        attn_f = np.zeros_like(xf)
        for hd in range(HEADS):
            qn = projf(hf, qnope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
            qp = projf(hf, qpe_w[:, hd * QK_ROPE:(hd + 1) * QK_ROPE])
            kn = projf(kvlora_nf, knope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
            vh = projf(kvlora_nf, v_w[:, hd * V_HEAD:(hd + 1) * V_HEAD])
            qp_rot = ropef(qp)
            s = (qn @ kn.T + qp_rot @ kpe_rot_f.T) * ss
            mask = np.triu(np.full((16, 16), -1e30), 1)
            p = softmaxf(s, mask)
            attn_f += (p @ vh) @ (o_w[hd * V_HEAD:(hd + 1) * V_HEAD] / F)
        xf = xf + attn_f
        h2f = rmsf(xf, post_norm)

        # fp layer
        hq = rmsq(xq, in_norm)
        kvlora_q = projq(hq, kvlora_w)
        kpe_q = projq(hq, kpe_w)
        kvlora_nq = rmsq(kvlora_q, kva_ln)
        kpe_rot_q = ropeq(kpe_q)
        attn_q = np.zeros_like(xq)
        for hd in range(HEADS):
            qn = projq(hq, qnope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
            qp = projq(hq, qpe_w[:, hd * QK_ROPE:(hd + 1) * QK_ROPE])
            kn = projq(kvlora_nq, knope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
            vh = projq(kvlora_nq, v_w[:, hd * V_HEAD:(hd + 1) * V_HEAD])
            qp_rot = ropeq(qp)
            s16 = rd((qn @ kn.T + qp_rot @ kpe_rot_q.T) * sf, 1 << 32)
            maskq = np.triu(np.full((16, 16), -(1 << 30)), 1)
            p = softmaxq(s16, maskq)
            a16 = rd(p @ vh, F)
            attn_q += rd(a16 @ o_w[hd * V_HEAD:(hd + 1) * V_HEAD], F)
        xq = xq + attn_q
        h2q = rmsq(xq, post_norm)

        # FFN
        if L == 0:
            gw = li("L0_gate_w_i32.bin").reshape(H, DENSE_PAD)
            uw = li("L0_up_w_i32.bin").reshape(H, DENSE_PAD)
            dw = li("L0_down_w_i32.bin").reshape(DENSE_PAD, H)
            g = projf(h2f, gw); u = projf(h2f, uw)
            ff_f = (g / (1.0 + np.exp(-g)) * u) @ (dw / F)
            g = projq(h2q, gw); u = projq(h2q, uw)
            act = rd(silu_t[np.clip(g + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * u, F)
            ff_q = projq(act, dw)
        else:
            eg = li("L%d_experts_gate_i32.bin" % L).reshape(N_ROUTED, H, MOE_PAD)
            eu = li("L%d_experts_up_i32.bin" % L).reshape(N_ROUTED, H, MOE_PAD)
            ed = li("L%d_experts_down_i32.bin" % L).reshape(N_ROUTED, MOE_PAD, H)
            sg = li("L%d_shared_gate_i32.bin" % L).reshape(H, SHARED_PAD)
            su = li("L%d_shared_up_i32.bin" % L).reshape(H, SHARED_PAD)
            sd = li("L%d_shared_down_i32.bin" % L).reshape(SHARED_PAD, H)
            router = li("L%d_router_i32.bin" % L).reshape(H, N_ROUTED)
            # gate from router (both float and fp use the SAME float routing for comparison)
            lg = h2f @ (router / F)
            s = np.exp(lg - lg.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            topi = np.argsort(-s, axis=-1)[:, :6]
            gate = np.zeros((16, N_ROUTED))
            for t in range(16):
                gate[t, topi[t]] = s[t, topi[t]]
            ff_f = np.zeros_like(h2f)
            ff_q = np.zeros_like(h2q)
            for e in range(N_ROUTED):
                g = projf(h2f, eg[e]); u = projf(h2f, eu[e])
                act = (g / (1.0 + np.exp(-g)) * u)
                ff_f += (act @ (ed[e] / F)) * gate[:, e][:, None]
                g = projq(h2q, eg[e]); u = projq(h2q, eu[e])
                act = rd(silu_t[np.clip(g + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * u, F)
                d = projq(act, ed[e])
                ff_q += rd(d * (gate[:, e][:, None] * F), F)
            g = projf(h2f, sg); u = projf(h2f, su)
            ff_f += (g / (1.0 + np.exp(-g)) * u) @ (sd / F)
            g = projq(h2q, sg); u = projq(h2q, su)
            act = rd(silu_t[np.clip(g + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * u, F)
            ff_q += projq(act, sd)
        xf = xf + ff_f
        xq = xq + ff_q
        diff = np.abs(xf - xq / F)
        print("L%d: max|x| float=%.4f fp=%.4f  max_abs_diff=%.6f  mean_diff=%.8f" %
              (L, np.abs(xf).max(), np.abs(xq / F).max(), diff.max(), diff.mean()), flush=True)



    final_norm = li("final_norm_i32.bin")
    lm = li("lm_head_i32.bin").reshape(H, VOCAB_PAD)
    xf2 = rmsf(xf, final_norm)
    xq2 = rmsq(xq, final_norm)
    lf = xf2 @ (lm / F)
    lq = (xq2 @ lm).astype(np.float64) / F
    amf = lf[:, :VOCAB].argmax(-1)
    amq = lq[:, :VOCAB].argmax(-1)
    gt = li("gt_argmax_i32.bin").astype(np.int64)
    print("float argmax:", amf.tolist())
    print("fp    argmax:", amq.tolist())
    print("gt    argmax:", gt.tolist())
    print("float match=", (amf == gt).sum(), " fp match=", (amq == gt).sum())
if __name__ == "__main__":
    main()
