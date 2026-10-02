#!/usr/bin/env python3
"""Fixed-point numpy forward for DeepSeek-V2-Lite (mirrors the Rust prover),
in float64 for exactness of the i64 fixed-point values (< 2^53).

Seq-configurable via DS_SEQ. The MoE FFN only evaluates the routed (top-6)
experts, so this stays fast for larger sequences.
"""
import os

import numpy as np

DS = "models/deepseek-v2-lite/weights"
GPT = "models/gpt2/weights"

SEQ = int(os.environ.get("DS_SEQ", "16"))

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


def rsqrt_index(var):
    raw = rd(var, 1 << 18)
    fine = raw < (1 << 20)
    idx = np.where(fine, np.maximum(raw, 0), (1 << 20) + ((raw - (1 << 20)) / (1 << 8)).astype(np.int64))
    return idx.astype(np.int64)


def rms_norm(x, w, rsqrt_t):
    ms = rd((x * x).sum(-1), x.shape[-1])
    rstd = rsqrt_t[rsqrt_index(ms)]
    return rd(x * rstd[:, None] * w[None, :], 1 << 32)


def projection(x, W):
    return rd(x @ W, 1 << 16)


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    xf, xs = x[..., :half], x[..., half:]
    of = rd(xf * cos - xs * sin, 1 << 16)
    os = rd(xs * cos + xf * sin, 1 << 16)
    return np.concatenate([of, os], -1)


def softmax(s16, exp_t):
    m = s16.shape[0]
    mask = np.triu(np.full((m, m), -(1 << 30)), k=1)
    masked = s16 + mask
    row_max = masked.max(-1, keepdims=True)
    shifted = np.clip((masked - row_max) + (1 << 21), 0, (1 << 21) - 1).astype(np.int64)
    ev = exp_t[shifted]
    return rd(ev * (1 << 16), ev.sum(-1, keepdims=True))


def main():
    F = 1 << 16
    rsqrt_t = lgp("rsqrt_table_i32.bin")
    exp_t = lgp("exp_table_i32.bin")
    silu_t = li("silu_table_i32.bin")
    x = li("x0_i32.bin").reshape(SEQ, H)
    cos = li("rope_cos_i32.bin").reshape(SEQ, QK_ROPE // 2)
    sin = li("rope_sin_i32.bin").reshape(SEQ, QK_ROPE // 2)

    mscale = 0.1 * 0.707 * np.log(40.0) + 1.0
    softmax_scale = 192.0 ** -0.5 * mscale * mscale
    sf = round(softmax_scale * (1 << 16))

    for L in range(27):
        in_norm = li("L%d_in_norm_i32.bin" % L)
        post_norm = li("L%d_post_attn_norm_i32.bin" % L)
        h = rms_norm(x, in_norm, rsqrt_t)
        qnope_w = li("L%d_qnope_w_i32.bin" % L).reshape(H, HEADS * QK_NOPE)
        qpe_w = li("L%d_qpe_w_i32.bin" % L).reshape(H, HEADS * QK_ROPE)
        kvlora_w = li("L%d_kvlora_w_i32.bin" % L).reshape(H, KV_LORA)
        kpe_w = li("L%d_kpe_w_i32.bin" % L).reshape(H, QK_ROPE)
        kva_ln = li("L%d_kva_ln_i32.bin" % L)
        knope_w = li("L%d_knope_w_i32.bin" % L).reshape(KV_LORA, HEADS * QK_NOPE)
        v_w = li("L%d_v_w_i32.bin" % L).reshape(KV_LORA, HEADS * V_HEAD)
        o_w = li("L%d_o_w_i32.bin" % L).reshape(HEADS * V_HEAD, H)
        kvlora = projection(h, kvlora_w)
        kpe = projection(h, kpe_w)
        kvlora_n = rms_norm(kvlora, kva_ln, rsqrt_t)
        kpe_rot = rope(kpe, cos, sin)
        head_outs = []
        for hd in range(HEADS):
            qn = projection(h, qnope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
            qp = projection(h, qpe_w[:, hd * QK_ROPE:(hd + 1) * QK_ROPE])
            kn = projection(kvlora_n, knope_w[:, hd * QK_NOPE:(hd + 1) * QK_NOPE])
            vh = projection(kvlora_n, v_w[:, hd * V_HEAD:(hd + 1) * V_HEAD])
            qp_rot = rope(qp, cos, sin)
            s_nope = qn @ kn.T
            s_rope = qp_rot @ kpe_rot.T
            s16 = rd((s_nope + s_rope) * sf, 1 << 32)
            probs = softmax(s16, exp_t)
            attn16 = rd(probs @ vh, 1 << 16)
            head_outs.append(projection(attn16, o_w[hd * V_HEAD:(hd + 1) * V_HEAD]))
        x = x + sum(head_outs)
        h2 = rms_norm(x, post_norm, rsqrt_t)
        if L == 0:
            gw = li("L0_gate_w_i32.bin").reshape(H, DENSE_PAD)
            uw = li("L0_up_w_i32.bin").reshape(H, DENSE_PAD)
            dw = li("L0_down_w_i32.bin").reshape(DENSE_PAD, H)
            g = projection(h2, gw)
            u = projection(h2, uw)
            act = rd(silu_t[np.clip(g + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * u, 1 << 16)
            ff = projection(act, dw)
        else:
            router = li("L%d_router_i32.bin" % L).reshape(H, N_ROUTED)
            logits = (h2 @ router).astype(np.float64)
            logits_16 = rd(logits, F)
            row_max = logits_16.max(-1, keepdims=True)
            shifted = np.clip((logits_16 - row_max) + (1 << 21), 0, (1 << 21) - 1).astype(np.int64)
            ev = exp_t[shifted]
            s = rd(ev * F, ev.sum(-1, keepdims=True))
            topi = np.argsort(-s, axis=-1)[:, :6]
            gate = np.zeros((SEQ, N_ROUTED))
            for t in range(SEQ):
                gate[t, topi[t]] = s[t, topi[t]]
            gate = gate.astype(np.int64)
            np.asarray(gate, dtype=np.int32).tofile(DS + "/L%d_gate_i32.bin" % L)
            eg = li("L%d_experts_gate_i32.bin" % L).reshape(N_ROUTED, H, MOE_PAD)
            eu = li("L%d_experts_up_i32.bin" % L).reshape(N_ROUTED, H, MOE_PAD)
            ed = li("L%d_experts_down_i32.bin" % L).reshape(N_ROUTED, MOE_PAD, H)
            sg = li("L%d_shared_gate_i32.bin" % L).reshape(H, SHARED_PAD)
            su = li("L%d_shared_up_i32.bin" % L).reshape(H, SHARED_PAD)
            sd = li("L%d_shared_down_i32.bin" % L).reshape(SHARED_PAD, H)
            ff = np.zeros_like(h2)
            for e in np.unique(topi):
                g = projection(h2, eg[e])
                u = projection(h2, eu[e])
                act = rd(silu_t[np.clip(g + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * u, 1 << 16)
                d = projection(act, ed[e])
                ff = ff + rd(d * gate[:, e][:, None], 1 << 16)
            g = projection(h2, sg)
            u = projection(h2, su)
            act = rd(silu_t[np.clip(g + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * u, 1 << 16)
            d = projection(act, sd)
            ff = ff + d
        x = x + ff
        print("L%d max|x|=%.1f mean(x^2)=%.1f" % (L, np.abs(x).max(), (x.astype(np.float64) ** 2).mean()), flush=True)

    x = rms_norm(x, li("final_norm_i32.bin"), rsqrt_t)
    logits = x @ li("lm_head_i32.bin").reshape(H, VOCAB_PAD)
    argmax = np.argmax(logits[:, :VOCAB], axis=1)
    np.asarray(argmax, dtype=np.int32).tofile(DS + "/gt_argmax_i32.bin")
    print("argmax:", argmax.tolist())
    print("saved gt_argmax_i32.bin (%d tokens)" % SEQ)


if __name__ == "__main__":
    main()
