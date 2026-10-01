#!/usr/bin/env python3
import numpy as np

DS = "models/deepseek-v2-lite/weights"
GPT = "models/gpt2/weights"
H = 2048
HEADS = 16
QK_NOPE = 128
QK_ROPE = 64
KV_LORA = 512


def li(name):
    return np.fromfile("%s/%s" % (DS, name), dtype="<i4").astype(np.float64)


def lgp(name):
    return np.fromfile("%s/%s" % (GPT, name), dtype="<i4").astype(np.float64)


def rd(a, b):
    q = np.floor(a / b)
    r = a - q * b
    return q + (r * 2 >= b)


def rms(x, w):
    ms = rd((x * x).sum(-1), x.shape[-1])
    rstd = np.round((1.0 / np.sqrt(ms / (1 << 32) + 1e-6)) * (1 << 16))
    return rd(x * rstd[:, None] * w[None, :], 1 << 32)


def proj(x, W):
    return rd(x @ W, 1 << 16)


def main():
    x = li("x0_i32.bin").reshape(16, H)
    h = rms(x, li("L0_in_norm_i32.bin"))
    qnope_w = li("L0_qnope_w_i32.bin").reshape(H, HEADS * QK_NOPE)
    qpe_w = li("L0_qpe_w_i32.bin").reshape(H, HEADS * QK_ROPE)
    kvlora_w = li("L0_kvlora_w_i32.bin").reshape(H, KV_LORA)
    kpe_w = li("L0_kpe_w_i32.bin").reshape(H, QK_ROPE)
    knope_w = li("L0_knope_w_i32.bin").reshape(KV_LORA, HEADS * QK_NOPE)
    v_w = li("L0_v_w_i32.bin").reshape(KV_LORA, HEADS * QK_NOPE)
    kvlora = proj(h, kvlora_w)
    kpe = proj(h, kpe_w)
    kvlora_n = rms(kvlora, li("L0_kva_ln_i32.bin"))
    cos = li("rope_cos_i32.bin").reshape(16, QK_ROPE // 2)
    sin = li("rope_sin_i32.bin").reshape(16, QK_ROPE // 2)

    def rope(x):
        hh = x.shape[-1] // 2
        return np.concatenate([rd(x[..., :hh] * cos - x[..., hh:] * sin, 1 << 16),
                               rd(x[..., hh:] * cos + x[..., :hh] * sin, 1 << 16)], -1)
    kpe_rot = rope(kpe)
    mscale = 0.1 * 0.707 * np.log(40.0) + 1.0
    ss = 192.0 ** -0.5 * mscale * mscale
    sf = round(ss * (1 << 16))

    # head 0, in FLOAT (weights / 2^16) and FP (Q16)
    qn_f = proj(h, qnope_w[:, 0:QK_NOPE]).astype(np.float64) / (1 << 16)
    qp_f = proj(h, qpe_w[:, 0:QK_ROPE]).astype(np.float64) / (1 << 16)
    kn_f = proj(kvlora_n, knope_w[:, 0:QK_NOPE]).astype(np.float64) / (1 << 16)
    qp_rot_f = rope(proj(h, qpe_w[:, 0:QK_ROPE])).astype(np.float64) / (1 << 16)
    kpe_rot_f = kpe_rot.astype(np.float64) / (1 << 16)
    score_float = (qn_f @ kn_f.T + qp_rot_f @ kpe_rot_f.T) * ss

    qn = proj(h, qnope_w[:, 0:QK_NOPE])
    qp = proj(h, qpe_w[:, 0:QK_ROPE])
    kn = proj(kvlora_n, knope_w[:, 0:QK_NOPE])
    qp_rot = rope(qp)
    s = qn @ kn.T + qp_rot @ kpe_rot.T
    s16 = rd(s * sf, 1 << 32)
    score_fp = s16.astype(np.float64) / (1 << 16)

    print("softmax_scale ss=%.6f sf=%d" % (ss, sf))
    print("float score[5,:6]=", score_float[5, :6])
    print("fp    score[5,:6]=", score_fp[5, :6])
    print("float score[3,:6]=", score_float[3, :6])
    print("fp    score[3,:6]=", score_fp[3, :6])
    print("max abs diff scores=", np.abs(score_float - score_fp).max())


if __name__ == "__main__":
    main()
