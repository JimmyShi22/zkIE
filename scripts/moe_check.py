#!/usr/bin/env python3
import numpy as np

DS = "models/deepseek-v2-lite/weights"
H = 2048
MOE_PAD = 2048
SHARED_PAD = 4096
N_ROUTED = 64


def li(name):
    return np.fromfile("%s/%s" % (DS, name), dtype="<i4").astype(np.float64)


def rd(a, b):
    q = np.floor(a / b)
    r = a - q * b
    return q + (r * 2 >= b)


def main():
    L = 1
    F = 1 << 16
    silu_t = li("silu_table_i32.bin")
    gate = li("L%d_gate_i32.bin" % L).reshape(16, N_ROUTED)  # Q16
    eg = li("L%d_experts_gate_i32.bin" % L).reshape(N_ROUTED, H, MOE_PAD)
    eu = li("L%d_experts_up_i32.bin" % L).reshape(N_ROUTED, H, MOE_PAD)
    ed = li("L%d_experts_down_i32.bin" % L).reshape(N_ROUTED, MOE_PAD, H)
    sg = li("L%d_shared_gate_i32.bin" % L).reshape(H, SHARED_PAD)
    su = li("L%d_shared_up_i32.bin" % L).reshape(H, SHARED_PAD)
    sd = li("L%d_shared_down_i32.bin" % L).reshape(SHARED_PAD, H)

    # use the fp L1 input h2 (Q16) as the common input; also its float version
    h2q = np.random.default_rng(0).integers(-10000, 10000, (16, H)).astype(np.float64)
    h2f = h2q / F

    # float MoE
    def siluf(x):
        return x / (1.0 + np.exp(-x))
    ff_f = np.zeros((16, H))
    for e in range(N_ROUTED):
        g = h2f @ (eg[e] / F)
        u = h2f @ (eu[e] / F)
        act = siluf(g) * u
        d = act @ (ed[e] / F)
        ff_f += d * (gate[:, e] / F)[:, None]
    g = h2f @ (sg / F)
    u = h2f @ (su / F)
    act = siluf(g) * u
    ff_f += act @ (sd / F)

    # fp MoE
    def proj(x, W):
        return rd(x @ W, F)
    ff_q = np.zeros((16, H), dtype=np.float64)
    for e in range(N_ROUTED):
        g = proj(h2q, eg[e])
        u = proj(h2q, eu[e])
        act = rd(silu_t[np.clip(g + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * u, F)
        d = proj(act, ed[e])
        ff_q += rd(d * gate[:, e][:, None], F)
    g = proj(h2q, sg)
    u = proj(h2q, su)
    act = rd(silu_t[np.clip(g + (1 << 23), 0, (1 << 24) - 1).astype(np.int64)] * u, F)
    d = proj(act, sd)
    ff_q += d

    print("float ff max=%.5f  fp ff max=%.5f" % (np.abs(ff_f).max(), np.abs(ff_q / F).max()))
    print("float ff[0,:4]=%.5f %.5f %.5f %.5f" % tuple(ff_f[0, :4]))
    print("fp    ff[0,:4]=%.5f %.5f %.5f %.5f" % tuple((ff_q / F)[0, :4]))
    diff = np.abs(ff_f - ff_q / F)
    print("max abs diff=%.6f" % diff.max())


if __name__ == "__main__":
    main()
