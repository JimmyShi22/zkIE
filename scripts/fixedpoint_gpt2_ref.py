import numpy as np
SCALE = 65536
D, FFN, HEADS, DH, SEQ = 1024, 4096, 12, 64, 512
N_REAL = 768

def load(name):
    return np.fromfile(f"models/gpt2_stack/{name}", dtype=np.int32).astype(np.int64)

def dr(a, b):
    a = np.asarray(a, dtype=np.int64)
    q, r = np.divmod(a, b)
    return np.where(2 * r >= b, q + 1, q)

def rs(raw, sh):
    return dr(raw, 1 << sh)

def rsqrt_index(var):
    FINE = 1 << 20
    raw = dr(var, 1 << 18)
    return np.where(raw < FINE, np.maximum(raw, 0), FINE + ((raw - FINE) >> 8))

def layer_norm(x, w, b, rsqrt_t, m, d, n_real):
    mean = dr(x[:, :n_real].sum(axis=1), n_real)
    centered = x - mean[:, None]
    var = dr((centered[:, :n_real] ** 2).sum(axis=1), n_real)
    idx = rsqrt_index(var)
    rstd = rsqrt_t[idx][:, None]
    raw = centered * rstd * w
    out = rs(raw, 32) + b
    return out

def main():
    rsqrt_t = np.fromfile("models/rsqrt_table_i32.bin", dtype=np.int32).astype(np.int64)
    exp_t = np.fromfile("models/exp_table_i32.bin", dtype=np.int32).astype(np.int64)
    gelu_t = np.fromfile("models/gpt2_stack/gelu_table_i32.bin", dtype=np.int32).astype(np.int64)
    embed = load("embedding_i32.bin").reshape(SEQ, D)
    x = embed.astype(np.int64)

    for L in range(12):
        ln1_w = load(f"L{L}_ln1_w_i32.bin")
        ln1_b = load(f"L{L}_ln1_b_i32.bin")
        ln2_w = load(f"L{L}_ln2_w_i32.bin")
        ln2_b = load(f"L{L}_ln2_b_i32.bin")
        q_w = load(f"L{L}_q_w_i32.bin").reshape(D, D)
        k_w = load(f"L{L}_k_w_i32.bin").reshape(D, D)
        v_w = load(f"L{L}_v_w_i32.bin").reshape(D, D)
        o_w = load(f"L{L}_o_proj_w_i32.bin").reshape(D, D)
        fc_w = load(f"L{L}_fc_w_i32.bin").reshape(D, FFN)
        proj_w = load(f"L{L}_proj_w_i32.bin").reshape(FFN, D)
        q_b = load(f"L{L}_q_b_i32.bin")
        k_b = load(f"L{L}_k_b_i32.bin")
        v_b = load(f"L{L}_v_b_i32.bin")
        o_b = load(f"L{L}_o_proj_b_i32.bin")
        fc_b = load(f"L{L}_fc_b_i32.bin")
        proj_b = load(f"L{L}_proj_b_i32.bin")

        h = layer_norm(x, ln1_w, ln1_b, rsqrt_t, SEQ, D, N_REAL)
        heads_out = []
        for hd in range(HEADS):
            q = rs(h @ q_w[:, hd*DH:(hd+1)*DH], 16) + q_b[hd*DH:(hd+1)*DH]
            k = rs(h @ k_w[:, hd*DH:(hd+1)*DH], 16) + k_b[hd*DH:(hd+1)*DH]
            v = rs(h @ v_w[:, hd*DH:(hd+1)*DH], 16) + v_b[hd*DH:(hd+1)*DH]
            q = rs(q * 23170, 16)
            k = rs(k * 23170, 16)
            scores = rs(q @ k.T, 16)
            mask = np.triu(np.full((SEQ, SEQ), -(1 << 30)), k=1)
            scores = scores + mask
            c = scores.max(axis=1, keepdims=True)
            idx = np.clip(scores - c + (1 << 21), 0, len(exp_t) - 1)
            e = exp_t[idx]
            s = e.sum(axis=1, keepdims=True)
            sm = dr(e * SCALE, s)
            attn = rs(sm @ v, 16)
            heads_out.append(rs(attn @ o_w[hd*DH:(hd+1)*DH, :], 16) + o_b)
        attn_out = sum(heads_out)
        x2 = x + attn_out
        h2 = layer_norm(x2, ln2_w, ln2_b, rsqrt_t, SEQ, D, N_REAL)
        fc = rs(h2 @ fc_w, 16) + fc_b
        gelu_idx = np.clip(fc + (1 << 23), 0, len(gelu_t) - 1)
        act = gelu_t[gelu_idx]
        proj = rs(act @ proj_w, 16) + proj_b
        x = x2 + proj
        if L == 0:
            print("layer0 x[0,:4] =", x[0, :4].tolist())

    lnf_w = load("ln_f_w_i32.bin")
    lnf_b = load("ln_f_b_i32.bin")
    hf = layer_norm(x, lnf_w, lnf_b, rsqrt_t, SEQ, D, N_REAL)
    lm_w = load("lm_head_w_i32.bin").reshape(D, 65536)
    logits = hf @ lm_w
    argmax = logits.argmax(axis=1)
    gt = np.fromfile("models/gpt2_stack/gt_argmax_512_i32.bin", dtype=np.int32)
    print("argmax matches:", (argmax == gt).sum(), "/", SEQ)

if __name__ == "__main__":
    main()
