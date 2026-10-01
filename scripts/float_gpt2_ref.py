import numpy as np
def load(name):
    return np.fromfile(f"models/gpt2_stack/{name}", dtype=np.int32).astype(np.float64) / 65536.0

D, FFN, HEADS, DH, SEQ, N_REAL = 1024, 4096, 12, 64, 512, 768

def ln(x, w, b):
    xr = x[:, :N_REAL]
    mean = xr.mean(axis=1, keepdims=True)
    var = xr.var(axis=1, keepdims=True)
    return (xr - mean) / np.sqrt(var + 1e-5) * w[:N_REAL] + b[:N_REAL]

x = load("embedding_i32.bin").reshape(SEQ, D)[:, :N_REAL]

for L in range(12):
    ln1_w = load(f"L{L}_ln1_w_i32.bin")
    ln1_b = load(f"L{L}_ln1_b_i32.bin")
    ln2_w = load(f"L{L}_ln2_w_i32.bin")
    ln2_b = load(f"L{L}_ln2_b_i32.bin")
    q_w = load(f"L{L}_q_w_i32.bin").reshape(D, D)[:N_REAL, :N_REAL]
    k_w = load(f"L{L}_k_w_i32.bin").reshape(D, D)[:N_REAL, :N_REAL]
    v_w = load(f"L{L}_v_w_i32.bin").reshape(D, D)[:N_REAL, :N_REAL]
    o_w = load(f"L{L}_o_proj_w_i32.bin").reshape(D, D)[:N_REAL, :N_REAL]
    fc_w = load(f"L{L}_fc_w_i32.bin").reshape(D, FFN)[:N_REAL, :3072]
    proj_w = load(f"L{L}_proj_w_i32.bin").reshape(FFN, D)[:3072, :N_REAL]
    q_b = load(f"L{L}_q_b_i32.bin")[:N_REAL]
    k_b = load(f"L{L}_k_b_i32.bin")[:N_REAL]
    v_b = load(f"L{L}_v_b_i32.bin")[:N_REAL]
    o_b = load(f"L{L}_o_proj_b_i32.bin")[:N_REAL]
    fc_b = load(f"L{L}_fc_b_i32.bin")[:3072]
    proj_b = load(f"L{L}_proj_b_i32.bin")[:N_REAL]

    h = ln(x, ln1_w, ln1_b)
    attn_heads = []
    for hd in range(HEADS):
        q = h @ q_w[:, hd*DH:(hd+1)*DH] + q_b[hd*DH:(hd+1)*DH]
        k = h @ k_w[:, hd*DH:(hd+1)*DH] + k_b[hd*DH:(hd+1)*DH]
        v = h @ v_w[:, hd*DH:(hd+1)*DH] + v_b[hd*DH:(hd+1)*DH]
        q = q * 0.3535533845424652
        k = k * 0.3535533845424652
        scores = q @ k.T
        mask = np.triu(np.full((SEQ, SEQ), -1e30), k=1)
        probs = np.exp(scores + mask - (scores + mask).max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        attn = probs @ v
        attn_heads.append(attn @ o_w[hd*DH:(hd+1)*DH, :])
    x2 = x + sum(attn_heads) + o_b
    h2 = ln(x2, ln2_w, ln2_b)
    fc = h2 @ fc_w + fc_b
    act = 0.5 * fc * (1.0 + np.tanh(0.7978845608028654 * (fc + 0.044715 * fc**3)))
    proj = act @ proj_w + proj_b
    x = x2 + proj

lnf_w = load("ln_f_w_i32.bin")
lnf_b = load("ln_f_b_i32.bin")
hf = ln(x, lnf_w, lnf_b)
lm_w = load("lm_head_w_i32.bin").reshape(D, 65536)[:N_REAL, :]
logits = hf @ lm_w[:, :50257]
argmax = logits.argmax(axis=1)
gt = np.fromfile("models/gpt2_stack/gt_argmax_512_i32.bin", dtype=np.int32)
print("float argmax matches:", (argmax == gt).sum(), "/", SEQ)
