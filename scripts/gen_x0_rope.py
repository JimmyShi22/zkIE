#!/usr/bin/env python3
"""Generate seq-configurable x0 + RoPE tables for DeepSeek-V2-Lite.

Reuses extract_deepseek.Sharded to read the embedding, then writes only the
seq-dependent inputs (x0 and rope). Weights are left untouched.
"""
import numpy as np
import extract_deepseek as E


def main():
    seq = E.SEQ
    S = E.Sharded()
    emb = S.arr("model.embed_tokens.weight").astype(np.float64)
    rng = np.random.default_rng(0)
    input_ids = rng.integers(1, E.VOCAB, (seq,)).astype(np.int64)
    x0 = E.pad2(E.q(emb[input_ids]), seq, E.H)
    E.save("x0_i32.bin", x0)
    pos = np.arange(seq, dtype=np.float64)[:, None]
    i = np.arange(E.QK_ROPE // 2, dtype=np.float64)[None, :]
    theta = pos * np.power(E.ROPE_BASE, -2.0 * i / E.QK_ROPE)
    E.save("rope_cos_i32.bin", E.q(np.cos(theta)))
    E.save("rope_sin_i32.bin", E.q(np.sin(theta)))
    print("x0", x0.shape, "rope", theta.shape)


if __name__ == "__main__":
    main()
