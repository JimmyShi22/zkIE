#!/usr/bin/env python3
"""Numpy float reference forward for DeepSeek-V2-Lite -> ground-truth argmax.

Loads the bf16 shards, runs the full MLA + MoE forward in float64 for a fixed
input sequence, and writes gt_argmax_i32.bin + gt_logits_f32.bin. Used to
validate the fixed-point extractor / Rust prover.
"""
import math
import os

import numpy as np

from extract_deepseek import (
    Sharded, SCALE, H, VOCAB, HEADS, QK_NOPE, QK_ROPE, V_HEAD, KV_LORA,
    N_ROUTED, TOPK, MOE_INTER, DENSE_INTER, SHARED_INTER, ROPE_BASE, SEQ,
)

OUT = "models/deepseek-v2-lite/weights"


def rms_norm(x, w, eps=1e-6):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    return np.concatenate([x[..., :half] * cos - x[..., half:] * sin,
                           x[..., half:] * cos + x[..., :half] * sin], axis=-1)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    S = Sharded()
    emb = S.arr("model.embed_tokens.weight").astype(np.float64)
    rng = np.random.default_rng(0)
    ids = rng.integers(1, VOCAB, (SEQ,)).astype(np.int64)
    x = emb[ids]  # [seq, 2048], no scaling

    mscale = 0.1 * 0.707 * math.log(40.0) + 1.0
    softmax_scale = (QK_NOPE + QK_ROPE) ** -0.5 * mscale * mscale

    pos = np.arange(SEQ, dtype=np.float64)[:, None]
    inv = np.arange(QK_ROPE // 2, dtype=np.float64)[None, :]
    theta = pos * np.power(ROPE_BASE, -2.0 * inv / QK_ROPE)
    cos, sin = np.cos(theta), np.sin(theta)

    for L in range(27):
        p = "model.layers.%d." % L
        in_norm = S.arr(p + "input_layernorm.weight").astype(np.float64)
        post_norm = S.arr(p + "post_attention_layernorm.weight").astype(np.float64)
        q_w = S.arr(p + "self_attn.q_proj.weight").astype(np.float64)
        kva_w = S.arr(p + "self_attn.kv_a_proj_with_mqa.weight").astype(np.float64)
        kva_ln = S.arr(p + "self_attn.kv_a_layernorm.weight").astype(np.float64)
        kvb_w = S.arr(p + "self_attn.kv_b_proj.weight").astype(np.float64)
        o_w = S.arr(p + "self_attn.o_proj.weight").astype(np.float64)

        residual = x
        h = rms_norm(x, in_norm)
        q = (h @ q_w.T).reshape(SEQ, HEADS, QK_NOPE + QK_ROPE)
        q_nope, q_pe = q[..., :QK_NOPE], q[..., QK_NOPE:]
        ckv = h @ kva_w.T  # [seq, 576]
        compressed_kv, k_pe = ckv[..., :KV_LORA], ckv[..., KV_LORA:KV_LORA + QK_ROPE]
        kv = (kvb_w @ rms_norm(compressed_kv, kva_ln).T).T  # [seq, 4096]
        kv = kv.reshape(SEQ, HEADS, QK_NOPE + V_HEAD)
        k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
        q_pe = apply_rope(q_pe, cos[:, None, :], sin[:, None, :])  # [seq,16,64]
        k_pe = apply_rope(k_pe[:, None, :], cos[:, None, :], sin[:, None, :])  # [seq,1,64]
        query = np.concatenate([q_nope, q_pe], axis=-1)  # [seq,16,192]
        key = np.concatenate([k_nope, np.broadcast_to(k_pe, (SEQ, HEADS, QK_ROPE))], axis=-1)
        scores = np.einsum("ihd,jhd->ihj", query, key) * softmax_scale  # [seq,16,seq]
        mask = np.triu(np.full((SEQ, SEQ), -1e30), k=1)
        probs = softmax(scores + mask[:, None, :], axis=-1)
        attn = np.einsum("ihk,khd->ihd", probs, v)  # [seq,16,128]
        attn = attn.reshape(SEQ, HEADS * V_HEAD) @ o_w.T  # [seq,2048]
        x = residual + attn

        residual = x
        h = rms_norm(x, post_norm)
        if L == 0:
            gw = S.arr(p + "mlp.gate_proj.weight").astype(np.float64)
            uw = S.arr(p + "mlp.up_proj.weight").astype(np.float64)
            dw = S.arr(p + "mlp.down_proj.weight").astype(np.float64)
            mlp = silu(h @ gw.T) * (h @ uw.T)
            mlp = mlp @ dw.T
        else:
            router = S.arr(p + "mlp.gate.weight").astype(np.float64)
            sg = S.arr(p + "mlp.shared_experts.gate_proj.weight").astype(np.float64)
            su = S.arr(p + "mlp.shared_experts.up_proj.weight").astype(np.float64)
            sd = S.arr(p + "mlp.shared_experts.down_proj.weight").astype(np.float64)
            logits = h @ router.T  # [seq, 64]
            scores = softmax(logits, axis=-1)
            topi = np.argsort(-scores, axis=-1)[:, :TOPK]  # [seq,6]
            topw = np.take_along_axis(scores, topi, axis=-1)  # [seq,6]
            gate = np.zeros((SEQ, N_ROUTED))
            for t in range(SEQ):
                for k in range(TOPK):
                    gate[t, topi[t, k]] = topw[t, k]
            np.clip(np.round(gate * SCALE), -(2 ** 31), 2 ** 31 - 1).astype(np.int32).tofile(
                os.path.join(OUT, "L%d_gate_i32.bin" % L))
            mlp = np.zeros_like(h)
            for e in range(N_ROUTED):
                eg = S.arr(p + "mlp.experts.%d.gate_proj.weight" % e).astype(np.float64)
                eu = S.arr(p + "mlp.experts.%d.up_proj.weight" % e).astype(np.float64)
                ed = S.arr(p + "mlp.experts.%d.down_proj.weight" % e).astype(np.float64)
                for t in range(SEQ):
                    for k in range(TOPK):
                        if topi[t, k] == e:
                            out = silu(h[t] @ eg.T) * (h[t] @ eu.T)
                            mlp[t] += topw[t, k] * (out @ ed.T)
            sh = silu(h @ sg.T) * (h @ su.T)
            mlp = mlp + sh @ sd.T
        x = residual + mlp
        print("L%d max=%.4f mean(x^2)=%.6f" % (L, np.abs(x).max(), (x*x).mean()))

    x = rms_norm(x, S.arr("model.norm.weight").astype(np.float64))
    logits = x @ S.arr("lm_head.weight").astype(np.float64).T  # [seq, vocab]
    argmax = np.argmax(logits, axis=1).astype(np.int32)
    np.asarray(argmax, dtype=np.int32).reshape(-1, 1).tofile(os.path.join(OUT, "gt_argmax_i32.bin"))
    np.asarray(logits, dtype=np.float32).tofile(os.path.join(OUT, "gt_logits_f32.bin"))
    print("argmax:", argmax.tolist())


if __name__ == "__main__":
    main()
