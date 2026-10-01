#!/usr/bin/env python3
"""Extract Gemma 3 270M weights from safetensors into padded int32 fixed-point.

Needs numpy (Python 3.11). Parses the bf16 safetensors, quantizes at 2^16,
zero-pads each 2D weight dim to a power of two, writes `.bin` files, precomputes
the embedded input and the RoPE cos/sin tables, and runs a numpy float reference
forward to emit ground-truth logits + argmax.

Weight layout is `[in, out]` (transposed from HF's `[out, in]`), matching the
existing GPT-2/TimesFM extractors.
"""
import json
import math
import os
import struct

import numpy as np

RAW = "models/gemma3/raw/model.safetensors"
OUT = "models/gemma3/weights"

SEQ = int(os.environ.get("GEMMA_SEQ", "16"))
SCALE = 65536.0
H = 640
H_PAD = 1024
INTER = 2048
HEADS = 4
KV_HEADS = 1
HDIM = 256
VOCAB = 262144
LAYERS = 18

LAYER_TYPES = (
    ["sliding_attention"] * 5
    + ["full_attention"]
    + ["sliding_attention"] * 5
    + ["full_attention"]
    + ["sliding_attention"] * 5
    + ["full_attention"]
)
assert len(LAYER_TYPES) == LAYERS

ROPE_LOCAL = 10000.0
ROPE_GLOBAL = 1000000.0


def next_pow2(n):
    return 1 << (n - 1).bit_length()


class Loader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        hlen = struct.unpack("<Q", self.data[:8])[0]
        self.header = json.loads(self.data[8:8 + hlen].decode("utf-8"))
        self.start = 8 + hlen

    def arr(self, name):
        v = self.header[name]
        s, e = v["data_offsets"]
        raw = self.data[self.start + s:self.start + e]
        if v["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2")
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return f32.reshape(v["shape"])
        if v["dtype"] == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(v["shape"])
        raise ValueError("unsupported dtype %s" % v["dtype"])


def q(x):
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * SCALE), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)


def pad2(a, r, c):
    o = np.zeros((r, c), dtype=np.int32)
    a = np.asarray(a)
    o[:a.shape[0], :a.shape[1]] = a
    return o


def pad1(a, n):
    o = np.zeros(n, dtype=np.int32)
    a = np.asarray(a)
    o[:a.size] = a.reshape(-1)
    return o


def save(name, arr):
    np.asarray(arr, dtype=np.int32).tofile(os.path.join(OUT, name))
    print("  wrote %-34s %s" % (name, list(np.asarray(arr).shape)))


def rms_norm(x, w, eps=1e-6):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ms + eps)) * w


def apply_rope(x, cos, sin):
    # x [seq, heads, d], cos/sin [seq, d/2]
    half = HDIM // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    L = Loader(RAW)

    emb = L.arr("model.embed_tokens.weight").astype(np.float64)  # [vocab, hidden]
    rng = np.random.default_rng(0)
    input_ids = rng.integers(1, VOCAB, (SEQ,)).astype(np.int64)

    sq = math.sqrt(H)
    x0 = pad2(q(emb[input_ids] * sq), SEQ, H_PAD)
    save("x0_i32.bin", x0)
    save("lm_head_i32.bin", pad2(q(emb.T), H_PAD, VOCAB))

    for i in range(LAYERS):
        p = "model.layers.%d." % i
        save("L%d_q_w_i32.bin" % i, pad2(q(L.arr(p + "self_attn.q_proj.weight").T), H_PAD, HEADS * HDIM))
        save("L%d_k_w_i32.bin" % i, pad2(q(L.arr(p + "self_attn.k_proj.weight").T), H_PAD, KV_HEADS * HDIM))
        save("L%d_v_w_i32.bin" % i, pad2(q(L.arr(p + "self_attn.v_proj.weight").T), H_PAD, KV_HEADS * HDIM))
        save("L%d_o_w_i32.bin" % i, pad2(q(L.arr(p + "self_attn.o_proj.weight").T), HEADS * HDIM, H_PAD))
        save("L%d_gate_w_i32.bin" % i, pad2(q(L.arr(p + "mlp.gate_proj.weight").T), H_PAD, INTER))
        save("L%d_up_w_i32.bin" % i, pad2(q(L.arr(p + "mlp.up_proj.weight").T), H_PAD, INTER))
        save("L%d_down_w_i32.bin" % i, pad2(q(L.arr(p + "mlp.down_proj.weight").T), INTER, H_PAD))
        save("L%d_in_norm_i32.bin" % i, pad1(q(1.0 + L.arr(p + "input_layernorm.weight")), H_PAD))
        save("L%d_post_attn_norm_i32.bin" % i, pad1(q(1.0 + L.arr(p + "post_attention_layernorm.weight")), H_PAD))
        save("L%d_pre_ffn_norm_i32.bin" % i, pad1(q(1.0 + L.arr(p + "pre_feedforward_layernorm.weight")), H_PAD))
        save("L%d_post_ffn_norm_i32.bin" % i, pad1(q(1.0 + L.arr(p + "post_feedforward_layernorm.weight")), H_PAD))
        save("L%d_q_norm_i32.bin" % i, pad1(q(1.0 + L.arr(p + "self_attn.q_norm.weight")), HDIM))
        save("L%d_k_norm_i32.bin" % i, pad1(q(1.0 + L.arr(p + "self_attn.k_norm.weight")), HDIM))

    save("final_norm_i32.bin", pad1(q(1.0 + L.arr("model.norm.weight")), H_PAD))

    for base, tag in ((ROPE_LOCAL, "local"), (ROPE_GLOBAL, "global")):
        pos = np.arange(SEQ, dtype=np.float64)[:, None]
        i = np.arange(HDIM // 2, dtype=np.float64)[None, :]
        theta = pos * np.power(base, -2.0 * i / HDIM)
        save("rope_cos_%s_i32.bin" % tag, q(np.cos(theta)))
        save("rope_sin_%s_i32.bin" % tag, q(np.sin(theta)))

    logits = reference_forward(L, emb, input_ids)
    argmax = np.argmax(logits, axis=1).astype(np.int32)
    save("gt_argmax_i32.bin", argmax.reshape(-1, 1))
    np.asarray(logits, dtype=np.float32).tofile(os.path.join(OUT, "gt_logits_f32.bin"))
    print("  ground-truth argmax:", argmax.tolist())


def reference_forward(L, emb, input_ids):
    x = emb[input_ids] * math.sqrt(H)

    def rope_tables(base):
        pos = np.arange(SEQ, dtype=np.float64)[:, None]
        i = np.arange(HDIM // 2, dtype=np.float64)[None, :]
        th = pos * np.power(base, -2.0 * i / HDIM)
        return np.cos(th), np.sin(th)

    cos_local, sin_local = rope_tables(ROPE_LOCAL)
    cos_global, sin_global = rope_tables(ROPE_GLOBAL)

    for i in range(LAYERS):
        p = "model.layers.%d." % i
        lt = LAYER_TYPES[i]
        in_norm = 1.0 + L.arr(p + "input_layernorm.weight").astype(np.float64)
        post_attn_norm = 1.0 + L.arr(p + "post_attention_layernorm.weight").astype(np.float64)
        pre_ffn_norm = 1.0 + L.arr(p + "pre_feedforward_layernorm.weight").astype(np.float64)
        post_ffn_norm = 1.0 + L.arr(p + "post_feedforward_layernorm.weight").astype(np.float64)
        q_w = L.arr(p + "self_attn.q_proj.weight").astype(np.float64)  # [1024, 640]
        k_w = L.arr(p + "self_attn.k_proj.weight").astype(np.float64)  # [256, 640]
        v_w = L.arr(p + "self_attn.v_proj.weight").astype(np.float64)  # [256, 640]
        o_w = L.arr(p + "self_attn.o_proj.weight").astype(np.float64)  # [640, 1024]
        q_norm = 1.0 + L.arr(p + "self_attn.q_norm.weight").astype(np.float64)
        k_norm = 1.0 + L.arr(p + "self_attn.k_norm.weight").astype(np.float64)
        gate_w = L.arr(p + "mlp.gate_proj.weight").astype(np.float64)  # [2048, 640]
        up_w = L.arr(p + "mlp.up_proj.weight").astype(np.float64)
        down_w = L.arr(p + "mlp.down_proj.weight").astype(np.float64)  # [640, 2048]

        residual = x
        h = rms_norm(x, in_norm)
        qq = h @ q_w.T  # [seq, 1024]
        kk = h @ k_w.T  # [seq, 256]
        vv = h @ v_w.T  # [seq, 256]
        qq = rms_norm(qq.reshape(SEQ, HEADS, HDIM), q_norm)
        kk = rms_norm(kk.reshape(SEQ, KV_HEADS, HDIM), k_norm)
        cos, sin = (cos_global, sin_global) if lt == "full_attention" else (cos_local, sin_local)
        qq = apply_rope(qq, cos, sin)  # [seq, 4, 256]
        kk = apply_rope(kk, cos, sin)  # [seq, 1, 256]
        inv_sqrt = 1.0 / math.sqrt(256.0)
        attn_heads = []
        for hh in range(HEADS):
            scores = (qq[:, hh, :] @ kk[:, 0, :].T) * inv_sqrt  # [seq, seq]
            mask = np.triu(np.full((SEQ, SEQ), -1e30), k=1)
            scores = scores + mask
            probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            attn_heads.append(probs @ vv)  # [seq, 256]
        attn = np.concatenate(attn_heads, axis=-1) @ o_w.T  # [seq, 640]
        attn = rms_norm(attn, post_attn_norm)
        x = residual + attn

        residual = x
        h2 = rms_norm(x, pre_ffn_norm)
        gate = h2 @ gate_w.T
        up = h2 @ up_w.T
        act = 0.5 * gate * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate ** 3))) * up
        down = act @ down_w.T
        down = rms_norm(down, post_ffn_norm)
        x = residual + down

    x = rms_norm(x, 1.0 + L.arr("model.norm.weight").astype(np.float64))
    return x @ emb.T  # [seq, vocab]


if __name__ == "__main__":
    main()
