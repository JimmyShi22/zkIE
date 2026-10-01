#!/usr/bin/env python3
"""Extract DeepSeek-V2-Lite weights from sharded safetensors into padded int32.

Needs numpy (Python 3.11). Reads the 4 bf16 shards, quantizes at 2^16,
transposes HF [out, in] -> [in, out], zero-pads each dim to a power of two, and
writes .bin files. Also writes the RoPE cos/sin tables, a SiLU table, and the
precomputed embedded input for a fixed input sequence.

Weight layout is `[in, out]` (matches the existing GPT-2/TimesFM/Gemma
extractors). MoE routed experts are batched as [64, in, out] per layer.
"""
import json
import math
import os
import struct

import numpy as np

RAW = "models/deepseek-v2-lite/raw"
OUT = "models/deepseek-v2-lite/weights"

SEQ = int(os.environ.get("DS_SEQ", "16"))
SCALE = 65536.0
H = 2048
VOCAB = 102400
VOCAB_PAD = 131072
LAYERS = 27
HEADS = 16
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
Q_DIM = HEADS * (QK_NOPE + QK_ROPE)  # 3072
Q_PAD = 4096
KVA = 512 + QK_ROPE  # 576
KVA_PAD = 1024
KVB = HEADS * (QK_NOPE + V_HEAD)  # 4096
O_DIM = HEADS * V_HEAD  # 2048
KV_LORA = 512
N_ROUTED = 64
N_SHARED = 2
TOPK = 6
MOE_INTER = 1408
MOE_PAD = 2048
DENSE_INTER = 10944
DENSE_PAD = 16384
SHARED_INTER = 2816
SHARED_PAD = 4096
ROPE_BASE = 10000.0


def next_pow2(n):
    return 1 << (n - 1).bit_length()


class Sharded:
    def __init__(self):
        self.data = []
        self.header = {}
        for i in range(1, 5):
            p = "%s/model-0000%d-of-000004.safetensors" % (RAW, i)
            with open(p, "rb") as f:
                d = f.read()
            hlen = struct.unpack("<Q", d[:8])[0]
            h = json.loads(d[8:8 + hlen].decode())
            start = 8 + hlen
            self.data.append((d, start))
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                v = dict(v)
                v["_shard"] = i - 1
                self.header[k] = v

    def arr(self, name):
        v = self.header[name]
        d, start = self.data[v["_shard"]]
        s, e = v["data_offsets"]
        raw = d[start + s:start + e]
        if v["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2")
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return f32.reshape(v["shape"])
        raise ValueError("unsupported dtype %s" % v["dtype"])


def q(x):
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * SCALE), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)


def pad2(a, r, c):
    o = np.zeros((r, c), dtype=np.int32)
    a = np.asarray(a)
    o[: a.shape[0], : a.shape[1]] = a
    return o


def pad1(a, n):
    o = np.zeros(n, dtype=np.int32)
    a = np.asarray(a)
    o[: a.size] = a.reshape(-1)
    return o


def save(name, arr):
    np.asarray(arr, dtype=np.int32).tofile(os.path.join(OUT, name))


def main():
    os.makedirs(OUT, exist_ok=True)
    S = Sharded()

    emb = S.arr("model.embed_tokens.weight").astype(np.float64)  # [vocab, hidden]
    rng = np.random.default_rng(0)
    input_ids = rng.integers(1, VOCAB, (SEQ,)).astype(np.int64)
    x0 = pad2(q(emb[input_ids]), SEQ, H)
    save("x0_i32.bin", x0)
    save("lm_head_i32.bin", pad2(q(S.arr("lm_head.weight").T), H, VOCAB_PAD))
    save("final_norm_i32.bin", pad1(q(S.arr("model.norm.weight")), H))

    for L in range(LAYERS):
        p = "model.layers.%d." % L
        q_w = S.arr(p + "self_attn.q_proj.weight").astype(np.float64).reshape(HEADS, QK_NOPE + QK_ROPE, H)
        q_nope = q_w[:, :QK_NOPE, :].reshape(HEADS * QK_NOPE, H)
        q_pe = q_w[:, QK_NOPE:, :].reshape(HEADS * QK_ROPE, H)
        save("L%d_qnope_w_i32.bin" % L, pad2(q(q_nope.T), H, HEADS * QK_NOPE))
        save("L%d_qpe_w_i32.bin" % L, pad2(q(q_pe.T), H, HEADS * QK_ROPE))
        kva = S.arr(p + "self_attn.kv_a_proj_with_mqa.weight").astype(np.float64)
        kv_lora = kva[:KV_LORA]
        k_pe = kva[KV_LORA:KV_LORA + QK_ROPE]
        save("L%d_kvlora_w_i32.bin" % L, pad2(q(kv_lora.T), H, KV_LORA))
        save("L%d_kpe_w_i32.bin" % L, pad2(q(k_pe.T), H, QK_ROPE))
        save("L%d_kva_ln_i32.bin" % L, pad1(q(S.arr(p + "self_attn.kv_a_layernorm.weight")), KV_LORA))
        kvb = S.arr(p + "self_attn.kv_b_proj.weight").astype(np.float64).reshape(HEADS, QK_NOPE + V_HEAD, KV_LORA)
        k_nope = kvb[:, :QK_NOPE, :].reshape(HEADS * QK_NOPE, KV_LORA)
        v = kvb[:, QK_NOPE:, :].reshape(HEADS * V_HEAD, KV_LORA)
        save("L%d_knope_w_i32.bin" % L, pad2(q(k_nope.T), KV_LORA, HEADS * QK_NOPE))
        save("L%d_v_w_i32.bin" % L, pad2(q(v.T), KV_LORA, HEADS * V_HEAD))
        save("L%d_o_w_i32.bin" % L, pad2(q(S.arr(p + "self_attn.o_proj.weight").T), O_DIM, H))
        save("L%d_in_norm_i32.bin" % L, pad1(q(S.arr(p + "input_layernorm.weight")), H))
        save("L%d_post_attn_norm_i32.bin" % L, pad1(q(S.arr(p + "post_attention_layernorm.weight")), H))

        if L == 0:
            # dense FFN layer
            save("L0_gate_w_i32.bin", pad2(q(S.arr(p + "mlp.gate_proj.weight").T), H, DENSE_PAD))
            save("L0_up_w_i32.bin", pad2(q(S.arr(p + "mlp.up_proj.weight").T), H, DENSE_PAD))
            save("L0_down_w_i32.bin", pad2(q(S.arr(p + "mlp.down_proj.weight").T), DENSE_PAD, H))
        else:
            # MoE layer
            save("L%d_router_i32.bin" % L, pad2(q(S.arr(p + "mlp.gate.weight").T), H, N_ROUTED))
            save("L%d_shared_gate_i32.bin" % L, pad2(q(S.arr(p + "mlp.shared_experts.gate_proj.weight").T), H, SHARED_PAD))
            save("L%d_shared_up_i32.bin" % L, pad2(q(S.arr(p + "mlp.shared_experts.up_proj.weight").T), H, SHARED_PAD))
            save("L%d_shared_down_i32.bin" % L, pad2(q(S.arr(p + "mlp.shared_experts.down_proj.weight").T), SHARED_PAD, H))
            eg = np.zeros((N_ROUTED, H, MOE_PAD), dtype=np.int32)
            eu = np.zeros((N_ROUTED, H, MOE_PAD), dtype=np.int32)
            ed = np.zeros((N_ROUTED, MOE_PAD, H), dtype=np.int32)
            for e in range(N_ROUTED):
                ep = p + "mlp.experts.%d." % e
                eg[e] = pad2(q(S.arr(ep + "gate_proj.weight").T), H, MOE_PAD)
                eu[e] = pad2(q(S.arr(ep + "up_proj.weight").T), H, MOE_PAD)
                ed[e] = pad2(q(S.arr(ep + "down_proj.weight").T), MOE_PAD, H)
            save("L%d_experts_gate_i32.bin" % L, eg)
            save("L%d_experts_up_i32.bin" % L, eu)
            save("L%d_experts_down_i32.bin" % L, ed)

    # RoPE tables (base 10000, rope head dim 64)
    pos = np.arange(SEQ, dtype=np.float64)[:, None]
    i = np.arange(QK_ROPE // 2, dtype=np.float64)[None, :]
    theta = pos * np.power(ROPE_BASE, -2.0 * i / QK_ROPE)
    save("rope_cos_i32.bin", q(np.cos(theta)))
    save("rope_sin_i32.bin", q(np.sin(theta)))

    # SiLU table (for SwiGLU): x / (1 + exp(-x)), 2^16 scale, offset 2^23, size 2^24
    GELU_OFFSET = 1 << 23
    GELU_SIZE = 1 << 24
    xs = (np.arange(GELU_SIZE, dtype=np.float64) - GELU_OFFSET) / SCALE
    silu = xs / (1.0 + np.exp(-xs))
    np.clip(np.round(silu * SCALE), -(2 ** 31), 2 ** 31 - 1).astype(np.int32).tofile(os.path.join(OUT, "silu_table_i32.bin"))

    print("extraction done; x0", x0.shape, "lm_head [%d,%d]" % (H, VOCAB_PAD))


if __name__ == "__main__":
    main()
