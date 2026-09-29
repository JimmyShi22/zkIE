#!/usr/bin/env python3
"""Dump GPT-2 124M weights/tables for the GKR prover (mirrors TimesFM extract).

Writes models/gpt2_stack/*.i32.bin: per-layer weights/biases (padded to power
of two), the precomputed embedding for a fixed input, the causal mask, and the
gelu_new lookup table. Also runs onnxruntime to save ground-truth logits and
their argmax for the prover's validation.

Run in Python >= 3.10 with onnx, onnxruntime, numpy installed.
"""
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(MODELS, "gpt2_stack")
os.makedirs(OUT, exist_ok=True)

SCALE = 65536.0
H = 768
H_PAD = 1024
FFN = 3072
FFN_PAD = 4096
VOCAB = 50257
VOCAB_PAD = 65536
SEQ = int(os.environ.get("GPT2_SEQ", "16"))
HEADS = 12
HDIM = 64
N_LAYERS = 12


def q(arr):
    return np.clip(np.round(np.asarray(arr, dtype=np.float64) * SCALE), -(2**31), 2**31 - 1).astype(np.int32)


def pad1(arr, n):
    o = np.zeros(n, dtype=np.int32)
    o[: arr.size] = arr
    return o


def pad2(arr, r, c):
    o = np.zeros((r, c), dtype=np.int32)
    a = np.asarray(arr)
    o[: a.shape[0], : a.shape[1]] = a
    return o


def save(name, arr):
    path = os.path.join(OUT, name)
    np.asarray(arr, dtype=np.int32).tofile(path)
    print("  wrote %-28s %s" % (name, list(np.asarray(arr).shape)))


def main():
    m = onnx.load(os.path.join(MODELS, "gpt2.onnx"))
    inits = {}
    for i in m.graph.initializer:
        inits[i.name] = numpy_helper.to_array(i).astype(np.float64)

    wte = inits["model.lm_head.weight"]  # [50257, 768] (tied with wte)
    wpe = inits["model.transformer.wpe.weight"]  # [1024, 768]

    # fixed public input
    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, VOCAB, (1, SEQ)).astype(np.int64)
    attention_mask = np.ones((1, SEQ), dtype=np.int64)
    position_ids = np.arange(SEQ, dtype=np.int64).reshape(1, SEQ)

    # precomputed embedding = wte[input_ids] + wpe[0:SEQ]
    token_emb = wte[input_ids[0]]  # [SEQ, 768]
    pos_emb = wpe[:SEQ]  # [SEQ, 768]
    hidden = token_emb + pos_emb  # [SEQ, 768]
    save("embedding_i32.bin", pad2(q(hidden), SEQ, H_PAD))

    # causal mask (additive, -2^21 on strictly-upper triangle)
    mask = np.triu(np.full((SEQ, SEQ), -(1 << 30)), k=1).astype(np.int32)
    save("mask_i32.bin", mask)

    # per-layer weights
    for h in range(N_LAYERS):
        p = f"model.transformer.h.{h}."
        save(f"L{h}_ln1_w_i32.bin", pad1(q(inits[p + "ln_1.weight"]), H_PAD))
        save(f"L{h}_ln1_b_i32.bin", pad1(q(inits[p + "ln_1.bias"]), H_PAD))
        c_attn_w = inits[p + "attn.c_attn.weight"]  # [768, 2304]
        c_attn_b = inits[p + "attn.c_attn.bias"]    # [2304]
        save(f"L{h}_q_w_i32.bin", pad2(q(c_attn_w[:, 0:H]), H_PAD, H_PAD))
        save(f"L{h}_k_w_i32.bin", pad2(q(c_attn_w[:, H:2 * H]), H_PAD, H_PAD))
        save(f"L{h}_v_w_i32.bin", pad2(q(c_attn_w[:, 2 * H:3 * H]), H_PAD, H_PAD))
        save(f"L{h}_q_b_i32.bin", pad1(q(c_attn_b[0:H]), H_PAD))
        save(f"L{h}_k_b_i32.bin", pad1(q(c_attn_b[H:2 * H]), H_PAD))
        save(f"L{h}_v_b_i32.bin", pad1(q(c_attn_b[2 * H:3 * H]), H_PAD))
        save(f"L{h}_o_proj_w_i32.bin", pad2(q(inits[p + "attn.c_proj.weight"]), H_PAD, H_PAD))
        save(f"L{h}_o_proj_b_i32.bin", pad1(q(inits[p + "attn.c_proj.bias"]), H_PAD))
        save(f"L{h}_ln2_w_i32.bin", pad1(q(inits[p + "ln_2.weight"]), H_PAD))
        save(f"L{h}_ln2_b_i32.bin", pad1(q(inits[p + "ln_2.bias"]), H_PAD))
        save(f"L{h}_fc_w_i32.bin", pad2(q(inits[p + "mlp.c_fc.weight"]), H_PAD, FFN_PAD))
        save(f"L{h}_fc_b_i32.bin", pad1(q(inits[p + "mlp.c_fc.bias"]), FFN_PAD))
        save(f"L{h}_proj_w_i32.bin", pad2(q(inits[p + "mlp.c_proj.weight"]), FFN_PAD, H_PAD))
        save(f"L{h}_proj_b_i32.bin", pad1(q(inits[p + "mlp.c_proj.bias"]), H_PAD))

    # final LN + LM head (transposed to [hidden, vocab])
    save("ln_f_w_i32.bin", pad1(q(inits["model.transformer.ln_f.weight"]), H_PAD))
    save("ln_f_b_i32.bin", pad1(q(inits["model.transformer.ln_f.bias"]), H_PAD))
    save("lm_head_w_i32.bin", pad2(q(wte.T), H_PAD, VOCAB_PAD))

    # gelu_new lookup table: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    GELU_OFFSET = 1 << 23
    GELU_SIZE = 1 << 24
    idx = np.arange(GELU_SIZE, dtype=np.float64)
    x = (idx - GELU_OFFSET) / SCALE
    gelu = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    gelu_table = np.clip(np.round(gelu * SCALE), -(2**31), 2**31 - 1).astype(np.int32)
    gelu_table.tofile(os.path.join(OUT, "gelu_table_i32.bin"))
    print("  wrote %-28s %s" % ("gelu_table_i32.bin", list(gelu_table.shape)))

    # ground truth via onnxruntime
    sess = ort.InferenceSession(os.path.join(MODELS, "gpt2.onnx"))
    logits = sess.run(["logits"], {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })[0]  # [1, 16, 50257]
    gt = logits[0]  # [16, 50257]
    gt_argmax = gt.argmax(axis=1).astype(np.int32)  # [16]
    np.save(os.path.join(OUT, "gt_logits_f32.npy"), logits.astype(np.float32))
    gt_argmax.tofile(os.path.join(OUT, "gt_argmax_i32.bin"))
    print("ground truth logits shape", logits.shape)
    print("ground truth argmax:", gt_argmax.tolist())
    print("input_ids:", input_ids[0].tolist())


if __name__ == "__main__":
    main()