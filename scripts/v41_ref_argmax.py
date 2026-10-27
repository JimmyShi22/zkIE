#!/usr/bin/env python3
"""Float64 reference for the full 40-layer DeepSeek-V4.1-Flash text forward with
routed MoE + final head, producing greedy argmax ground truth for seq16/seq512.
Matches the Rust builder's fixed-point (Q16) weight semantics."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v41_ref_cpu import bf16_vec, dequant_fp4_weight, dequant_fp8_weight, read_safetensors, rms_norm, rope, sparse_attn
from v41_ref_forward import hc_mixes, hc_pre, hc_post, build_attention, build_compressor, build_shared_expert, build_engram, DIM, HC, N_HEADS, HEAD_DIM, QK, N_EXPERTS, INTER, ROUTE_SCALE, fixed_float

VOCAB = 129280

def sqrt_softplus(x):
    return np.sqrt(np.log1p(np.exp(x)))

def silu(x):
    return x / (1.0 + np.exp(-x))

def build_routed_experts(t, layer, x, m):
    gw = bf16_vec(t[f"layers.{layer}.ffn.gate.weight"]).reshape(N_EXPERTS, DIM)
    z = x @ gw.T
    s = sqrt_softplus(z)
    topk = 6
    sel = np.zeros((m, N_EXPERTS), dtype=np.float64)
    for i in range(m):
        idx = np.argpartition(-s[i], topk)[:topk]
        sel[i, idx] = 1.0
    s_sel = s * sel
    weights = s_sel / s_sel.sum(-1, keepdims=True) * ROUTE_SCALE
    out = np.zeros((m, DIM), dtype=np.float64)
    for e in range(N_EXPERTS):
        tokens = np.where(sel[:, e] == 1.0)[0]
        if len(tokens) == 0:
            continue
        w1 = fixed_float(dequant_fp4_weight(t[f"layers.{layer}.ffn.experts.{e}.w1.weight"], t[f"layers.{layer}.ffn.experts.{e}.w1.scale"], DIM))
        w2 = fixed_float(dequant_fp4_weight(t[f"layers.{layer}.ffn.experts.{e}.w2.weight"], t[f"layers.{layer}.ffn.experts.{e}.w2.scale"], INTER))
        w3 = fixed_float(dequant_fp4_weight(t[f"layers.{layer}.ffn.experts.{e}.w3.weight"], t[f"layers.{layer}.ffn.experts.{e}.w3.scale"], DIM))
        g = silu(x[tokens] @ w1.T)
        u = np.clip(x[tokens] @ w3.T, -10.0, 10.0)
        hidden = g * u
        out[tokens] += weights[tokens, e, None] * (hidden @ w2.T)
    return out

def build_final_head(t, x4, pre_mix, m):
    h = hc_pre(x4, pre_mix)
    norm = bf16_vec(t["norm.weight"])
    h = rms_norm(h, norm)
    head = bf16_vec(t["head.weight"]).reshape(VOCAB, DIM)
    logits = h @ head.T
    return logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--no-routed", action="store_true")
    ap.add_argument("--hash-ids", default=None)
    ap.add_argument("--embed", default=None)
    args = ap.parse_args()

    shards = {l: read_safetensors(f"{args.raw}/model-{l+3:05d}-of-00048.safetensors") for l in range(args.layers)}
    hash_ids = None
    if args.hash_ids:
        hash_ids = np.fromfile(args.hash_ids, dtype=np.int32).reshape(args.m, -1)

    if args.embed:
        x = np.fromfile(args.embed, dtype=np.int32).astype(np.float64).reshape(args.m, DIM) / 65536.0
    else:
        x = np.ones((args.m, DIM), dtype=np.float64)
    x4 = np.repeat(x[:, None, :], HC, axis=1)
    pre_mix = np.zeros((args.m, HC), dtype=np.float64)
    pre_mix[:, 0] = 1.0
    compress_kv = None

    for layer in range(args.layers):
        if hash_ids is not None and layer in (1, 14):
            eng = read_safetensors(f"{args.raw}/model-{47 if layer == 1 else 48:05d}-of-00048.safetensors")
            x4 = build_engram(eng, layer, x4, hash_ids, args.m)
        t = shards[layer]
        ratio = 0 if layer < 2 else (2 if layer <= 19 else 1)
        hc_attn_fn = np.frombuffer(t[f"layers.{layer}.hc_attn_fn"]["data"], dtype="<f4").astype(np.float64).reshape((2+HC)*HC, HC*DIM)
        hc_attn_base = np.frombuffer(t[f"layers.{layer}.hc_attn_base"]["data"], dtype="<f4").astype(np.float64)
        hc_attn_scale = np.frombuffer(t[f"layers.{layer}.hc_attn_scale"]["data"], dtype="<f4").astype(np.float64)
        attn_norm = bf16_vec(t[f"layers.{layer}.attn_norm.weight"])
        attn_pre, attn_post, attn_comb = hc_mixes(x4, hc_attn_fn, hc_attn_scale, hc_attn_base)
        attn_in = rms_norm(hc_pre(x4, pre_mix), attn_norm)
        if layer in (2, 8, 14, 20):
            compress_kv = build_compressor(t, layer, attn_in, args.m, ratio)
        attn_out = build_attention(t, layer, attn_in, args.m, ratio, compress_kv)
        x4 = hc_post(attn_out, x4, attn_post, attn_comb)

        hc_ffn_fn = np.frombuffer(t[f"layers.{layer}.hc_ffn_fn"]["data"], dtype="<f4").astype(np.float64).reshape((2+HC)*HC, HC*DIM)
        hc_ffn_base = np.frombuffer(t[f"layers.{layer}.hc_ffn_base"]["data"], dtype="<f4").astype(np.float64)
        hc_ffn_scale = np.frombuffer(t[f"layers.{layer}.hc_ffn_scale"]["data"], dtype="<f4").astype(np.float64)
        ffn_norm = bf16_vec(t[f"layers.{layer}.ffn_norm.weight"])
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x4, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
        ffn_in = rms_norm(hc_pre(x4, attn_pre), ffn_norm)
        shared_out = build_shared_expert(t, layer, ffn_in)
        routed_out = build_routed_experts(t, layer, ffn_in, args.m) if not args.no_routed else np.zeros_like(shared_out)
        ffn_out = shared_out + routed_out
        x4 = hc_post(ffn_out, x4, ffn_post, ffn_comb)
        pre_mix = ffn_pre
        print(f"layer {layer} done", flush=True)

    head_shard = read_safetensors(f"{args.raw}/model-00043-of-00048.safetensors")
    logits = build_final_head(head_shard, x4, pre_mix, args.m)
    am = np.argmax(logits, axis=-1)
    print("argmax =", am.tolist())
    print("top2_logits =", [sorted(logits[i].tolist())[-2:] for i in range(args.m)])

if __name__ == "__main__":
    main()
