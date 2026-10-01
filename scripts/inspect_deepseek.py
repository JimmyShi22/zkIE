#!/usr/bin/env python3
import json
import struct

RAW = "models/deepseek-v2-lite/raw"


def shard_headers():
    out = {}
    for i in range(1, 5):
        p = "%s/model-0000%d-of-000004.safetensors" % (RAW, i)
        with open(p, "rb") as f:
            d = f.read()
        hlen = struct.unpack("<Q", d[:8])[0]
        h = json.loads(d[8:8 + hlen].decode())
        for k, v in h.items():
            if k == "__metadata__":
                continue
            out[k] = v["shape"]
    return out


def main():
    shapes = shard_headers()
    print("tensor count:", len(shapes))

    def show(pat):
        for k in sorted(shapes):
            if pat in k:
                print("  %-55s %s" % (k, shapes[k]))

    print("== embeddings / norm / lm_head ==")
    for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        print("  %-55s %s" % (k, shapes[k]))

    print("== layer 0 (dense) ==")
    for k in sorted(shapes):
        if "layers.0." in k:
            print("  %-55s %s" % (k, shapes[k]))

    print("== layer 1 (MoE) attention ==")
    for k in sorted(shapes):
        if "layers.1.self_attn" in k or "layers.1.input_layernorm" in k or "layers.1.post_attention_layernorm" in k:
            print("  %-55s %s" % (k, shapes[k]))

    print("== layer 1 (MoE) router + experts ==")
    for k in sorted(shapes):
        if "layers.1.mlp.gate.weight" in k or "layers.1.mlp.shared_experts" in k or "layers.1.mlp.experts.0." in k or "layers.1.mlp.experts.1." in k:
            print("  %-55s %s" % (k, shapes[k]))


if __name__ == "__main__":
    main()
