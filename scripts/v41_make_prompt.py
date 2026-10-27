#!/usr/bin/env python3
"""Regenerate the DeepSeek-V4.1-Flash prompt embedding + engram hash ids for a
real N-token prompt (no BOS padding), replacing the earlier 3-token + BOS pad
data that made the model amplify repeated BOS positions past Q16 range."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/data/zkIE/scripts")
from v41_ref_cpu import bf16_vec, read_safetensors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/data/zkIE/models/deepseek-v4.1-flash/raw")
    ap.add_argument("--tokenizer", default="/data/zkIE/models/deepseek-v4.1-flash/raw/tokenizer.json")
    ap.add_argument("--prompt", default="The sky is blue and the sun is shining today over the hills and valleys nearby")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--out", default="/data/zkIE/models/deepseek-v4.1-flash/cache/engram")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt).ids
    print("encoded", len(ids), ids)
    assert len(ids) == args.seq, f"prompt must encode to exactly {args.seq} tokens, got {len(ids)}"

    e = read_safetensors(f"{args.raw}/model-00002-of-00048.safetensors")
    w = bf16_vec(e["embed.weight"]).reshape(-1, 5120)
    # Fill seq rows: first `seq` are the real prompt, the rest (up to 512 for
    # future seq512 runs) are BOS so seq16 never touches them.
    full = [ids[i] if i < len(ids) else 0 for i in range(512)]
    emb = np.round(w[full] * 65536.0).astype(np.int32)
    os.makedirs(args.out, exist_ok=True)
    emb.tofile(f"{args.out}/embed_prompt_i32.bin")
    np.asarray(full, dtype=np.int32).tofile(f"{args.out}/prompt_ids_i32.bin")
    print("wrote", f"{args.out}/embed_prompt_i32.bin", emb.shape, "first ids", ids)


if __name__ == "__main__":
    main()
