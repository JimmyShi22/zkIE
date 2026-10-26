#!/usr/bin/env python3
"""Extract DeepSeek-V4.1-Flash Engram hash constants and hash ids.

Reimplements the deterministic `engram.py` hash (pure Python: no sympy, so the
prime sieve is a small Miller-Rabin) and dumps the constants the Rust builder
needs: token_map, per-layer multipliers, per-(layer,ngram,head) bucket primes,
offsets, and the hash ids for the default prompt.

Usage:
  python3 scripts/extract_v41_engram.py \
    --tokenizer /data/zkIE/models/deepseek-v4.1-flash/raw/tokenizer.json \
    --config /data/zkIE/models/deepseek-v4.1-flash/raw/config.json \
    --out /data/zkIE/models/deepseek-v4.1-flash/cache/engram \
    --prompt "The sky is" --seq 16
"""

import argparse
import json
import math

import numpy as np


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start, seen):
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


def build_compressed_token_map(tokenizer):
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    norm = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_new = {}
    lookup = [0] * tokenizer.get_vocab_size()
    for token_id in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = norm.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The sky is")
    ap.add_argument("--seq", type=int, default=16)
    args = ap.parse_args()

    cfg = json.load(open(args.config))["text_config"]
    layer_ids = tuple(cfg["engram_layer_ids"])
    max_ngram = cfg["engram_max_ngram_size"]
    n_heads = cfg["engram_n_heads"]
    vocab_size = cfg["engram_compressed_vocab_size"]
    pad_id = cfg["engram_pad_token_id"]

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    token_map, cvocab = build_compressed_token_map(tok)
    assert cvocab == vocab_size, (cvocab, vocab_size)

    # Primes per (layer, ngram, head), and per-layer offsets.
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))

    flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
    offsets = [np.cumsum([0, *sizes[:-1]]).tolist() for sizes in flat]

    max_long = np.iinfo(np.int64).max
    mult_bound = max(1, (max_long // vocab_size) // 2)
    multipliers = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        values = gen.integers(0, mult_bound, size=(max_ngram,), dtype=np.int64)
        multipliers.append((values * 2 + 1).tolist())

    # Hash ids for the prompt.
    ids = tok.encode(args.prompt).ids
    ids = (ids + [0] * args.seq)[: args.seq]
    compressed = [token_map[t] for t in ids]
    pad_compressed = token_map[pad_id]

    n_layers = len(layer_ids)
    hash_ids = []
    for pos in range(len(ids)):
        row = []
        for li in range(n_layers):
            # rolling xor over lookback 0..max_ngram; each (ngram, head) bucket
            # uses the running xor modulo that head's prime, then adds offset.
            rolling = compressed[pos] * multipliers[li][0]
            hashes_per_ngram = []
            for i in range(1, max_ngram):
                src = compressed[pos - i] if pos >= i else pad_compressed
                rolling = rolling ^ (src * multipliers[li][i])
                hashes_per_ngram.append(
                    [rolling % primes[li][i - 1][h] for h in range(n_heads)]
                )
            for ng in range(max_ngram - 1):
                for h in range(n_heads):
                    row.append(hashes_per_ngram[ng][h] + offsets[li][ng * n_heads + h])
        hash_ids.append(row)

    out = args.out
    import os

    os.makedirs(out, exist_ok=True)
    np.asarray(token_map, dtype=np.int32).tofile(f"{out}/token_map_i32.bin")
    np.asarray(multipliers, dtype=np.int64).tofile(f"{out}/multipliers_i64.bin")
    np.asarray(primes, dtype=np.int64).tofile(f"{out}/primes_i64.bin")
    np.asarray(offsets, dtype=np.int64).tofile(f"{out}/offsets_i64.bin")
    np.asarray(hash_ids, dtype=np.int32).tofile(f"{out}/hash_ids_i32.bin")
    print("token_map_len", len(token_map))
    print("multipliers", multipliers)
    print("primes", primes)
    print("offsets", offsets)
    print("prompt_ids", ids)
    print("hash_ids[0]", hash_ids[0])


if __name__ == "__main__":
    main()
