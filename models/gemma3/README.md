# Gemma 3 270M

`zkie-models-gemma3` builds the Gemma 3 270M forward pass as a shard-DAG of op
primitives (RMSNorm, RoPE, GQA attention, gated-GELU MLP) and proves it with
the `zkie-ops` shard composer.

## Weights

The weights are gated on Hugging Face (`google/gemma-3-270m`, bf16 safetensors,
~536 MB). They are not committed to the repo.

1. Accept the Gemma license and set `HF_TOKEN`.
2. Download the checkpoint:

   ```sh
   mkdir -p models/gemma3/raw
   curl -sSL -H "Authorization: Bearer $HF_TOKEN" \
     -o models/gemma3/raw/model.safetensors \
     https://huggingface.co/google/gemma-3-270m/resolve/main/model.safetensors
   ```

3. Extract fixed-point weights (needs `numpy`, Python 3.11):

   ```sh
   /usr/bin/python3.11 scripts/extract_gemma3.py
   ```

   This writes the padded int32 `.bin` weights, the RoPE cos/sin tables, the
   precomputed embedded input, and the ground-truth argmax/logits under
   `models/gemma3/weights/`.

   sha256 (`model.safetensors`): recorded at download time (see
   `scripts/extract_gemma3.py`).

## Prove / benchmark

```sh
GEMMA_SEQ=16 cargo run --release -p zkie-models-gemma3 --example prove
GEMMA_SEQ=16 cargo run --release -p zkie-models-gemma3 --example bench
```
Measured (seq=16, release, 64-thread CPU):

| shards | prove | verify | total | argmax | peak RSS |
| --- | --- | --- | --- | --- | --- |
| 22 (per layer) | 30.4 s | 15.8 s | ~46.2 s | 16/16 | ~28 GB |
