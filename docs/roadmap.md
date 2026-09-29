# Roadmap

Model progression for the GKR/WHIR proving stack.

## Step 1 - TimesFM 1.0 200M (done)

Full forward pass (prologue + 20 layers + output head) proves end-to-end
through the unified op interface. See `docs/benchmarks.md`.

## Step 2 - GPT-2 124M (done)

The smallest standard decoder-only Transformer. Open (MIT), single checkpoint,
and a good first like-for-like target for the proving stack.

Op deltas vs TimesFM:

- FFN activation is GELU instead of ReLU/SiLU. GELU is a table lookup, already
  covered by `gelu.rs` plus the generic `lookup` op.
- Causal self-attention uses a single fused QKV projection split into heads,
  then concatenated back - a different head layout than TimesFM's per-head
  Q/K/V split.
- LayerNorm (already supported), no RevIN prologue.
- Token + positional embeddings instead of TimesFM's RevIN cat.

Main new risk is long-sequence attention (`QK^T` is O(seq^2) and dominates as
the context grows). Proven at seq=16 and seq=512; the seq=512 causal mask must
be deepened (see `docs/benchmarks.md`).

## Step 3 - Gemma 3 270M (current)

Same decoder-only Transformer family, adds Rotary Position Embedding (RoPE)
and RMSNorm.
