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

## Step 3 - Gemma 3 270M (next)

Same decoder-only Transformer family, adds Rotary Position Embedding (RoPE)
and RMSNorm. RMSNorm is already covered (TimesFM's `input_layernorm`); the new
piece is RoPE.

Completion includes RoPE in fixed point (sin/cos lookup + rotary), because the
same RoPE is reused by DeepSeek-V2-Lite's MLA (decoupled RoPE). This must not be
deferred to the MLA step.

Gemma 3 is the reuse target: the same op primitives and the same autotune flow.

## Step 4 - DeepSeek-V2-Lite (after Gemma 3)

15.7B total / 2.4B active, 27 layers, hidden 2048. Architecture = MLA
(Multi-head Latent Attention) + DeepSeekMoE (2 shared experts + 64 routed
experts, top-6).

The smallest model that exercises both DeepSeek signatures (MLA + MoE) at a
scale this machine can run. New capabilities to prove:

- MoE expert routing (top-k).
- Full commitment of expert weights + selective opening of only the routed
  experts (prover loads active experts; memory scales with active, not total,
  parameters).
- op list built per-forward from the route (a data-dependent chain, not a
  static compile-time chain).

Landing order: first a small MoE (e.g. Qwen3-30B-A3B) to prove the routing +
selective-loading skeleton and measure the O(num-experts) top-k comparison
cost, then DeepSeek-V2-Lite to add MLA. MLA's latent-KV compression is standard
matmul; its only new piece is decoupled RoPE, already proven in Step 3
(Gemma 3).

After this, the "MoE commitment + lazy open" mechanism generalizes, and
DeepSeek-V3/R1 full-scale becomes reachable on larger hardware.
