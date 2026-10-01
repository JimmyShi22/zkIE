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

## Step 3 - Gemma 3 270M (in progress)

Same decoder-only Transformer family, adds Rotary Position Embedding (RoPE)
and RMSNorm. RMSNorm is already covered (TimesFM's `input_layernorm`); the new
piece is RoPE.

Completion includes RoPE in fixed point (sin/cos lookup + rotary), because the
same RoPE is reused by DeepSeek-V2-Lite's MLA (decoupled RoPE). This must not be
deferred to the MLA step.

Gemma 3 is the reuse target: the same op primitives and the same autotune flow.

## Step 4 - DeepSeek-V2-Lite (MoE + MLA, next)

Target: **DeepSeek-V2-Lite** (15.7B total / 2.4B active, 27 layers, hidden 2048,
MLA + DeepSeekMoE: 2 shared experts + 64 routed experts, top-6).

New capabilities to prove:

1. **MoE expert routing (top-k)** - a variant of the existing `GeluIndex` /
   `Lookup` data-dependent-index pattern, plus a compare/select primitive
   (range-check `s_j <= threshold` for the non-selected experts, O(num-experts)).
   Not an architectural breakthrough.
2. **Full commitment + lazy opening of expert weights** - one commitment tree
   over all experts; the prover loads/opens only the routed experts after
   routing. Memory scales with *active* parameters, not total. This is the only
   lever that lets MoE scale.
3. **op list built per-forward from the route** - the static chain becomes a
   data-dependent chain. Since a proof is already for one concrete forward and
   the verifier only checks the proof (does not re-run), this only changes the
   build flow, not the proof abstraction.

Expected order of magnitude (extrapolated from current calibration):
prove ~10-14 min (seq=512), peak memory ~50-150 GB (fully-resident commitment
~150 GB, lazy-open ~50 GB).

## Step 5 - DeepSeek-V4.1-Flash (phased, reference point)

Target: **DeepSeek-V4.1-Flash** (552B total / 8B active prefill, 16B decode,
40-layer causal encoder-decoder, CSA2 sparse attention, multimodal, 1M context,
MIT, 2026-09).

Key judgement: it is **memory/commitment-bound, not compute-bound**. Relative
to V2-Lite, active parameters rise only ~3.3x (8B / 2.4B), but the full
commitment rises ~35x (552B / 15.7B).

- Time: ~22-30 min prove at seq=512; ~3-4 h at 4K; 1M context is currently
  infeasible.
- Memory: full commitment ~2.2 TB (552B x 4B i32) + ~32 GB active experts.
  This is the main wall: it needs 2 TB-class memory + streaming/lazy loading.

Three phases - do not attempt full-scale at once:

- **Phase A: MoE/FFN** - run the 552B expert layers on the lazy-commitment
  machinery above, proving "large total params + small active params" is
  workable. Most likely to land first.
- **Phase B: CSA2 sparse attention** - Full / Reindex / Reuse modes + a
  hierarchical sparse indexer; a new attention proof (the indexer itself must
  be proven). Scale-independent, designed separately.
- **Phase C: vision encoder + fp8** - a multimodal op family + 8-bit
  fixed-point fidelity.

One-line summary: **the DeepSeek path is: first land MoE via V2-Lite (routing +
lazy expert commitment), then split V4.1-Flash into MoE / CSA2 / vision phases.
Large total parameters are absorbed by lazy loading; the real hard parts are
CSA2 and vision - two new op classes that are scale-independent.**
