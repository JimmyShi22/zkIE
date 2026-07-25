# zkIE Sub-Project 4 — Scope Decision: TimesFM Integration + EVM Verifier

**Date**: 2026-07-26
**Status**: Decided (no user available to confirm in real time — session running autonomously overnight per explicit instruction; flagged here for review)

## Why this needed a real decision, not just "keep building"

Research (via web search against the official `google-research/timesfm` repo, arXiv paper,
and GitHub issue tracker) surfaced two facts that change what "prove TimesFM end-to-end" can
honestly mean in this environment:

1. **Nobody has ever published a working ONNX export of TimesFM.** Two independent GitHub
   issues asking exactly this ([#252](https://github.com/google-research/timesfm/issues/252),
   [#115](https://github.com/google-research/timesfm/issues/115)) are open with zero replies,
   over a year apart. A real PyTorch implementation of TimesFM 1.0 does exist
   (`v1/src/timesfm/pytorch_patched_decoder.py`, confirmed by reading the source directly),
   and a single forward pass (forecast horizon ≤ 128) has no data-dependent control flow, so
   `torch.onnx.export` should be *applicable in principle* — but this is genuinely unattempted
   territory, not a known-quantity integration.

2. **TimesFM 1.0's real architecture differs from the assumptions in the original concept
   doc**, and its real size makes literal full-model proving computationally infeasible here:
   - 20 transformer layers, hidden dim 1280, per-layer matmuls up to 1280×3840 (qkv
     projection) and 1280×1280 (output/FFN projections) — confirmed from the actual
     `TimesFMConfig` in the source, ~200M params total, matching the paper.
   - Normalization is RMSNorm (attention) + standard LayerNorm (MLP), not pure LayerNorm.
   - FFN activation is **ReLU, not GELU** (contradicts the original design doc's "GELU: FFN
     activation" assumption — `zkie-core`'s `GeluChip` may simply not be needed for TimesFM's
     own FFN blocks; a `ReluChip` doesn't exist yet, since sub-project 2 didn't need it for
     the ISA coverage tackled so far).
   - Attention includes a nonstandard learned per-head softplus-based scaling, not plain
     `1/sqrt(d)`.
   - This project's chip architecture proves *one scalar dot product at a time*
     (`DotProductChip`, `K` rows of accumulation per output element). A single 1280×1280
     matmul over even one token already needs 1280 output elements × 1280 accumulation rows
     ≈ 1.64M circuit rows; a single-layer QKV projection over a realistic multi-patch context
     (e.g. 16 patches × 3840 output dims × 1280 accumulation rows) is already ~79M rows —
     **before** counting all 20 layers or the other 3 matmuls per layer. Real KZG proving at
     that row count is not feasible on a single machine in any reasonable time/memory budget
     with this chip design (this is a known, hard, open problem in ZKML generally — production
     systems use batched/vectorized matmul arguments, GPU proving, or proving clusters, which
     is a distinct research undertaking from what sub-projects 1–3 built).

## Decision

Rather than either (a) silently attempting to build/prove the literal full 200M-parameter
model — which would not finish, or would finish incorrectly after consuming enormous time —
or (b) stopping sub-project 4 entirely for lack of a "clean" path, this sub-project proceeds
in three genuinely-achievable, honestly-scoped pieces:

1. **Circuit assembler** (real, general infrastructure, valuable at any scale): a component
   that takes a `zkie-compiler::CompiledProgram` and dispatches each instruction to the
   corresponding `zkie-core` chip, building an actual working `halo2_proofs::Circuit` —
   proven via real KZG roundtrip tests on small/synthetic programs (including the existing
   linear-layer test program from sub-project 3).
2. **A real, good-faith attempt at exporting the actual TimesFM 1.0-200M PyTorch
   implementation to ONNX**, and compiling that real graph (or as much of it as the current
   op mapper's curated op subset supports) via `zkie-compiler`. This is attempted honestly —
   not guaranteed to succeed, since it is unattempted territory — and the outcome (success,
   partial success, or specific blockers found) will be reported plainly either way. This
   alone is worth doing regardless of the proving step: successfully compiling TimesFM's real
   graph structure (even without generating a full proof) validates the ISA/op-mapper design
   against a real, non-toy model.
3. **A real KZG proof + EVM verifier + Anvil on-chain verification**, demonstrated on a scale
   that actually completes in this environment — e.g. a single TimesFM-shaped
   transformer block (one attention + one FFN layer, real architecture, either a truncated
   slice of the real exported graph or a small hand-built model with identical op structure)
   rather than the full 20-layer, 1280-dim model. This is the same "prove the pipeline is
   real, not a toy" bar sub-projects 1–3 already met, applied at the largest scale that's
   actually provable here — not the full pretrained checkpoint.

**What this explicitly does NOT claim**: this does not produce a real KZG proof of a full
TimesFM 1.0-200M inference at production context lengths. That remains future work requiring
a different (batched/vectorized) circuit architecture — a substantial follow-on research
project in its own right, consistent with the original concept doc's own "Future Directions"
section (GPU-accelerated proving, recursive composition, etc.).
