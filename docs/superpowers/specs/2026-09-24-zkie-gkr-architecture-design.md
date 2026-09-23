# zkIE GKR-Route Architecture Design (Draft)

**Date**: 2026-09-24
**Status**: Discussion notes, to be finalized into a formal spec
**Branch**: `feature/gkr`
**Scope**: Covers only the GKR/WHIR proving route. It supersedes/extends the older
Halo2/KZG-route specs rather than patching on top of them.

## 1. Background & Motivation

The `feature/gkr` branch already has a working set of GKR/WHIR proving primitives
(`crates/zkie-gkr`) that can prove the full TimesFM 8M/200M forward pass. But it lacks a
compilation stack down to ONNX and a convergence path up to on-chain verification; meanwhile
every spec under `docs/superpowers/specs/` is written for the Halo2/KZG route and is
disconnected from the GKR route.

This document records the GKR route's target architecture and key decisions, as the basis for
the follow-up specs/plans.

## 2. Core Principles

1. **First-principles, bottom-up redesign.** Everything above GKR (fixed-point semantics,
   program model, compiler, executor) can only borrow ideas from the old Halo2 route, not
   reuse its code — the two differ in fixed-point system, field, and proving unit: the old
   route is I18 (1e18 scale) + BN254 `Fr` + Halo2 chip ISA; the new route is int16/i32/i64
   (2^16/2^32/2^48 scale) + Goldilocks + GKR proving primitives.
2. **Non-ZK.** zkIE needs "prove correctness + keep it succinct", not "hide the witness".
   GKR/sumcheck/WHIR/FRI are inherently non-ZK proof protocols; in public-model mode weights
   and activations are public anyway, and private model is a separate follow-up topic.

## 3. Current State (Have / Missing)

Have:

- GKR proving primitives: `prove_matmul` / `prove_softmax` / `prove_layer_norm` /
  `prove_rms_norm` / `prove_relu` / `prove_affine` / `prove_scale` / `prove_lookup` /
  `prove_add`, etc. (`crates/zkie-gkr/src/committed.rs`).
- WHIR commitment wrapper (`whir.rs`), including prescribed-point opening (`OpeningProtocol` /
  `open_at` / `verify_at`), which is the basis for shard-boundary glue.
- Shard DAG skeleton (`zkie-compiler/src/dag/`: `build_dag` / `Shard` / `EdgeKind` / `Linker` /
  `MockProver`) doing def-use analysis. It is orthogonal to the proof backend; only its
  **ideas** are reusable, not its types.

Missing:

- A single authoritative fixed-point semantics contract (see §5.1).
- A fixed-point program IR + executor (see §5.2).
- One builder per model (see §5.3).
- A real GKR `Prover` (wrapping the primitives behind a pluggable `Prover`), N-ary aggregation,
  and on-chain convergence.

## 4. Target Layered Architecture

Bottom-up:

```text
GKR proving primitives (have)
  ↑ fixed-point semantics contract (missing, the foundation)
  ↑ fixed-point program IR + executor (missing)
  ↑ one builder per model (missing, thin)
  ↑ sharding: autotuning loop (cut → measure → iterate), not a static planner
  ↑ linking across shards: shared commitment (glue, see §5.5)
  ↑ aggregation (missing, recursive composition, added last)
  ↑ on-chain convergence (missing, done last)
```

Sharding/aggregation/scheduling are an orchestration layer, orthogonal to the proof backend;
on-chain convergence is the final layer.

## 5. Key Architectural Decisions

### 5.1 The fixed-point semantics contract must be unified

For every op, the input scale, output scale, internal accumulation scale, rounding rule
(round-half-up vs half-to-even), table-indexing rule, power-of-two padding rule, and overflow
boundary must be written as **one authoritative definition** that the compiler, executor, and
proving primitives all follow. Today these semantics are scattered across `committed.rs`
comments, `simulate_*.py`, and `prove_*.rs`, and already contain a half-up vs half-to-even
inconsistency. This is the first step of the bottom-up design.

### 5.2 Fixed-point program IR + executor

Abstract "a sequence of fixed-point op instructions + register def-use + weight/table
references" into a `Program` data structure, plus a deterministic executor
`run(program, input) -> witness`. This replaces the hand-wired proof chains in the examples and
the hardcoded forward passes in the scripts. It is a thin layer — not an "automatic ONNX
compiler".

### 5.3 One builder per model

`build_timesfm_8m() -> Program`, `build_timesfm_200m() -> Program`, etc., translate model
structure into a `Program`. The effort is roughly the size of the current `simulate_*.py`, but
produces a structured program. Model structure (layer count, where weights live, where QKV is
split) depends only on architecture + export code, so it is written once. "Automatically parse
arbitrary ONNX" is the nice-to-have, deferrable part.

### 5.4 Fixed op → GKR-primitive mapping, no per-shard customization

The "which op maps to which proving primitive" table is fixed (MatMul → `prove_matmul`, etc.),
written once, reused by every model. The GKR/WHIR route has no traditional per-circuit circuit
or keygen (FRI-based, no per-circuit trusted setup), and WHIR parameters are global. Therefore
when sharding is non-fixed, **there is no need to customize a circuit or key per shard** — only
the "op → primitive" mapping is needed; a shard is just a cut in the program, and primitives are
looked up from the mapping. The only per-size configuration is the WHIR instance itself (by
tensor size `2^num_variables`), which is global and sized by tensor, independent of sharding.

### 5.5 Shard linking: internal reduction + cross-shard shared commitment

Two orthogonal choices:

- **Shared commitment**: every tensor is committed once; an op's output commitment equals the
  next op's input commitment. Fully parallel, but memory-heavy (every intermediate tensor is
  committed).
- **GKR layer reduction**: whole layers are reduced by sumcheck; intermediate layers are not
  committed. Memory-light and compact, but sequential across layers.

Adopted hybrid: **GKR reduction inside a shard, shared commitment across shards.** Shard size is
the memory/parallelism trade-off knob: tight memory → larger shards (closer to full-graph GKR);
ample memory → smaller shards (more parallel). The glue is WHIR prescribed-point opening: the
upstream shard's output commitment is opened at a random point → used as the starting point of
the downstream shard's GKR reduction → reduced to its output → committed again.

### 5.6 Sharding is an autotuning loop, not a static planner

Because sharding is free to change (no per-shard circuit/keygen, §5.4), the cut should be found
by **measuring**, not estimated by a static cost model. Static models mispredict the factors that
actually dominate proving time (memory pressure, cache, GPU launch/transfer, parallel
scheduling), so an empirical search beats a deterministic planner.

Approach: an **autotuning loop** that repeatedly proposes a partition, measures its proving time
(or a cheap proxy), and lets an optimizer/search pick the next cut until the time converges:

- **Cheap cost proxy**: measure commit + sumcheck time on representative shards, not the whole
  end-to-end proof every iteration.
- **Search strategy**: start from a coarse per-layer cut; split the slowest bottleneck shard and
  merge overly small shards; iterate. Escalate to Bayesian optimization / evolutionary search if
  needed.
- **Stopping condition**: stop when improvement drops below a threshold for N consecutive rounds.

The "AI" here is best realized as an optimizer (local search / Bayesian / evolutionary) over the
structured shard-boundary space, not necessarily an LLM; an LLM can act as a candidate proposer
but is optional. The concrete methodology is TBD (see §7).

### 5.7 Aggregation via recursive composition, added last

Cross-shard shared commitment only solves "linking", not "converging into one final proof".
Whether to then fold the N shard proofs into one succinct root via recursive composition is an
independent decision, added last and kept off the critical path.

## 6. Performance & GPU

- Prover wall-clock time: shared commitment > GKR reduction > recursion (given many cores/GPUs).
  Shared commitment is fully parallel, and its WHIR commitment (NTT + Merkle) is the
  data-parallel, most GPU-friendly part; GKR reduction is sequential across layers; recursion is
  scalar-dense and sequential, benefiting least from GPU.
- GPU pays off only when commitments are large and few: measured, a 2^15 small commitment is
  launch/transfer-bound (GPU slower), while a 2^22 commitment is 25.9x on GPU. The 200M proof's
  "many small commitments" measured 402s on 64-thread CPU vs 548s on GPU. The optimization is to
  use WHIR batching to commit many tensors at once, not hundreds of small commitments.
- The hybrid structure (internal reduction + cross-shard shared commitment) naturally reduces
  the number of boundary commitments and enlarges each one, which is more GPU-friendly.

## 7. TODO / Open Questions

- The concrete fixed-point semantics contract (exact scales, rounding, table indexing, padding
  tables).
- The `Program` IR op list and register/weight representation.
- The recursive-aggregation proof-system choice (ties into on-chain convergence; needs a
  feasibility study).
- The on-chain convergence route (Goldilocks GKR/WHIR → EVM-verifiable) option selection.
- The sharding autotuning methodology: cost-proxy design, search-strategy choice, and the
  convergence/stopping criteria.
