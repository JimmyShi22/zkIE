# Store-free verification plan

Goal: make `verify` run without re-running forward and without reading the
witness `Store` (no weights, no intermediate activations). The verifier checks a
proof against a statement (model/shard structure + commitment roots) plus public
input/output.

## Current state (baseline)

- Verifier re-runs forward: `verify_shard_dag` / `verify_shard` call
  `forward_ops`, then read full tensors via `store.get(...)`, `mle::eval(...)`,
  `mle::partial_eval(...)` to recompute claimed evaluations and sumcheck
  reductions.
- Challenges are prover-chosen (`XorShift64`, `rng.field()`), not derived from a
  transcript.
- Only shard boundary tensors have committed variants
  (`prove_committed_shard_dag`); weights and lookup tables are read raw from the
  store, not committed. `verify_committed_shard_dag` still calls
  `verify_shard(store, ...)`.

## Claim reduction direction

Verification proceeds from shard output toward shard input:

```
shard output claim
  -> matmul reduction (claims eval(A)(u), eval(B)(v))
  -> nonlinear reduction (rsqrt / softmax / gelu logUp claims)
  -> ... recurse to the producing op ...
  -> leaf opening (weight / boundary / lookup table commitment)
```

A claim is `(tensor_id, point, claimed_value)`. Intermediate activations stay
virtual (never committed); only weights, lookup tables, and shard boundaries are
committed.

## Transcript order (Fiat-Shamir)

Absorb in this fixed order, deriving every challenge (u / v / r / alpha / beta)
from the transcript:

1. model id and shard/op structure (types + shapes, deterministic encoding)
2. commitment roots (weights, lookup tables, boundary) — present from P3 onward
3. in proof order: each sumcheck round message, each claim, each opening

Use a Goldilocks sponge (reuse `p3_symmetric::PaddingFreeSponge`, mirror
`batch_open.rs::BatchTranscript`).

## Final verifier interface

- `Statement`: model id, op/shard structure, weight/lookup/boundary roots, FRI
  parameters.
- `ProverWitness`: tensors + mmap, prover-only.
- `verify_shard(statement, proof) -> bool` (no Store).
- `verify(proof_bytes, public_inputs) -> bool` (standalone).

## Phases

- P0: this document + baseline freeze.
- P1: Fiat-Shamir transcript replaces `rng.field()`.
- P2: verifier consumes claimed evals from proof; no store reads.
- P3: commit weights / lookup / boundary; open leaf claims via WHIR.
- P4: serde serialization + standalone verify entry + full-model regression.

## Acceptance

GPT-2 seq16 (16/16) and seq512 (511/512) argmax aligned; prove/verify not slower
than the current layer baseline (GPT-2 512 ~41s/17s, Gemma3 512 ~58s/31s);
tampered claim / opening / boundary / root rejected.
