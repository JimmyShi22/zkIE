# One public artifact and an independent lm-head chain verifier

## Scheme

- Scheme: `lm-head-artifact`.
- Current code: `746bd8bf` (this scheme), following `6dc7ca08` (artifact
  recorder and codec).
- Previous code: `6dc7ca08`.
- Scope: the real GPT-2 LM-head local chain. The production three-stage
  command does not carry this chain yet.

## Purpose

Produce one versioned artifact from the public prover messages, verify the
whole chain from that artifact without prover state, and reject the nine
negative classes of the acceptance task card. The proof volume and prove
throughput are recorded under the current performance focus.

## Measurement basis

- Same machine, release build:
  `cargo test -p zkie --release real_gpt2_lm_head_chain -- --ignored --nocapture`.
- The test commits the real asset, proves, builds the artifact, decodes it,
  and runs the independent verifier in a fresh session built from the same
  public setup. Every negative mutation is verified by the same
  `verify_limited_chain` entry and must return an error.
- Full workspace gate: `cargo test --workspace --release`.
- Values are bytes unless noted; wall times are one run per row on this
  trace. The artifact contents depend on the drawn columns and private
  material, so sizes vary by a few kilobytes between runs.

## Artifact

The artifact holds only public messages:

- version and protocol identity;
- the canonical `LimitedKeyStatement` (model, table, configuration, shape,
  matrix parameters and flags);
- G_u and Omega roots and their sample answers;
- one physical column opening per unique drawn position, sorted by
  position (repeats reuse it);
- the FOLD start message and its round polynomials;
- per switch: declaration identity, declared root, sample answer, member
  openings, fresh opening, the initial old-mask root and opening, and the
  following stage FOLD start and rounds;
- the BASE artifact;
- the final seal.

It holds no folded message, switched row, encoding tail, old-mask message or
fresh-mask coefficient. The verifier reconstructs the FOLD input claim from
public inputs:

- sample and record rows from the verifier session;
- column rows from the verified openings and the equation weights;
- the LIN row from the public `psi`/`zeta` weights and the group shape.

The prover driver and the verifier call one shared assembler, so the FOLD
handoff is derived from the same declarations and BAT challenge instead of
being accepted from the prover.

## Real chain

| run | commit wall | prove wall | commit HWM KiB | prove HWM KiB | BASE bytes | artifact bytes | sealed |
|---|---:|---:|---:|---:|---:|---:|---|
| shared assembler | 29.505 s | 8.293 s | 526,612 | 603,660 | 63,528 | 5,361,592 | yes |
| independent chain | 29.801 s | 8.391 s | 526,720 | 603,816 | 62,760 | 5,387,504 | yes |
| negative matrix | 29.817 s | 8.389 s | 526,280 | 603,364 | 62,760 | 5,378,504 | yes |
| final matrix | 29.516 s | 8.328 s | 526,672 | 603,768 | 61,240 | 5,368,336 | yes |

The cached artifact on disk is the final run:
`chain-artifact.bin`, 5,368,336 B, sha256
`48ae2ab25f716dc536574780099dcb43265b280cc219df4519459492ebec16f1`.

Prove throughput on this relation chain, 16 sequence tokens per chain:
- 16 / 8.293 s = 1.929 tokens/s;
- 16 / 8.391 s = 1.907 tokens/s;
- 16 / 8.328 s = 1.921 tokens/s.

This is the LM-head relation chain only; it is not a full-model tokens/s
figure. Commit and prove still run in one process for these rows.

## Proof volume

Measured on an independent run with the same plan:

| component | bytes | share |
|---|---:|---:|
| column openings (603 logical, 599 unique) | 5,311,224 | 98.6% |
| column symbols | 4,939,776 | 91.7% |
| Merkle paths | 328,032 | 6.1% |
| BASE artifact | 62,376 | 1.16% |
| everything else (statement, roots, answers, FOLD, switches) | ~13,000 | 0.24% |

- `gzip -9` reduces 5,386,896 B to 5,344,665 B (-0.8%); field words are
  nearly uniform 64-bit values, so there is no packing gain.
- Storing one opening per unique position is implemented; repeats were 1-4
  in the observed runs, saving about 8-35 KB.
- Sharing Merkle path nodes across openings is not implemented; a node pool
  with 32-bit indices is estimated to save roughly 78-119 KB (1.4-2.2%).
- The column symbols are the physical opened columns required by the current
  column-claim path. They are not redundant under that path; reducing them
  would require a protocol-level change.

## Negative matrix

All entries run the full decode and independent verification path and must
end in rejection.

| class | mutation | result |
|---|---|---|
| same-chain wrong statement | table root / model commitment changed | rejected |
| replaced input graph edge | G_u root or Omega root changed | rejected |
| replaced commitment opening | column path or member opening symbol changed | rejected |
| missing declaration | switch, stage FOLD, answer or column opening removed | rejected |
| wrong challenge order | FOLD round changed; switch records swapped | rejected |
| missing stage | switch or stage FOLD removed | rejected |
| budget exhaustion | opening list exceeds the planned draw count | rejected |
| truncated / trailing | artifact truncated or extended | rejected |
| version mismatch | artifact or statement version changed | rejected |

Budget exhaustion inside the read and hiding ledgers is covered by the
engine budget tests (`prover_reads_cache_and_bound_the_budget`, the holder
reservation test); the artifact path additionally bounds the opening list by
the planned draw count. The verifier also rejects a wrong protocol identity,
statement validation failure and incomplete records.

Fold-segment negatives (wrong statement, changed column path, missing
column) are exercised separately and are included in the logs.

## Correctness evidence

- `workspace-tests.txt`: full workspace release test suite green.
- `real-chain-negatives.txt`: positive independent verification returns the
  prover seal, and every negative mutation above is rejected.
- `chain-artifact.bin`: the final artifact from the positive run.
- The independent verifier starts a fresh session from the same parameters,
  decodes the artifact and uses no prover witness. The private mask provider
  carried by the reconstructed setup is not read; moving the verifier to a
  truly public, separate-process setup is the stage-handoff step.

## Open items

- The commit, prove and verify stages still share one process; the controlled
  stage handoff and independent-process measurements are the next step.
- The production three-stage command does not carry this chain yet.
- Full-model tokens/s, the three model groups and seq512 remain outside this
  relation-chain trace.
- The verifier derivation is implemented under the established rule that the
  verifier derives the declaration from the same bound declaration set and
  BAT challenge; expert review happens with the batch delivery.
