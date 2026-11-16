# table-key: the complete key chain over the registered candidate instance

## Scheme and provenance

- Scheme: table-key. Code under test: `d96fc2c1` (per-row bits encoding tails),
  `2545dcda` (key enumeration in the plan's matrix order), `acbf1f89` (plan
  table identity = the registered reader identity), on top of `41090c52`
  (fixed public R19 declaration) and `c7b7795b` (the evidence driver takes the
  range from the public configuration). Previous scheme close: `c7b7795b`;
  the diagnostic iteration that localized the failures is the trace
  `20261116-1115-table-key-c7b7795b-e46f5efb` of the same round.
- Purpose: the clean release evidence of the complete key chain under the
  fixed R19 declaration: first + shared + second parts, the canonical byte
  bundle, the independent `verify_key`, the real-bundle negatives, the durable
  read budget and the new candidate instance registration.
- Boundary: this is the chain evidence for the one registered candidate
  instance; it is not a performance comparison across schemes. The R18 refusal
  record stays; the diagnostic roots stay spent.

## Declaration

- Fixed public declaration: lo=-262144, hi=262143, d=19, mu=5, offset=262144,
  bit_entries=2^24 (exactly the admitted limit), key_bits=20, table_log=19.
  No clamping, no dropped entries; the 48 R18 outliers stay on record.
- The verifier rebuilds the declaration from the same public configuration;
  the driver uses only the public derivation. The range change changes the key
  plan/statement identity.

## New candidate instance

A fresh candidate instance was registered for this run; the previous
instances are kept and not overwritten.

| item | value |
|---|---|
| commit outcome | PASS, stage 3,912 ms |
| table root | `4b35cc42…` |
| model-level table identity | `ad0803db…` |
| reader identity (bound by the plan) | `5bf1709b…` |
| table configuration | `9017dd1a…` |
| record length | 1024 |
| previous instance root | `601d921c…` (kept) |

`registration.log`: the registration test binds the model-level meta identity
and the reader identity (root, record, configuration, shape) and checks the
old instance's root differs.

## Clean release run

- Command: `cargo test --release -p zkie --test key_chain_evidence --
  --ignored --nocapture` with `ZKIE_KEY_EV` and `ZKIE_KEY_INSTANCE` pointed at
  this directory and instance; the run's true status is EXIT=0.
- Log: `key-chain-evidence.log`; machine-readable result:
  `key-chain-evidence.json`.

| item | value |
|---|---|
| test result | 1 passed, 0 failed, 278.62 s |
| proof bytes | 4,543,400 |
| first / shared / second bytes | 3,552,176 / 704,616 / 286,576 |
| prove wall | 188,193 ms |
| verify wall | 58,432 ms |
| VmHWM (process) | 35,432,232 KiB |
| durable ledger | 0 -> 52 |
| plan digest | `d8baed93…` |
| table root | `4b35cc42…` |
| reader identity | `5bf1709b…` |

The run covers the first part (index mask, range lookup, pending claim, W,
Eval row, key group FOLD and multi-stage IOPP, one-leaf binding), the shared
part (Gu/Omega messages, folded table, multi-stage IOPP), the second part
(own group W + Omega + index mask, Eval and Dot rows, IOPP) and the
independent `verify_key` over the canonical byte bundle (a version plus three
length-prefixed sections). The real-bundle negatives are exercised in the
same run: truncation, one trailing byte, a changed version, and a repeated
verification on the spent read budget are all refused.

## Tests

- Workspace gate: `cargo test --workspace` EXIT=0; 1,235 passed, 0 failed,
  21 ignored across all targets (`workspace-tests.log`). The suite includes
  the fixture-scale key chain, its tamper matrix (roots, samples, claims,
  transcripts, byte bundle) and the frozen-geometry range tests.
- The registration test is in `registration.log` (2 passed).

## Ledger

- This instance: 0 -> 52, one charge, no refund.
- The diagnostic roots of the round (`gpt2-seq16-r653-a` .. `-k`, and
  `601d921c`, `r19-a/b/c`) remain spent and are not reset.

## Status propagation

The run wrapper records the true `cargo test` status and propagates it as the
command's exit status; the recorded EXIT above is the real status, not an
outer wrapper's zero.

## Deterministic functional evidence (amended after review)

This run used the fixed public constant `Digest([0x6B; 32])` as the run seed
for `prove_key`. The hiding material of the index mask and the lookup is
derived from that public seed, so this run is deterministic functional
evidence over the fixed table and range declaration, not hidden-key delivery
evidence. The run and its numbers are kept unchanged in this trace. The
private-seed rework run is `20261116-1147-table-key-8f9faab0-df994559`; this
instance's durable budget stays spent (52) and is not reset.
