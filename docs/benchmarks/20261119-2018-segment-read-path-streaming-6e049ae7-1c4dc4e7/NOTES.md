# Segment shared read path streaming (gpt2 A04)

## Scheme and baseline

- Scheme: `segment-read-path-streaming`.
- Code: `6e049ae7`; previous scheme head `1c4dc4e7` (the bounded-liveness
  acceptance).
- Purpose: complete the step-2 shared read path hardening at the target
  scale of the gemma3 output head (2^18 columns, about 6.4 GB at field
  width): (1) the frozen table view fills whole-column results in bounded
  `TABLE_READ_CHUNK_ROWS` chunks while the prover-side combined column keeps
  its O(rows) contract, so every range read stays bounded; (2) the parameter
  asset value reader refuses files above a 1 MiB bound before it reads (the
  norm gain/bias vectors are a few KiB); (3) the stored table commit keeps
  its `needed <= max_bytes` refusal before allocation and the bounded asset
  source refuses a per-row factor buffer above the cap; (4) the segment
  statement carries the bounded source identity derived from the source kind
  and its three bounds, and the verifier compares it with the plan-fixed
  derivation, so a statement produced by a different source is a statement
  mismatch. The source identity deliberately does not enter the identity
  binding digest, so the session derivation and the segment carrier bytes
  stay unchanged.

## Method

- Single machine. One fresh commit; the three stages ran as separate
  processes through the acceptance driver (the durable one-request column
  ledger is keyed by asset directory and table root with no refund, so each
  commit supports exactly one prove).
- The negative matrix runs through the full public verify path; every
  tampered artifact is republished with a self-consistent manifest and
  complete marker, and the control run must pass first.

## Results (gpt2-seq16, round 1)

| stage | wall ms | VmHWM KiB | outcome |
|---|---|---|---|
| commit | 236,732 | 27,296,464 | PASS |
| prove | 755,339 | 212,621,908 | PASS |
| verify | 191,993 | 125,784,612 | PASS |

- Proof bytes: 3,589,344,213 B.
- Carrier sha256:
  `3815f1af7ac5913195930449a8ecbf61175aa6df2e20595df549b21367d16819`
  (size 1,707,601,596 B) - byte-identical to the four accepted rounds and to
  the two bounded-liveness runs; the segment proof bytes are unchanged.
- Trace `proof.bin`: 1,879,423,254 B (same serialized length; its values
  vary per commit as in the accepted rounds). `statement.bin` 463 B (the
  bounded source identity adds 32 B to the previous 431 B);
  `public-inputs.bin` 72 B.
- Negatives (real verifier): control PASS; `statement_identity`,
  `statement_source_identity`, `version`, `reader_root`, `record_digest`,
  `messages_truncated` and `source_rows` all rejected.

## Gates

- Workspace: 56 suites, 1,319 passed, 0 failed, 32 ignored (release), six
  new component tests included.
- Hard gate: 45,080 B sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (byte-identical).

## Comparison with the previous scheme head

- Same-machine single runs, previous head `1bf7d00a` to this head
  `6e049ae7`: prove 765,577 -> 755,339 ms (-1.3%) and VmHWM 212,605,940 ->
  212,621,908 KiB (+15,968 KiB, +0.0075%); verify 195,936 -> 191,993 ms
  (-2.0%) and VmHWM 125,866,204 -> 125,784,612 KiB (-81,592 KiB, -0.06%);
  commit 235,045 -> 236,732 ms (+0.7%). The deltas are inside the 3% noise
  band.
- No VmHWM reduction is claimed: the three read paths were already chunked
  or otherwise bounded at gpt2 scale, and the delivery is the explicit
  bound, the byte-identity component tests and the refusals. The carrier
  hash is the hard evidence that the proof bytes did not change.

## Bounded accounting

- The frozen table view reads a whole padded column in
  `TABLE_READ_CHUNK_ROWS` (4096) chunks; the content check already streamed
  4096-row chunks and the evaluation chunks are unchanged. The lookup
  prover's combined column stays one O(rows) vector by contract (its stream
  product tree is a future item only if profiling shows it peaking).
- The parameter values read refuses any asset above 1 MiB before reading;
  the two call sites (prove and verify) keep the same values and leaf root.
- The encoded table build already streamed rows into a disk row store with
  `needed <= max_bytes` checked before allocation; the per-row factor buffer
  now also has to fit the cap.

## Boundaries

- The source identity is a statement-level binding checked against the
  plan-fixed derivation; a swapped-source statement is rejected (negative
  case `statement_source_identity`). It is not remote attestation: the
  production wiring discipline is that the single provider path builds the
  table through the bounded asset source.
- This run covers gpt2-seq16 at 16 positions. The gemma3 2^18 consumer
  binding is verified when that lane retests.
- The statement grows by the 32-byte binding field, recorded honestly; the
  segment carrier bytes are unchanged.
