# table-key: private-seed run with the on-disk bundle and a JSON record

## Scheme and provenance

- Scheme: table-key rework of the evidence entry (F1-F3 of the review). Code
  under test: `9d87589c` (private run-seed source), `f575c1ea` (complete key
  bundle kept on disk; JSON evidence record), `8f9faab0` (verify
  resident-set delta in the record). Previous scheme close: `df994559`; the
  earlier fixed-seed trace of the same round is
  `20261116-1125-table-key-acbf1f89-c7b7795b` and is kept as deterministic
  functional evidence, not hidden-key delivery evidence.
- Purpose: run the complete key chain on a new candidate instance with a
  private run seed, keep the complete canonical bundle on disk, verify the
  same bundle read back from disk through the public entry only, and emit a
  machine-readable JSON record.
- Boundary: this is the key chain evidence for one candidate instance; no
  protocol, shaping or sampling change; no new CLI or process framework; the
  bundle is a binary artifact kept outside git.

## Private run seed

- The evidence entry draws `KeyRunRandomness::Private`, which samples the
  master seed from the operating system generator. This is the existing
  private-randomness pattern of the code base (the last-base masking oracle
  uses the same fixed-in-tests / operating-system-in-production split).
- The seed is not logged, not written to the result file and not part of any
  artifact; nothing beside the proof leaves the entry.
- The fixture tests keep explicit fixed seeds; their runs are functional
  tests, not delivery evidence.

## On-disk bundle

- The complete canonical key bundle is written to `key-proof.bin` in this
  directory (binary artifact, ignored by git). The result records the path,
  the byte length and the SHA-256 digest.
- The verification reads the same bundle back from disk and calls the public
  entry with only the trusted plan, the registered reader, the public range
  lookup table and the bundle bytes. The prover witness is never passed.
- The prover's durable column budget is charged once for the instance; the
  verifier's read ledger is consumed once; a repeated verification on the
  same plan is refused and the prover's budget count is unchanged.

## Result record

- `key-chain-evidence.json` is a valid JSON object. The key-plan summary is a
  nested JSON object, not a debug string.
- Memory fields distinguish the process-cumulative high-water mark
  (`vmhwm_process_kib`) from the verifier's own stage measurements in the same
  process: `verify_vmhwm_before_kib`, `verify_vmhwm_after_kib`,
  `verify_hwm_delta_kib`, `verify_vmrss_before_kib`, `verify_vmrss_after_kib`
  and `verify_rss_delta_kib`. The high-water mark never decreases, so the
  verifier's marginal peak is the delta; its live memory is the resident set.
  No separate verifier process was added (out of scope).

## Candidate instance

A fresh candidate instance was registered for this run; the previous
instances are kept and not overwritten.

| item | value |
|---|---|
| commit outcome | PASS, stage 3,914 ms |
| table root | `404cca82…` |
| model-level table identity | `ad0803db…` |
| reader identity (bound by the plan) | `8aebae88…` |
| table configuration | `9017dd1a…` |
| record length | 1024 |
| previous instance root | `4b35cc42…` (kept) |

`registration.log`: the registration test binds the model-level meta identity
and the reader identity and checks the previous instance's root differs.

## Run

- Command: `cargo test --release -p zkie --test key_chain_evidence --
  --ignored --nocapture` with `ZKIE_KEY_EV` and `ZKIE_KEY_INSTANCE` pointed at
  this directory and instance; the run's true status is EXIT=0.
- Log: `key-chain-evidence.log`; machine-readable result:
  `key-chain-evidence.json` (valid JSON).

| item | value |
|---|---|
| test result | 1 passed, 0 failed, 281.37 s |
| proof bytes | 4,551,488 |
| first / shared / second bytes | 3,553,072 / 713,312 / 285,072 |
| bundle sha256 | `b3e218e9…` (matches the on-disk file) |
| prove wall | 190,938 ms |
| verify wall | 58,632 ms |
| process VmHWM (cumulative) | 35,431,832 KiB |
| verify HWM before/after | 35,431,832 / 35,431,832 KiB (marginal delta 0) |
| verify RSS before/after | 16,948,916 / 16,832,672 KiB |
| durable ledger | 0 -> 52 |
| read ledger | spent once; repeated verification refused |
| plan digest | `90550317…` |
| table root | `404cca82…` |
| reader identity | `8aebae88…` |

The run covers the first part (index mask, range lookup, pending claim, W,
Eval row, key group FOLD and multi-stage IOPP, one-leaf binding), the shared
part (Gu/Omega messages, folded table, multi-stage IOPP), the second part
(own group W + Omega + index mask, Eval and Dot rows, IOPP) and the
independent verification of the on-disk bundle. The bundle negatives are
exercised on the bytes read from disk: truncation, one trailing byte and a
changed version are refused; a repeated verification on the spent read
ledger is refused and the prover's durable budget is unchanged.

## Tests

- Workspace gate: `cargo test --workspace` EXIT=0; 1,235 passed, 0 failed,
  21 ignored across all targets (`workspace-tests.log`). The suite includes
  the fixture-scale key chain, its tamper matrix and the frozen-geometry
  range tests.
- The registration test is in `registration.log` (2 passed).

## Ledger

- This instance: 0 -> 52, one charge, no refund.
- The previous instances of the round (the fixed-seed candidate `4b35cc42`
  and the diagnostic roots) stay spent and are not reset.

## Status propagation

The run wrapper records the true `cargo test` status and propagates it as the
command's exit status; the recorded EXIT above is the real status, not an
outer wrapper's zero.

## Round closure

- The batch review answered YES for the two public semantics, the fixed R19
  declaration, the three wiring fixes and the three evidence-entry reworks
  (private run seed, on-disk bundle, JSON record). The earlier fixed-seed
  trace stays as deterministic functional evidence and is not the delivery
  evidence; this trace is the round's delivery evidence.
- Commits of the round: `41090c52` (fixed R19 declaration), `c7b7795b`
  (driver takes the public declaration), `d96fc2c1` (per-row bits tails),
  `2545dcda` (matrix entry order), `acbf1f89` (reader identity), `16fd8e4d`
  (diagnostic trace), `df994559` (fixed-seed trace), `9d87589c` (private
  run-seed source), `f575c1ea` (bundle on disk and JSON record), `8f9faab0`
  (verify resident-set delta), `01467a77` (fixed-seed traces marked as
  functional evidence), `140d5984` (private-seed trace), and this closure
  commit.
- Memory numbers: the recorded `verify_hwm_delta_kib = 0` only says that the
  verification stage did not raise the process high-water mark above the
  level the run already reached; it does not mean the verifier used zero
  memory. The `verify_rss_delta_kib` field is a non-negative growth amount
  computed with a saturating subtraction, not a signed net change; the raw
  resident-set numbers are 16,948,916 KiB before and 16,832,672 KiB after the
  verification, a net change of -116,244 KiB. The resident-set numbers are
  same-process measurements of the verifier stage, not an independent
  verifier-process peak measurement; the only process-cumulative peak is
  `vmhwm_process_kib` (35,431,832 KiB).
- Ledger: the rework instance is charged 0 -> 52 once; the earlier instances
  of the round stay spent and are not reset. The complete bundle
  `key-proof.bin` stays on disk in this directory and is not committed.
