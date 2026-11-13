# Encoded projection table attached to the gpt2 model commitment

## Scheme and baseline

- Scheme: `model-table-commit` (first commit-artifact wiring of the
  production encoded table).
- Code: `6a93660a`; previous commit `b5e1b6d5`; branch `refactor/v2`.
- Scope: the layer-0 key projection weight is committed both as the raw row
  commitment and as the encoded table of the trusted plan. The consumer
  relation, the verifier table wiring and the policy regeneration remain the
  next steps.

## What changed

- `models/gpt2/src/plan.rs`: `asset_table_params()` is the trusted table list
  of the model (the layer-0 K projection weight) and `MODEL_VERSION` is 2.
- `models/gpt2/src/executor.rs`: `build_commitment` takes the table inputs and
  commits with `commit_assets_with_tables`. The raw row commitment is
  unchanged; the encoded table adds one anchor leaf over the same rows.
- `crates/zkie-ops/src/proof/asset_table.rs`: the private randomness set codec
  binds each entry by asset id. Unknown, duplicate or missing assets and a
  shape-mismatched randomness are rejected, and the decoded set is aligned
  with the trusted table list.
- `crates/zkie-ops/src/proof/model_commitment.rs`: the asset link proof is
  built over the full anchor record list (assets followed by encoded tables),
  because the anchor registry now covers the table leaves too. Without this,
  every asset opening failed at the registry index check once a table leaf was
  present.
- `zkie/src/provider.rs`: commit samples the private randomness of the trusted
  tables, commits with them and publishes `private/asset-table-randomness.bin`
  in the manifest with its digest. Prove reads the file back, decodes it
  against the trusted parameters, rebuilds the same commitment and rejects a
  rebuilt table set that differs from the published `public/asset-tables.bin`.
- `zkie/tests/s7_context.rs`: the historical context fixture now asserts the
  existing model-version gate rejects the old-version artifact; the artifact
  stays on disk and is not deleted.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Trusted table list for the model | trusted plan expression; encoded table anchor leaf in the commit | `asset_table_params`; `build_commitment` with tables | gpt2 model tests |
| Private randomness carried by the commit artifact | `private/asset-table-randomness.bin`; asset-id keyed set | encode/decode set; provider writes the file and digest | codec unit tests; manifest |
| Prove rebuilds the same table commitment | prove path; published `public/asset-tables.bin` | decode against trusted params; rebuild; reject a changed table set | codec tests; real commit and prove smoke |
| Asset openings survive the extra table leaf | anchor registry; link proof | link proof over the full record list | integration test with a table anchor |
| Old-version artifacts rejected by existing semantics | existing `model_version` gate | `MODEL_VERSION` 2; historical fixture asserts rejection | s7 context test |
| Verifier table wiring, policy regeneration, consumer relation | `verify_model_anchor_with_tables` caller | **not yet** | gap |

## Evidence

- `cargo test --workspace`: 50 suites, 990 passed, 0 failed, 17 ignored.
- Hard gate: norm-free toy output-head proof 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  identical before and after (captured by a temporary test write, reverted
  before commit); see `proof-bytes.txt` and `proof-45080.bin`.
- Focused: `zkie-ops` asset-table unit tests 3 passed; `zkie-ops` asset-table
  integration test 6 passed (including the table-anchor opening test); gpt2
  model tests 22 passed; s7 context test 3 passed. Transcripts are
  `asset-table-tests.txt`, `model-tests.txt` and `s7-context-tests.txt`.
- Real gpt2-seq16 commit and prove smoke on the same machine (release CLI,
  `smoke/`): commit PASS 3,818 ms, VmHWM 356,524 KiB; public
  `asset-tables.bin` 24,823 bytes; private `asset-table-randomness.bin`
  270,368 bytes; prove PASS 14,768 ms, VmHWM 4,496,208 KiB, `proof.bin`
  2,190,155,518 bytes (the CLI report counts the proof directory,
  2,190,159,494 bytes). The prove stage read the private randomness back,
  rebuilt the same tables and matched the published table leaves.
- This is not an optimization step; no before/after performance claim is made.
  The commit and prove stages now build the encoded table, which is real added
  work.
- No push.

## Boundaries

- Verify still calls `verify_model_anchor_with_tables` with an empty trusted
  table list, so a version-2 gpt2 commitment is not yet accepted by verify;
  the table wiring and the version gate are the next step.
- The gpt2 policy still pins model version 1. Historical proofs stay on disk
  and are rejected by the existing gate; policy regeneration is the next step.
- `zkie anchor` rebuilds the commitment without the encoded table, so its
  anchor identity does not yet match a table commit. Policy regeneration needs
  the table binding; a table commit samples fresh private randomness on every
  commit run, so the trusted policy anchor and the reproducibility of repeated
  commit runs need an explicit decision in the policy step.
- The L0 K projection consumer relation is not yet closed in the proof; the
  raw row commitment and every existing asset check remain, and no plaintext
  was removed. This is additive commit wiring, not an asset migration.
- gemma3 and timesfm are unchanged: their commits write an empty private
  randomness set and their anchor, version and policy path stay as before.
- The smoke commit's private randomness file is a gitignored binary that stays
  on disk under `smoke/commit/private/`; it is never committed.
- No claim that B2 is accepted, that the consumer closure is complete, or that
  plaintext can be removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `asset-table-tests.txt`, `model-tests.txt`,
`s7-context-tests.txt`, `smoke/`, `NOTES.md`.

## Development note

The first same-machine smoke of the working tree failed in the prove stage
with `invalid_index` and no proof was written. The asset link proof was built
over the asset leaf records only, while the anchor registry now also covered
the table leaf; the maximum registry index check rejected the opening. The fix
builds the link proof over the full anchor record list and is part of the
same code commit, covered by the table-anchor opening test. The failed run
used a temporary directory and its logs were overwritten by the final smoke.
