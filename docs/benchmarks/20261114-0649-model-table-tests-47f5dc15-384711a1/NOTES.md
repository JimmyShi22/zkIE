# Model table registration tests: seq512, reuse orchestration and negatives

## Scheme and baseline

- Scheme: `model-table-tests` (test wrap-up of the registration and reuse
  wiring; no source change in this round).
- Tested tree: `47f5dc15` (HEAD at run time); previous commit `384711a1`;
  branch `refactor/v2`.
- The registered instances are those of the registration trace
  `docs/benchmarks/20261114-0637-model-table-instance-d1824d49-a5de37d8`
  (seq16 anchor `91397c7e…0cc060`, seq512 anchor `fd6164ae…5e942e`). The
  tracked policy is bound to them (`models/gpt2/case.policy`, sha256
  `ab0ae6d4…e2d767`).

## What was exercised

1. The registered gpt2-seq512 instance end to end: prove and verify with the
   tracked policy.
2. The matrix reuse entry: both gpt2 cases ran against their registered
   commits (`--commit-dir <case_id>=<path>`); every run reuses the registered
   instance and no commit stage runs inside the matrix.
3. A negative matrix through the real CLI verification and proving paths,
   plus the in-process real-verifier table negatives already covered by the
   asset-table test.
4. The hard gate and the full workspace.

## seq512 full chain

| Stage | Outcome | Stage wall ms | VmHWM KiB | Bytes / digest |
| --- | --- | ---: | ---: | --- |
| prove (registered instance) | PASS | 61,624 | 27,376,900 | proof directory 11,127,652,976 |
| verify (tracked policy) | PASS, expected binding | 39,855 | 18,850,140 | digest `eac0949d…f0dfdd`, 512 elements = GT |

## Matrix reuse orchestration

Command: `zkie matrix --cases models/gpt2/case.toml --cold-runs 1
--warm-runs 1 --commit-dir gpt2-seq16=<registered16> --commit-dir
gpt2-seq512=<registered512> --out <trace>/matrix`. The matrix report names the
registered commit per case; no commit stage exists in the run directories.

| Case | Run | Prove ms / KiB | Verify ms / KiB | Proof bytes | Outcome |
| --- | --- | --- | --- | ---: | --- |
| gpt2-seq16 | cold-0 | 14,712 / 4,498,240 | 17,757 / 4,120,036 | 2,190,159,493 | PASS / PASS, digest GT |
| gpt2-seq16 | warm-0 | 14,717 / 4,496,864 | 17,943 / 4,123,144 | 2,190,159,493 | PASS / PASS, digest GT |
| gpt2-seq512 | cold-0 | 62,143 / 27,378,252 | 39,472 / 18,869,060 | 11,127,652,976 | PASS / PASS, digest GT |
| gpt2-seq512 | warm-0 | 61,546 / 27,377,804 | 39,855 / 18,868,552 | 11,127,652,976 | PASS / PASS, digest GT |

- New-commit cost is recorded separately, in the registration trace:
  provisioning commit gpt2-seq16 3,815 ms / 355,520 KiB and gpt2-seq512
  3,813 ms / 355,584 KiB. The reuse runs above contain no commit stage and
  their timings are not new-commit costs.

## Negative matrix (real verification and proving paths)

| Negative | Mutation / precondition | Chain | Result |
| --- | --- | --- | --- |
| New instance not registered | fresh commit with new private randomness, its own prove PASS | CLI verify, tracked policy | `statement_mismatch` |
| Policy names another instance | seq16 entries replaced by the seq512 anchor (non-zero, parse-valid) | CLI verify | `statement_mismatch` |
| Published table root changed | one byte in the root inside `asset-tables.bin` | CLI verify | `statement_mismatch` |
| Published table configuration changed | one byte in the configuration | CLI verify | `statement_mismatch` |
| Published table identity changed | one byte in the table identity | CLI verify | `statement_mismatch` |
| Published record changed | one byte inside the serialized record | CLI verify | `statement_mismatch` |
| Published asset path changed | layer tag in the table path | CLI verify | `asset_mismatch` (rejected before relation checks) |
| Policy anchor zeroed | seq16 anchor replaced by zero | CLI verify | `invalid_config` at policy parse (zero identity rejected) |
| Private material missing | commit artifact copied without the private file | CLI prove | `missing_commit_artifact` |
| Existing public proof after the private file is missing | verify the registered proof against the material-less public copy | CLI verify | PASS with the expected digest (verification never reads private material) |
| Private material changed | one byte in the saved randomness | CLI prove | `statement_mismatch` (rebuilt tables differ from the published ones) |
| Anchor without private material | `zkie anchor` for gpt2 without `--table-randomness` | CLI anchor | `missing_commit_artifact` |
| Committed table field changes | root, configuration, identity, shape, record, valid range, removed, duplicated, asset id, path, plus asset-opening value change | in-process `verify_model_anchor_with_tables` / `verify_asset_opening` | all rejected (asset-table test suite) |

- The policy file is not written by any verification path: sha256
  `ab0ae6d4…e2d767` before and after the whole matrix; the working tree stayed
  clean apart from this trace. No policy backfill from a proof or bundle.
- Every negative above runs through the production entry that consumes it;
  the in-process rows use the same verification function the CLI provider
  calls.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| seq512 registered instance end to end | registered commit artifact; tracked policy | measured prove and verify | chain table above |
| Reruns reuse the registered instance | matrix `--commit-dir` entry; run `reuse` | every run reuses the registered commit; no commit stage | matrix table and report |
| New-commit cost is separate | provisioning commit runs | recorded in the registration trace, not in reuse runs | 3,815 / 3,813 ms |
| Verifier needs no private randomness | policy expectation plus public bundle/proof | verify runs without the private file | material-less verify PASS |
| A new instance must be registered | expected anchor in the policy | unregistered instance rejected; policy unchanged | negatives |
| Table identity, root, record, configuration, path are bound | anchor leaf; table meta checks | mutations rejected | negatives |
| Missing or changed private material blocks proving | saved randomness is the rebuild input | prove rejects; existing proofs remain verifiable | negatives |
| Consumer relation negatives (consumer proof chain) | production projection consumer | **not yet wired** | gap |

## Boundaries and gaps

- The consumer relation is not in the production proof path yet, so the
  consumer-level negatives (reduction coefficient change, mask slot swap or
  free scalar, missing fold mask closure, raw product/output change, missing
  proximity test, bias losing the asset identity) are not executable in this
  wiring; they belong to the next segment. No plaintext was removed and every
  existing asset check remains.
- "Spent budget is not restored on restart or copy" has no counterpart in this
  path: the registration/commit/prove/verify wiring keeps no sampling-budget
  ledger. The hidden-session budget ledger and its test (`column_budget_is_enforced`)
  are untouched and were not run here. This is recorded as not applicable,
  not as verified.
- Entropy failure rejection is code evidence only: the sampler maps a
  `getrandom` failure to `EntropyUnavailable` and has no fixed-seed fallback;
  no fault-injection hook was added, so it was not exercised at runtime.
- gemma3 and timesfm were not part of this round; their table list is empty
  and their policy semantics are unchanged.
- The seq512 prove peak (27.4 GB) and verify peak (18.9 GB) are reported for
  the record; memory is not a gate.
- No five-case matrix was run in this round; this round covers the table
  registration semantics only.
- No push.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `seq512/`, `matrix/`, `unregistered-new/`,
`material-less/`, `bad-private/`, `tampered-root/`, `tampered-configuration/`,
`tampered-identity/`, `tampered-record/`, `tampered-path/`,
`policy-anchor-verify/`, `policy-wrong-anchor-verify/`, `policy-anchor.toml`,
`policy-wrong-anchor.toml`, `NOTES.md`.
