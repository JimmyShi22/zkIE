# Registered model table instances bound into the trusted policy

## Scheme and baseline

- Scheme: `model-table-instance` (one preprocessing instance per case; the
  policy pins its public identity; registration and reuse are separate).
- Code: `d1824d49`; policy registration commit: this trace commit; previous
  commit `a5de37d8`; branch `refactor/v2`.
- Scope: the operator registration tool, the policy binding and the reuse
  path. The consumer relation, the negative matrix and the five-case matrix
  remain the next steps.

## What changed

- `zkie/src/anchor.rs`: the anchor command accepts the saved private table
  randomness (`--table-randomness`). It rebuilds the same encoded-table
  commitment the commit published and prints the public instance identity:
  the anchor leaf count and, per table, the identity, configuration, root,
  record digest and valid range. A model with trusted tables and no saved
  material is rejected; a model without trusted tables rejects unexpected
  material. The private material is never read by the verifier.
- `crates/zkie-ops/src/proof/model_commitment.rs`: a digests-and-tables commit
  entry exists for the operator identity check. The commit, prove and anchor
  paths still share one commit implementation.
- `zkie/src/run.rs` and `zkie/src/lib.rs`: the matrix and decode-matrix accept
  a registered commit per case (`--commit-dir <case_id>=<path>`) and reuse it
  for every run. Without a registered commit the previous cold-create and
  warm-reuse behavior is unchanged. A registered instance reports its commit
  directory; new-commit cost belongs to the separate provisioning run.
- `models/gpt2/case.policy`: the two gpt2 cases are bound to the registered
  instances. Only the model version and the expected anchors change; the
  security parameters, inputs, ground truth and output constraints are
  unchanged.
- `zkie/tests/s7_context.rs`: the positive context test reads the registered
  instance artifacts from the registration archive; the legacy artifact stays
  on disk and is asserted to be rejected by the version gate under a
  version-one policy.

## Registration record (public identity)

| Case | Anchor id | Anchor leaves | Table identity | Table root | Record digest |
| --- | --- | ---: | --- | --- | --- |
| gpt2-seq16 | `91397c7e…0cc060` | 200 | `f1e9fda7…8ca5a5` | `4819d90b…2b32dff` | `67bbede2…05fc4a` |
| gpt2-seq512 | `fd6164ae…5e942e` | 200 | `18b36cf3…09b67d` | `b6c10232…de6d40c` | `27803e85…4efcd33` |

- Both instances cover `gpt2/weights/L0_k_w_i32.bin` (asset id 2) with the
  same trusted table geometry and the full i32 valid range; the configuration
  and identity differ per instance because they bind the input-derived model
  commitment. Full values are in `registration.json` and the anchor outputs.
- The model definition version stays at the definition version; the instance
  identity is the anchor, so a future re-registration changes only the anchor
  and table identity, not the model version.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| One preprocessing instance, saved and reused | saved private randomness in the commit artifact; instance anchor | anchor rebuilds the same commitment from the saved material | anchors match the commit public anchors |
| Policy pins the instance public identity | policy anchor per case; `check_policy` | policy updated to the registered anchors; version gate kept | both anchor policy checks pass; version-two proof verified |
| Verifier never needs private randomness | verify reads policy plus public bundle/proof | no private input in verify; unchanged | verified without the private file |
| New commit must be registered explicitly | expected anchor in the policy | an unregistered instance is rejected; no policy backfill from proof | unregistered verify rejected `statement_mismatch` |
| Same instance requires its private material | prove reads the saved material | missing material blocks proving; a changed material breaks the rebuilt tables | `missing_commit_artifact`; `statement_mismatch` |
| Reuse path for reruns | matrix registered commit option | all runs reuse the registered commit | code; matrix run not executed in this step |
| Consumer relation, record/mask separation, proximity checks | consumer-chain boundaries | **not yet** (consumer work) | gap |

## Evidence

- `cargo test --workspace` with the registered policy and instance artifacts:
  50 suites, 991 passed, 0 failed, 17 ignored. The context positive test and
  the legacy rejection test both ran (see `workspace.txt`).
- Hard gate: norm-free toy output-head proof 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  identical before and after; see `proof-bytes.txt` and `proof-45080.bin`.
- Provisioning (new instance) measurements, same machine, release CLI:

| Stage | Case | Outcome | Stage wall ms | VmHWM KiB |
| --- | --- | --- | ---: | ---: |
| commit (new instance) | gpt2-seq16 | PASS | 3,815 | 355,520 |
| commit (new instance) | gpt2-seq512 | PASS | 3,813 | 355,584 |
| prove (registered instance) | gpt2-seq16 | PASS, proof bytes 2,190,159,493 | 14,755 | 4,497,676 |
| verify with the registered policy | gpt2-seq16 | PASS, expected binding, digest `135c4d1c…339280` | 17,632 | 4,122,308 |

- Policy checks: the anchor rebuild from each instance's saved material
  matches the policy anchor, model identity, version and output expectation.
- Failure semantics measured:

| Negative | Result |
| --- | --- |
| verify an unregistered instance under the registered policy | rejected `statement_mismatch` |
| prove with the private material missing | rejected `missing_commit_artifact` |
| prove with one byte changed in the private material | rejected `statement_mismatch` |
| anchor for a table model without the private material | rejected `missing_commit_artifact` |

- The private randomness files are gitignored binaries under `smoke/` and are
  never committed; the verifier never reads them. The unregistered negative
  uses the earlier verification trace's instance artifacts, kept on disk.
- No push.

## Boundaries

- The seq512 instance is registered and policy-checked; its full prove and
  verify are deferred to the next test step. No five-case matrix was run in
  this step.
- The matrix registered-commit path is implemented but not exercised in a
  benchmark run here; benchmark orchestration (which runs reuse a registered
  instance, and how the provisioning cost is reported) stays with the TL. A
  new commit is a new instance until registered; reuse timings must not be
  reported as new-commit cost.
- The consumer relation for the L0 K projection is still not closed in the
  proof; no plaintext was removed and every existing asset check remains. The
  record public values and the private salts/tails are separate; no mask seed
  is derived from the instance material in this wiring. The preprocessing and
  proximity obligations of the consumer chain remain open.
- Hidden-session budgets and per-request mask generation are untouched by this
  step; no budget is reset on restart or copy in these paths.
- gemma3 and timesfm keep an empty table list and their previous policy
  semantics.
- No claim that B2 is accepted or that the consumer closure is complete.

## Artifacts

`identity.txt`, `change.diff`, `policy.diff`, `registration.json`,
`workspace.txt`, `proof-bytes.txt`, `proof-45080.bin`, `smoke/`, `negatives/`,
`NOTES.md`.
