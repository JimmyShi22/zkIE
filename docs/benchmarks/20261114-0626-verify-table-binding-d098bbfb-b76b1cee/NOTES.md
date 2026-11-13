# Trusted table binding in verification with the model-version gate

## Scheme and baseline

- Scheme: `verify-table-binding`.
- Code: `d098bbfb`; previous commit `b76b1cee`; branch `refactor/v2`.
- Scope: the verifier and the output-head context loader bind the published
  encoded-table set to the trusted plan; the model-version gate keeps its
  existing rejection semantics and runs before the table-aware anchor rebuild.
  Policy registration and the consumer relation remain the next steps.

## What changed

- `zkie/src/provider.rs`: the verifier builds the trusted table list once for
  the selected model and passes it to `verify_model_anchor_with_tables`
  together with the published `asset-tables.bin` metas. A version-two gpt2
  commitment now verifies against its table leaves; gemma3 and timesfm keep an
  empty list. The single table-parameter helper is shared by commit, prove and
  verify.
- `zkie/src/s7_context.rs`: the context loader reads the published table metas
  from the public bundle, binds them to the trusted table list in the anchor
  rebuild, and runs the model-version gate before the anchor check. An
  old-version artifact therefore keeps the existing `invalid_config`
  rejection instead of failing on the table-aware anchor; historical
  artifacts stay on disk.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Verify the anchor with the trusted table list | `verify_model_anchor_with_tables`; trusted plan table list | provider verify passes the trusted table list | smoke: version-two proof accepted |
| Published table metas are data checked against the plan | `asset-tables.bin`; `verify_asset_table_meta` | metas read from the public bundle and checked | smoke: tampered table set rejected |
| Old-version artifacts keep existing semantics | existing `model_version` gate | gate runs before the anchor rebuild | smoke: version-one policy rejected with `invalid_config` |
| Context loader binds the same trusted list | anchor rebuild in the loader | metas read and trusted list passed | s7 context tests |
| Policy registration, consumer relation | registration ruling; pending | **not yet** | gap |

## Evidence

- `cargo test --workspace`: 50 suites, 990 passed, 0 failed, 17 ignored.
- Hard gate: norm-free toy output-head proof 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  identical before and after; see `proof-bytes.txt` and `proof-45080.bin`.
- Focused: s7 context 3 passed; provider unit tests 4 passed; policy tests 17
  passed. Transcripts are `s7-context-tests.txt`, `provider-tests.txt` and
  `policy-tests.txt`.
- Same-machine smoke on gpt2-seq16 (`smoke/`, release CLI):

| Stage | Outcome | Wall ms | VmHWM KiB |
| --- | --- | ---: | ---: |
| commit | PASS | 3,818 | 357,260 |
| prove | PASS | 14,724 | 4,497,056 |
| verify with the test version-two policy | PASS, output binding expected | 17,656 | 4,102,120 |
| verify with the tracked version-one policy | FAIL, `invalid_config` | 1,412 | 2,177,768 |
| verify with one byte flipped in `asset-tables.bin` | FAIL, `statement_mismatch` | 1,382 | 2,179,400 |

- The accepted verify authenticated the expected output digest
  `135c4d1c…339280` over 16 elements, matching the policy expectation.
- The test version-two policy in `smoke/policy-v2.toml` is a local test
  fixture: the tracked policy with the model version raised and the seq16
  expected anchor replaced by this test instance anchor
  `58bde276…7886b5`. It is not a registered production policy; registration
  is the next step. `smoke/make-test-policy.sh` reproduces it.
- The version-one rejection happens after the proof decode, which is the
  existing verifier order; the rejection code is the existing version gate.
- This is not an optimization step; no before/after performance claim is made.
- No push.

## Boundaries

- The tracked gpt2 policy still pins model version 1 and the old instance
  anchors. Registering the new instance and updating the policy is the next
  step (one preprocessing instance; the policy pins its public identity;
  verifier never depends on the private randomness).
- `zkie anchor` still rebuilds without the encoded table, so it cannot yet
  produce the registered instance identity; the registration tool is the next
  step.
- Matrix orchestration for a registered instance is not changed here; the
  existing `run --commit-policy reuse --commit-dir` already reuses one commit
  artifact for subsequent requests.
- The L0 K projection consumer relation is still not closed in the proof; no
  plaintext was removed and every existing asset check remains.
- gemma3 and timesfm keep an empty table list, model version 1 and their
  previous policy semantics.
- No claim that B2 is accepted or that the consumer closure is complete.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `s7-context-tests.txt`, `provider-tests.txt`,
`policy-tests.txt`, `smoke/`, `NOTES.md`.
