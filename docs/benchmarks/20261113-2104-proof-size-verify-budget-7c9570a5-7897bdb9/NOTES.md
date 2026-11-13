# Verify decode budget sized from the proof file

## Scheme and baseline

- Scheme: `proof-size-verify-budget`.
- Code: `7c9570a5`; previous commit `7897bdb9`.
- Small repair requested during interval 1 review: after the bounded trace
  decoder was merged, the verify command used the one GiB default decode
  budget and rejected any proof file above one GiB before verification.

## What changed

- `zkie/src/verify.rs`: after parsing the request the verify command sizes its
  decode budget from the proof file. The budget becomes twice the file size
  plus one GiB when that is above the default; a small proof keeps the default.
  The size comes from the proof file metadata, and the decoder still enforces
  its own budget and allocation guard. No proof format, statement, planner or
  verifier semantics change.

## Results

- Direct CLI verify of the production proofs: the path-form proof passes
  (239.4 s, VmHWM 5,389,096 KiB) and the pathless proof passes (109.3 s,
  VmHWM 4,782,100 KiB); both bind the expected output digest
  `135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280`.
- Matrix (gpt2-seq16, cold and warm): prove PASS and verify PASS for both
  runs; proof bytes 3,564,380,854; verify binds the expected digest.
- `cargo test --workspace`: 48 suites, 969 passed, 0 failed, 17 ignored,
  exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  `cmp` identical (`proof-bytes.txt`).

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| Matrix verify works for proofs above one GiB | decode budget scaled from the proof file, default kept for small proofs | matrix cold/warm verify PASS; direct verifies PASS | the decoder budget and its allocation guard still apply |
| No semantic or format change | only the runtime budget value | expected output digest bound; workspace and gate unchanged | — |

## Scope and next

- This commit is separate from interval 2 and does not change any proof
  semantics, planner behavior or verifier check.
- No push until TL review.

## Artifacts

- `matrix.json`, `verify-cli.txt`, `workspace.txt`, `proof-bytes.txt`,
  `change.stat`, `NOTES.md`; the matrix proof tree stays on disk with binaries
  gitignored.
- No textual trace was deleted.
