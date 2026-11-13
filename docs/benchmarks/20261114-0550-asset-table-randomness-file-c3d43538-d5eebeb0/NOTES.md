# Encoded table randomness as a private artifact

## Scheme and baseline

- Scheme: `asset-table-randomness-file` (1b.2 foundation).
- Code: `c3d43538`; previous commit `d5eebeb0`; branch `refactor/v2`.

## What changed

- `AssetTableRandomness` is serializable. `encode_asset_table_randomness`
  produces the private bytes; `decode_asset_table_randomness` checks the
  decoded salts and tails against the trusted table parameters and rejects a
  truncated or mismatched private file before it can produce a table root.
- Tests: round trip, and a wrong tail length in the parameters is rejected.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Private carriage of salts and tails | B1 `AssetTableRandomness`; commit manifest private files | encode/decode with shape validation | two unit tests |
| Never public | public meta excludes randomness | no public field added | code |
| Commit provider writes/reads the file | `private/asset-table-randomness.bin` | **not yet** | gap |
| Model table input and version gate | gpt2 `build_commitment`, policy `model_version` | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib asset_table`: 2 passed.
- No production path, anchor or policy change; no push.

## Boundaries

- The private carriage is ready but unused by the commit provider yet. The
  model build, commit/prove wiring, version gate, policy regeneration and the
  consumer proof remain the next work.

## Artifacts

`identity.txt`, `change.diff`, `randomness-file-tests.txt`, `NOTES.md`.
