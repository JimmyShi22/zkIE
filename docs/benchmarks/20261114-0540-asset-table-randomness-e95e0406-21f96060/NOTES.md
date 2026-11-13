# Private randomness for encoded asset tables

## Scheme and baseline

- Scheme: `asset-table-randomness` (first wiring sub-step of the production
  commit migration).
- Code: `e95e0406`; previous commit `21f96060`; branch `refactor/v2`.

## What changed

- `zkie-ops::proof::asset_table::sample_asset_table_randomness` samples one
  fresh salt per column and one random code-tail element per row. Tail words
  are sampled by rejection so each element is canonical below the field
  modulus. The material stays private: the public meta carries only the table
  root and record.
- Unit test: the sampled salts and tails have exactly the table's column,
  row, salt and tail lengths.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Private salts and tails for one table | B1 `AssetTableRandomness`; hidden-session table prover | sampling function with getrandom and canonical rejection | unit test |
| Not in the public meta | B1 meta excludes randomness | no serialization change | code and test |
| Commit and prove wiring | `private/asset-table-randomness.bin` | **not yet** | gap |
| Anchor version gate and policy regeneration | `model_version` gate | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib asset_table`: 1 passed.
- No production path changed; no anchor or policy change; no push.

## Boundaries

- This is the randomness API only. The commit does not yet build the
  production encoded table, the private artifact is not written or read, and
  the anchor/model version and policies are unchanged. The production
  consumer closure remains the next work.

## Artifacts

`identity.txt`, `change.diff`, `randomness-tests.txt`, `NOTES.md`.
