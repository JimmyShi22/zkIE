# Encoded asset table identity and commit binding

## Scheme and baseline

- Scheme: `asset-table-identity` (B1 of the B work package).
- Code: `f741cba2`; previous commit `91ac3362` (A acceptance closure);
  branch `refactor/v2`.
- Scope: the general asset table identity structure, the encoded table base
  over the read-only asset rows, and the commit artifact trusted binding. The
  consumption closure (B2), the column sampling/fold/proximity tests (B3), the
  public LUT source (B4) and the plaintext removal plus five-case acceptance
  (B5) are not part of this step and are not claimed.

## What changed

- New module `crates/zkie-ops/src/proof/asset_table.rs`:
  - `AssetTableParams`: the trusted plan expression of one asset table —
    asset identity and encoding, table shape (code parameters, rows, answers,
    salt bytes, distances), valid value range, and layout (row width, message
    length, tail length, pads, valid elements, zero padding).
  - `AssetTableLayout`, `AssetTableRandomness` (private salts and tails),
    `AssetTableInput`, `AssetTableMeta` (public committed table).
  - `build_asset_table` streams the same read-only asset rows through
    `AssetReader::read_row` into the existing encoded table base
    (`table.rs::commit_table_streaming`). Each row is mapped with
    `fp_from_signed_word`, the plan's valid range is checked on real elements
    only (padding is zero), and the table's Merkle root and out-of-domain
    record are produced by the existing base.
  - `asset_table_configuration` derives the table's hash-start digest from the
    trusted model identity and the plan parameters, never from the
    committer's data. `asset_table_identity` covers the plan identity;
    `asset_table_statement_identity` covers the plan identity, the table root
    and the record.
  - `verify_asset_table_meta` independently checks the committed identity,
    shape, configuration, valid range, record structure and non-zero root
    against the trusted plan.
- Commit: `commit_assets_with_tables` builds one encoded table for every
  planned asset. The table leaf record (`leaf_id = specs.len() + index`,
  statement identity, table root) joins the anchor registry, so the anchor
  binds the plan identity, the table root and the record. The raw row
  commitment and every existing check are unchanged.
- Public artifact: the provider commit writes `public/asset-tables.bin`
  (bincode of `Vec<AssetTableMeta>`) and lists it in the commit manifest.
  `public/model-commitment.bin` keeps its format, so historical bundles and
  fixtures still deserialize.
- Verify: `verify_model_anchor_with_tables` checks every committed table
  against the trusted plan and rebuilds the anchor including the table leaves.
  `verify_model_anchor` delegates with no trusted tables and rejects a bundle
  that carries table leaves. The provider verify reads `asset-tables.bin`
  when present (absent means empty) and calls the table-aware check with the
  trusted plan's table parameters.
- The production model plans pass no table inputs in this step, so no
  production anchor, model id, proof byte or proof semantics change.

## Four-column mapping

| Requirement | Trusted plan expression | Change | Evidence |
| --- | --- | --- | --- |
| General asset identity, not the limited output-head identity | `AssetTableParams` per asset (`asset_id`, encoding, shape, valid range, layout) | new `asset_table` module; `LimitedTableIdentity` untouched | positive commit/verify; identity and shape negatives |
| Reuse the existing encoded table base | existing `table.rs` shape/code/record/commit primitives | `build_asset_table` calls `commit_table_streaming`; no second table implementation | asset_table tests; `s7_context` still green |
| Trusted identity and independent verification | shape, code, answers, salt bytes, distances, valid range, layout, configuration | verifier recomputes identity and configuration and checks the committed meta | changed root/configuration/identity/shape/code/record/range/asset-id/path all rejected |
| New table root enters the commit artifact trusted binding | model identity + plan parameters | table leaf record joins the anchor (statement identity + root) | changed root or record breaks the anchor; duplicate/removed table rejected |
| Real valid range, padding and layout | `valid_min`/`valid_max`, `valid_elements`, zero padding, row/message/tail/pads | commit checks the range on real elements, zero-fills padding | out-of-range commit rejected; params-vs-asset mismatch rejected |
| No shortcut removal and no second path | raw row commitment stays | production plans pass no table inputs; raw checks unchanged | workspace green; no proof-byte claim |
| Hidden material stays private | caller-supplied salts/tails | public meta carries only root/record/identity; randomness not serialized | meta fields checked; randomness count/length negatives |
| Hard gate and workspace | — | no hidden path change | 45,080 B sha256 identical; workspace result below |

## Tests

- `cargo test -p zkie-ops --test asset_table`: 5 passed.
  - positive commit, public bundle round trip, anchor binding;
  - changed root, configuration, identity, shape (v and code), record, valid
    range, removed table, duplicated table, changed asset id and path are all
    rejected;
  - out-of-range asset value rejected at commit;
  - wrong salt or tail counts rejected at commit;
  - plan parameters that do not describe the asset are rejected.
- `cargo test -p zkie --test s7_context`: 3 passed (the historical public
  bundle still deserializes; the new binary reads it).
- `cargo test -p zkie --test verify_policy`: 7 passed.
- `cargo test --workspace`: result below.

## Hard gate

The norm-free toy proof is 45,080 B with sha256
`7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`, identical
to the A package reference by `cmp`. The dump was produced by a temporary
local write in the test that was reverted; the tree was clean afterwards.

## Workspace

`cargo test --workspace`: 49 suites, 978 passed, 0 failed, 17 ignored
(`workspace.txt`).

## Boundaries and gaps

- This step supplies the structure and the binding only. Production model
  plans do not declare table parameters yet; no production asset plaintext is
  removed, no proof bytes change, and the B acceptance is not claimed.
- The table parameters (code shape, rows, answers, salt size, distances) are
  plan inputs. The fixture parameters are test values, not a security claim;
  production parameters, distances and query counts belong to the fold and
  proximity planning (B3).
- The salts and tails are caller-supplied private material in this step;
  production randomness generation and private carriage are wired with the
  consumer (B2/B3).
- The commit artifact gains `public/asset-tables.bin`; the model commitment
  bundle format is unchanged, so historical bundles remain readable. The
  anchor identity changes only when table leaves exist.
- The identity/table module reuses the shared table base and anchor; no second
  table implementation and no hidden-norm change. Coordination with the
  parallel input-binding line's production consumer is part of B2.
- No push.

## Artifacts

`identity.txt`, `change.diff`, `asset-table-tests.txt`, `s7-context-tests.txt`,
`verify-policy-tests.txt`, `workspace.txt`, `proof-bytes.txt`, `NOTES.md`.
