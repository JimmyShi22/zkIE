# Frozen interface of the norm gain and bias parameter tables

- Scheme: `norm-param-interface`
- Change: code commit `c0ee68bd` (new module `norm_param_table.rs` with the
  parameter table identity, the two-table plan, validation, digests and unit
  tests; module registration only). The interface-freeze document lives outside
  the repository in the task records.
- Baseline: previous commit `f5ee6898` (acceptance trace).
- Purpose: freeze the normative parameter-table identity/mapping, the asset
  binding conditions, the opening/ledger accounting and the transcript order
  before the norm output relations are wired (TURN 498: conditional fixture).

## What changed

- `NormParamTableIdentity`: version, parameter kind, external model identity,
  asset-table identity, committed encoded-table reader, parameter row, vector
  width, admitted i32 range and the explicit mapping-assumption flag. The
  digest covers the reader identity and every field; validation rejects wrong
  versions, zero identities, malformed tables, out-of-range rows and widths,
  and non-i32 ranges.
- `NormParamTablesPlan`: the gain and bias identities, the per-table column
  draw count and the per-table seed count. Two identities with the same reader
  are two rows of one committed table; different readers are separate tables.
  Duplicate rows in one table, swapped kinds and out-of-budget counts are
  rejected.
- Unit tests: field binding and digest changes, malformed-field rejection, and
  the packed/separate-table plan rules.
- No table is built, no challenge is drawn and no relation consumes an
  evaluation in this step.

## Evidence

- Focused: `norm_param_table` 3 passed (`focused.txt`).
- Workspace: 48 suites ok, 897 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 496/498) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| gain/bias committed once, shared across positions | one identity per parameter; the plan allows one packed table or two tables | unit tests for packed and separate plans | fixture uses an external trusted table identity |
| table identity/mapping fixed before challenges | identity digest binds model/asset/reader/row/width/range/flag | digest-changes test | no transcript wiring yet |
| asset binding conditions | identity fields separate the model identity and the asset-table identity from the encoded-table reader | validation tests | production source closure is a recorded dependency |
| opening/ledger and transcript contracts | frozen in the task-record document (counts, order, physical-tree accounting) | document; plan counts validated | ledger categories are wired in the next step |
| relation consumption contract | frozen in the document (`NormParamEval`, product/bias hooks) | document | not implemented yet |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 3 tests.
- Workspace: 897 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Wire the parameter-table identities into the hidden norm plan and messages;
  the table build from the read-only asset rows; the LIN/FOLDEQ evaluation
  consumption; the ledger categories; the transcript order.
- The norm output relations (product, rounding, bias, clamp, padding) and the
  hd group follow the frozen contract.
- Production source closure (normative commitment or verifiable content and
  mapping binding) remains a recorded dependency; the i32 mapping flag is an
  assumption, not a range proof.
- No push in this round.
