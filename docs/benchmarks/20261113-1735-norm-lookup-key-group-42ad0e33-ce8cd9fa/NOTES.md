# Lookup key group beyond the rounding layout

- Scheme: `norm-lookup-key-group`
- Change: code commit `42ad0e33` (the norm lookup plan carries its key
  group: first oracle, message shape and column count; the lookup view exposes
  it; the key-claim rows and the prover key-row selection use it).
- Baseline: previous commit `ce8cd9fa` (trace of the slack lookup plan
  carrier `e9524624`).
- Purpose: remove the rounding-layout restriction from the key sources so the
  same lookup machinery can address the committed tail group cells.

## What changed

- `NormLookupPlan` gains `key_oracle_base`, `key_shape`, `key_columns`,
  validated and absorbed into the plan digest; `from_round` sets them from
  the rounding group.
- `NormLookupView::key_group` exposes the group; the frozen-table view
  returns none and keeps the rounding fallback.
- `lookup_plain_rows` builds the key-claim members from the plan's key group
  (range, shape, columns) instead of the rounding group.
- The prover selects the key rows by the plan's key group: tail messages when
  the group is the tail group, rounding messages otherwise.

## Evidence

- Existing lookup tests (25) and the full output chain pass unchanged.
- Focused: hidden_norm 37, norm_output 1, rstd 16, lookup 25,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 913 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 504 一①) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| key sources address the committed slack cells | key group on the plan and rows | chain + lookup tests | the tail-keyed plan instance is not built yet |
| table identity fixed before challenges | unchanged digest binding | lookup tests | - |
| claims in BAT/FOLD/CS/BASE | key-claim members from the key group | chain pass | - |

## Open items

- The tail-keyed lookup plan instance and its multiplicity-bit witness,
  commitment messages, samples and openings in the fixture; the mask/ledger
  budget derivation; the plan/bind version bump.
- The clamp five-point acceptance and the negative matrix; the padding
  negatives; the hd binding.
- No push in this round.
