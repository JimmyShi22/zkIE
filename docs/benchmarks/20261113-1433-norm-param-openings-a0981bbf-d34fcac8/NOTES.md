# Norm parameter table openings drawn and verified in the first stage

- Scheme: `norm-param-openings`
- Change: code commit `a0981bbf` (first-stage position draw, prover table
  handle, opening verification against the plan readers; packed/separate bias
  rules; fixture tables and five tests).
- Baseline: previous commit `d34fcac8` (message carrier; trace `c50804a9`).
- Purpose: authenticate the actual parameter table columns before any dependent
  challenge, with the positions produced by the transcript after the table
  identity is fixed.

## What changed

- A prover-side parameter table handle opens the drawn columns of the gain and
  (when separate) bias tables.
- The first norm stage draws the column positions from the chain after the
  table identity is absorbed, opens them, and the verifier redraws the same
  positions, compares them with the message and verifies the openings against
  the plan's committed readers.
- Packed tables reuse the gain openings; separate bias tables carry their own
  positions and openings. Missing bias openings under a separate table are
  rejected.
- The witness parameter vectors are length checked against the padded column
  domain.
- The fixture builds two small encoded tables through the streaming committer
  from bounded row sources (one row of parameter values plus padding, one salt
  per column, random tails from the fixture coins) and commits each once.
- Tests: a positive chain with real openings; four negatives through the real
  verifier (changed position, changed symbol, missing bias openings, changed
  reader).

## Evidence

- Focused: norm 36, rstd 16, lookup 24, norm_param_table 3 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 901 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 498) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| sampling positions from the transcript after the root | chain-drawn positions; verifier redraw and compare | changed-position negative | the ledger entry is at the zkie path |
| opened columns authenticated | verify_columns against the plan readers | changed-symbol negative | the evaluation consumption is next |
| gain/bias one commitment, shared across positions | one table per kind in the fixture, shared by both positions | positive chain | packing variant not exercised |
| table identity fixed before dependent challenges | identity in the plan digest; draw after absorption | changed-reader negative (plan digest binding) | no ledger categories yet |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 79 tests across the four filters.
- Workspace: 901 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Ledger categories (logical/unique/physical counts, physical-tree union) are
  wired at the zkie path; the ops fixture has no ledger object.
- LIN/FOLDEQ evaluation consumption and the output relations; the hd group.
- Production source closure remains a dependency; the i32 mapping flag is an
  assumption.
- No push in this round.
