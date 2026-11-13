# Frozen-table rstd lookup wiring

- Scheme: `rstd-table-lookup`
- Change: code commit `5edf0da8` (prove/verify first and second stages, trusted
  table-view binding, hidden fixture and negatives).
- Baseline: previous code commit `f47d984e` (carrier; trace `44dc5cd7`).
- Purpose: bind the committed index and rstd value pair into the frozen table
  by a two-column lookup inside the real hidden verifier, and make the table's
  trusted identity, content root, real length, encoding and version part of the
  verification input.

## What changed

- First stage: the rstd bit and mask groups are committed with a dedicated
  seed, the roots are absorbed in the fixed transcript order, and the samples,
  drawn positions and openings are registered with the other norm groups.
- Second stage: the rstd lookup seed is drawn at its fixed chain point; the
  lookup is proved and verified with the table view; the claims and rows join
  the hidden BAT, fold, mask budget, sample rows and retained trees.
- Key plan: column zero reads the committed index value slot, column one the
  committed rstd value slot, one key source pair per position, and the
  remaining keys are padded with the public first table row.
- Verifier binding: the lookup table trait exposes the trusted `TableConfig`;
  the table view check compares identity, content root, real length, encoding
  and version against the plan, streams the real rows and checks their
  canonical digest against the trusted content root, and compares the
  first-row padding pair. `RangeTable` provides the streaming content check;
  the byte table and the test pair table carry fixed configurations.
- Out of range: the rounding plan's trusted length N and the index-bound
  relation (index + slack = N - 1) reject index >= N; plan validation ties the
  table real length to N.
- Fixture and negatives: a synthetic ranged table with eight rows (values
  100 + 7 i, identity 0x81) and committed index/value slots matching the
  rounding rows; the positive chain plus nine negatives.

## Evidence

- Focused: norm 29 passed, rstd 11 passed, lookup 24 passed (`focused.txt`).
- Workspace: 48 suites ok, 889 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| TURN 492 / TL ② requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| key column 0 = committed index, column 1 = committed rstd | key sources read the round index and rstd value slots per position | positive chain; slot unit test; committed-value and committed-index negatives | the lookup covers the rounding rows of this local norm session, not a model-level plan yet |
| table identity, real length, encoding, version bound | plan digest carries the trusted config; the verifier compares the table object's config against it | identity, content and length negatives | the verifier must be handed its own trusted table object |
| content summary and actual reads agree | the table view check streams the real rows and compares their canonical digest with the trusted content root | padded-row negative (source claims the trusted metadata but returns a different first row) | one stream over the real rows per check; table IO is recorded by the view, not benchmarked here |
| index < N | plan validation ties the table real length to the rounding plan; the index-bound relation rejects index >= N | out-of-range index negative | N is the trusted rounding-plan length |
| message roots and claims bound | first/second wire fields, transcript order and outcome comparison | tampered bits root, mask root and key claim negatives | bind version 6 |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 64 tests across the three filters.
- Workspace: 889 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- TURN 492 acceptance batch: selector, rounding and segment boundaries,
  challenge reordering, plaintext absence, and the remaining matrix rows.
- Model-level plan assembly and the rounding claim linkage; 2b.4 inputs.
- No push in this round.
