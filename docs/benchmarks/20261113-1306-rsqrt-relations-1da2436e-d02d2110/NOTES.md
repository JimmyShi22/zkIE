# Rsqrt relation maps and hidden row binding

- Scheme: `rsqrt-relations`
- Change: code commit `1da2436e` (relation indices 16..25, rel_map branches,
  hidden row binding, index-bound plan value, rsqrt witness plumbing).
- Baseline: previous commit `d02d2110` (rsqrt row layout trace).
- Purpose: express the segmented rsqrt index relation on the existing dot-PSC
  mechanism and bind it to the committed round message, with the rstd table
  length as a trusted plan value.

## What changed

- `NORM_BOUND_RELATIONS` now admits indices 0..25.
- `NormRoundPlan` carries the trusted `table_real_len`; it is validated,
  encoded and digested with the rest of the round plan.
- `rel_map` accepts the private `RsqrtRow` and builds relations 16..25:
  - 16: `variance + 2^17 - 2^18*raw - raw_remainder = 0`;
  - 17: `raw_remainder + raw_remainder_slack = 2^18 - 1`;
  - 18: selector Booleanity `s*(1-s) = 0` for the three selectors;
  - 19: one-hot `fine + middle + coarse = 1`;
  - 20: `fine*(FINE-1) - fine*raw - fine_slack = 0`;
  - 21: `middle*raw - middle*FINE - 256*q - r = 0`;
  - 22: `middle*(COARSE_END-1) - middle*raw - middle_slack = 0`;
  - 23: `coarse*raw - coarse*COARSE_END - 65536*q - 256*hi - lo = 0`;
  - 24: `index = fine*raw + middle*(FINE+q_middle) +
    coarse*(COARSE_END+q_coarse)`;
  - 25: `index + index_slack = table_real_len - 1`.
- The hidden row binding expands every scalar value through its committed
  round digit block (base-256 weights), binds the selectors, the index value
  and the variance member, folds the constant entries into the row target and
  adds the shown-mask units exactly as the existing relations do. Empty terms
  are not emitted.
- The fixture computes one `RsqrtRow` per position from the committed
  variance and the plan table length, commits the extended round message with
  `values_with_rsqrt`, and the first-message prover check uses the same
  values.
- A unit test checks every relation map sums to zero for raw values at
  `FINE - 1`, `FINE`, `COARSE_END - 1`, `COARSE_END` and beyond, and
  around the rounding half step. Two sign errors in the middle and coarse
  quotient terms were caught by this test and fixed in both the map and the
  binding.
- The hidden fixture keeps its previous relation list. Enabling the new
  relations makes `prove_second` fail in the proximity transcript
  (`InvalidProof`) after the norm PSC succeeds; the same failure appears when
  the seventeenth relation is a duplicate of relation 0, so it is a
  count-dependent fixture/transcript limit, not the new relation equations.
  This is recorded as the integration blocker.

## Evidence

- Focused: hidden norm 27 passed, norm-named hidden tests 28 passed; the new
  relation-map test passes at all segment boundaries.
- Workspace: exit 0 (see `workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (see `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 492) | Change | Evidence | Boundary / open |
| --- | --- | --- | --- |
| A: raw rounding quotient/remainder and no-wrap | relation 16; raw and remainder digit blocks | map test around the half step | relation not enabled in the hidden fixture yet |
| A: one-hot selectors with strict and non-strict segment bounds | relations 18, 19, 20, 21, 22, 23 | map test at FINE and COARSE_END; one-hot and slacks | same |
| A: per-segment index formulas bound to one committed raw/position | relations 24, 25; index value and bound slack | map test at every boundary; index against the runtime formula in the row test | same |
| B: prove 0 <= index < N | relation 25 with the trusted plan table length | row constructor rejects index >= N; map test uses a table length covering all boundaries | relation not enabled in the hidden fixture; the rstd table view is not wired into the lookup key plan yet |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused test wall time: 0.02 s + 0.44 s.
- Workspace: exit 0.
- No wall time or memory claim is made.

## Open items

- Resolve the count-dependent proximity transcript rejection before enabling
  relations 16..25 in the hidden fixture.
- Wire the rstd table view (already implemented and tested) into the norm
  lookup key plan: key column 0 = committed index value, key column 1 =
  committed table value; bind the table identity and real length.
- Negative matrix through the real hidden verifier and the acceptance batch.
- No push in this round.
