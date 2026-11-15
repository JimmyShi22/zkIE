# shared-lin-bounds

- Scheme: shared-lin-bounds (TURN 566 item 3: the shared linear stage's
  query-count and error-bound layer).
- Based on: the Omega opening count (305b0bb4) and its trace (fa391a72);
  previous baseline commit fa391a72.
- Purpose: derive the shared linear stage's column query count and its three
  error terms from the table's tested distance, reusing the interval planner
  base; do not derive the fold round count or the hiding polynomial.
- Change:
  - `SharedLinShape` carries the trusted parameters: the table's row
    variables, the shared point's low variables, the instance count and the
    comparison point's variables.
  - `shared_lin_bounds` returns the fold point's error (the instance count
    times the field error over v + k_lo plus the folded table's error: v
    times the per-variable proximity error plus the list size times the field
    error over v), the smallest column count whose agreement power meets the
    query target, that power as the column round's error, and the field error
    over the comparison point's variables.
  - The function refuses a zero instance, row-variable or
    comparison-variable count, an unreachable agreement, and a fold point
    that does not meet the closeness target.
  - Test: the three terms against the same bound algebra; refusal cases.
- Numbers:
  - focused: 1
  - workspace: 1128 passed / 0 failed / 20 ignored over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries: only the query-count/error-bound layer is derived; the fold
  round count and the hiding polynomial are not. The derived numbers are a
  planning result, not a formal soundness proof, and are not yet used to
  replace a registered plan value. LATER stays ONE_TIME-only: current
  support is ONE_TIME; LATER is not enabled. Not pushed.
