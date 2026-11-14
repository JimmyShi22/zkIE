# range-combination-values

- Scheme: range-combination-values (the combination values and their message,
  fixed before the batch challenge).
- Based on: the range combination object (code 61a1161f, trace 6645b2bc).
- Purpose: generate the stored combination values from the original matrix's
  integer cells and public coefficients, commit them under the range plan's
  tree, and carry their plan-positioned opening as one message. The tie and X4
  relations then read the same object.
- Change:
  - `combination_values`: `offset + sum_row coeff[row] * A(row, column)` with
    integer checks; the coefficients are public inputs of the range
    transcript, every cell must be in `[0, 2^20)`, a negative coefficient or a
    negative cell is refused, and the value is never reduced.
  - `RangeMessage` and `commit_range_values`: the stored values (one per plane
    and column) are committed as one main row under the plan's tree, with the
    value bound `2^62 - 1`; the message carries the root, the plan positions
    and the opening.
  - `verify_range_message`: positions must be strictly increasing and inside
    the stored columns; the opening is checked against the tree and root and
    each value is split into its low/high digits with the integer identity.
  - Unit tests: the combination values, the split digits through the message,
    a repeated position and a negative cell refused.
- Key numbers: focused tests 5 passed (range-values.log); workspace 1101
  passed / 0 failed (workspace.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the combination values and their message exist as a
  plan-addressed, plan-positioned object; no challenge is drawn here and no
  full table is handed to the verifier.
- Boundaries: the transcript placement of the message before t2Batch, the tie
  and X4 rows, the segment's own group closure and the kept owner binding are
  not in this step.
- Not pushed.
