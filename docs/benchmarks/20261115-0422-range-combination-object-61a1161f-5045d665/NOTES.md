# range-combination-object

- Scheme: range-combination-object (the range transcript's combination
  addresses, fixed before the batch challenge).
- Based on: the packed array root (code 5045d665, trace 77d8ce02).
- Purpose: give the range transcript one addressing and digit-splitting object
  for the original compute matrix's per-column combination values, so the
  prover and the verifier agree on the stored columns before the batch
  challenge; the tie and X4 relations then refer to the same object.
- Change:
  - `RangeCombinationPlan`: the column count, the combination plane count, the
    digit width and the offset. One column of plane `plane`, original column
    `index`, is addressed as `plane * columns + offset + index`; the digit base
    is `2^low_bits`; the plan refuses zero geometry and width out of range.
  - `split`: a nonnegative combination value becomes its low and high digits
    and the integer identity `value = low + d * high` is checked; a negative
    value is refused.
  - Unit tests: address order for both planes, out-of-range plane and index,
    the split at zero and high digits, the identity, a negative value and the
    zero/width geometry negatives.
- Key numbers: focused tests 4 passed (zz-step.log); workspace 1100
  passed / 0 failed (workspace.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the range combination object's addresses and digit split are
  fixed by plan data; no challenge is drawn by this object and no full table
  is handed to the verifier.
- Boundaries: the combination values themselves, their commitment and range
  messages, the tie and X4 rows, the segment placement after the U3 commit
  point and the kept owner binding are not in this step.
- Not pushed.
