# packed-array-root

- Scheme: packed-array-root (the packed symbol array's commitment and its
  plan-positioned opening).
- Based on: the packed symbol array geometry (code cc61e9f1, trace 83a81960).
- Purpose: commit the original compute matrix's packed symbol array to a root
  fixed by the plan, and open it at planned positions only; the range
  transcript and the tie/X4 relations come after this object exists.
- Change:
  - `commit_packed_array`: one main row of the packed entries, padded with
    zeros to the tree's columns, under the plan's runtime tree; every packed
    value must be in `[0, 2^60)` and is embedded as a base-field element, not
    reduced. A wrong entry count, a width beyond the tree or a wrong salt
    count is refused.
  - `verify_packed_opening`: verifies the runtime opening against the plan,
    tree and root, requires the opening's positions in the planned order and
    the `E_DEGREE` width, extracts the base-field value, refuses a non-base
    component and returns the packed values.
  - Unit tests: the commitment opens at planned positions with the packed
    values; a changed root and a changed position are refused; a value above
    the packed bound is refused at commitment.
- Key numbers: focused tests 3 passed (packed2-step.log); workspace 1099
  passed / 0 failed (workspace.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the packed array has a committed root and an opening check at
  plan positions; no protocol or plan message change yet.
- Boundaries: the range transcript and its ZZ combination object (before
  t2Batch), the tie and X4 relations after the U3 commit point, the segment
  placement and the kept owner binding are not in this step. Positions are
  plan-chosen, not caller-chosen, and no full table is handed to the
  verifier.
- Not pushed.
