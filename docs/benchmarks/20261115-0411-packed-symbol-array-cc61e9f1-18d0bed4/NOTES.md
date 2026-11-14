# packed-symbol-array

- Scheme: packed-symbol-array (the original compute matrix's symbol array, the
  missing object before the tie/X4 relations).
- Based on: the request source layout step (code 18d0bed4, trace 2fc8f16f).
- Purpose: represent the original compute matrix as one committed symbol
  array whose cells are packed along rows, with the packing and position
  geometry fixed before any challenge.
- Change:
  - New module `proof::packed_matrix`: groups of at most three consecutive
    rows are packed per column as `A(3s,c) + B*A(3s+1,c) + B^2*A(3s+2,c)`
    with `B = 2^20`; the position is `(s/J, c*J + s%J)` with one place per
    group (`J = 1`); the group count is `ceil(rows/3)`.
  - Every cell must be in `[0, 2^20)`; the packing is an integer identity, not
    a field reduction, so an out-of-range cell is refused. The packed bound is
    `2^60 - 1`. Wrong cell counts and zero geometry are refused.
  - Unit tests cover the group/column order, a full and a partial group, the
    position map, out-of-range and negative cells, the wrong size and zero
    geometry.
- Key numbers: focused tests 2 passed (packed-step.log); workspace 1098
  passed / 0 failed (workspace.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the original compute matrix has a fixed packed symbol-array
  geometry and integer identity; no protocol or plan change yet.
- Boundaries: the commitment of the array to a root, the range transcript and
  the ZZ combination object, the tie and X4 relations, the segment placement
  after the U3 commit point and the kept owner binding are not in this step.
  No duplicate weight table, no plaintext full table, no field reduction of
  out-of-range cells.
- Not pushed.
