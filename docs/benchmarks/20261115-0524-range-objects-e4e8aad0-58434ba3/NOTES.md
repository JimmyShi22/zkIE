# range-objects

- Scheme: range-objects (the segment's range messages, their order and
  their tie/read relations).
- Based on: the range plan (code 58434ba3, trace 49183998/ab1f2889).
- Purpose: run the original matrix's packed symbol array, the reduced read
  table and the digit object as real SEND messages of the segment frame,
  with the ruled commit order and the ruled challenge coefficients, and
  close the tie and read relations in the segment's proximity test.
- Change:
  - The three objects are SEND value messages with their own oracle, tree
    and sample answers. The packed array is committed before the section's
    points; the read table after them and its root is the session's
    reduction root; the digit object after the combination matrix. Each
    root and its answers are absorbed before the challenges that read them.
  - The range matrix is the functional's read orientation (the table's
    columns are its cell rows). The table is streamed once: packed symbols,
    read-table entries, and the digit values. Cells are signed, must lie in
    (-2^18, 2^18), and are embedded as signed words; the packed array's
    opening recovers the sign.
  - The combination matrix is drawn from the chain as two-bit entries
    (first bit nonzero, second bit sign) after the read table's sample. The
    digit object keeps tau rows over 2*mz columns; the window value Z+2^17
    splits into ten low and eight high bits at the plan's places.
  - The batching element is drawn after the digit object. The tie and read
    rows use vt = vp = theta^u, g[c] = sum_u vt[u] Cf[c][u], sx = sv =
    sum vt, rowX[i] = (theta^Np)^i, wl = rowX[i/3] B^(i mod 3), wr = g, and
    the claim scale theta^(claims + first). The equations are
    sum wl (Zlo + d Zhi) - o sx sw - sum wr symbols = 0 and
    sum phi (Zlo + d Zhi) - o sphi sv - sum g U = 0; one tie per matrix,
    one read per table block, the counts from the plan.
  - The three objects join the segment proximity group with their sample
    rows and trees; a zero column weight is refused (the batched equations
    would not test that column).
  - The request layout counts the three range trees and their sample
    answers.
- Key numbers: focused tests 15 passed (request_chain 6, packed_matrix 5,
  chain segment 1, coefficient 1, layout 1, range codec 1); zkie-ops lib
  527 passed / 0 failed; workspace 1104 passed / 0 failed (workspace.log);
  hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the three objects are real messages with sample answers in
  the ruled order; the ruled coefficients hold on committed data (checked
  row by row on a nonzero matrix) and the end-to-end session closes the
  segment with the tie and read rows in the proximity test. Session
  negatives refuse a missing object, changed roots/answers and a read-table
  root that differs from the reduction root.
- Boundaries: the token kept row and the source message keep their previous
  fixture form; this batch does not claim the token kept's real owner or
  the full consumer chain (R4/R5) is closed. The digit lookup and its
  multiplicity object are not part of this implementation (neither before
  nor now; nothing is deleted). The combination count is restricted to a
  power of two in this batch. The real consumer's weight cells are i32 up
  to about 2^17.5, outside the reference's signed-small-cell window; a limb
  representation must be ruled before the real entry can run. The old
  paths keep their bytes.
- Not pushed.
