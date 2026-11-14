# public-verifier-and-rows

- Scheme: public-verifier-and-rows (TURN 542 R1 and the verifier-side part of
  R3: the formal verifier reads no private mask value, and the stack's
  remaining row is landed with its hiding region).
- Based on: the mask-before-challenges step (commit f4911860, trace b5922749).
- Purpose: make the formal request-chain verifier depend only on the public
  plan, the received proof, the committed objects and their slots and public
  coefficients; keep the direct-evaluation checks as a prover/fixture helper;
  complete the segment row set with the stack remaining row.
- Change: verify_request_chain verifies the token and stack sumchecks and the
  merge from the received shown end values, their public factors and their
  slots; it no longer calls group_from_ends, group_bases or RowForm::check and
  never reads the caller's mask closure. verify_request_chain_direct keeps the
  old direct checks for the prover and the fixtures, and the unit negatives
  that exercise mask sources and read cells use it. The segment slot layout
  gains a stack hiding region; chain_relation_rows now returns five rows: the
  token kept row, the stack remaining row (RowForm::from_output with its
  hiding coefficients and count-weighted claim slots, including the token
  start's hiding slots), the A0 and B0 source rows and the merge remaining row.
- Key numbers: workspace 1094 passed / 0 failed (workspace.log); merged tests
  112 passed; hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb.
- Conclusions: the formal verifier path is free of the private mask closure;
  the stack's product/sum remaining form enters the committed relation rows.
- Boundaries: the first-chain segment closure and its proximity test (the
  remaining half of R3) are not in this step: the five rows are still checked
  by the second-chain own proximity. A first-chain segment proximity needs its
  own registered mask tree, a proximity message codec and a transcript segment
  before the section, and is not claimed here. R4 and R5 remain open.
- Not pushed.
