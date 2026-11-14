# request-source-layout

- Scheme: request-source-layout (TURN 546 A, first step: locate the real
  request source through the trusted plan's layout conversion).
- Based on: the first-chain segment closure (code c85340d3, trace cf288413).
- Purpose: fix and evidence where the real L0 K request source comes from and
  how it maps onto the committed activation and table layouts, before the
  forward claim/maps to pending construction. The mapping follows the trusted
  trace plan; the file's linear index is not used as a table column.
- Change:
  - The consumer scan of the two real diagnostic tests is one helper driven by
    the trusted plan: the twelve Projection steps that read
    `gpt2/weights/L0_k_w_i32.bin` and the shared activation step they consume.
  - A new ignored test asserts the layout conversion on the registered seq16
    trace: the logical activation entry (token, input) is the committed row at
    `token * HIDDEN + input`, padded to the matrix rows; the logical weight
    entry (input, output) is the table cell at row output and place input,
    i.e. the message value the session reduces; the encoded column object is
    the same commitment the session opens; and each of the twelve heads'
    verified output is the requantized reduction over its own 64-wide slice in
    the plan's consumer order.
- Key numbers: focused layout test passed in 194.09 s (layout.log); workspace
  1096 passed / 0 failed (workspace.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the real source objects and their layout conversion are fixed
  from the trusted plan; no protocol, plan or message change.
- Boundaries: the read-table equality (X4 reads, random accumulation and the
  proximity check, with plan-sampled openings and no caller-chosen positions)
  and the forward claim/maps to merge to pending construction are not in this
  step. The pending is not reverse-fitted and no kept value is invented. R5's
  real entry and its evidence package remain open.
- Not pushed.
