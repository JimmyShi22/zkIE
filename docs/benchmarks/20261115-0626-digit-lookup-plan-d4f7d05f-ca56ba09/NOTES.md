# digit-lookup-plan

- Scheme: digit-lookup-plan (the two digit lookups of the range object, plan
  level).
- Based on: the lookup chain (code ca56ba09, trace 4ee52dd8).
- Purpose: the range object's low and high digit cells must be rows of public
  tables; this step plans the two single-call lookups over the committed digit
  message before the segment wiring.
- Change:
  - New module `proof::digit_lookup`: `LowDigitTable` (rows `0..=1023`, its own
    trusted identity and content root), the digit widths (`LOW_DIGIT_LOG = 10`,
    `HIGH_DIGIT_LOG = 8`), the multiplicity planes (`DIGIT_MU = 5`, 32 planes)
    and the key-count bounds (at least 10 bits, at most 22).
  - `DigitLookupPlan { plane, lookup, key }` and `digit_lookup_plan(range,
    plane)`: `D = combinations * 2 * cell_rows()` real keys in dense order
    (`place * combinations + combination`), one source per cell on the
    committed digit message (oracle, row, place, coefficient one, offset zero),
    the key vector's tail padded with the public value zero, the low table's
    identity on plane 0 and the byte table's on plane 1.
  - `digit_multiplicity_bits`: counts each table row over the plane's digits,
    adds the padding keys to row zero so the product trees carry no extra
    factor, and writes the 32 bit planes; a digit outside the table's width is
    refused, never reduced.
  - Tests: every cell enumerated exactly once, the padding count is `N - D`,
    a plane/table identity mismatch is refused, the row counts include the
    padding keys, out-of-table digits are refused on both planes and a wrong
    key count is refused.
- Key numbers: workspace 1108 passed / 0 failed (workspace.log); hard gate
  45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the two calls' parameters, key enumeration, padding convention
  and multiplicity layout are fixed as plan data with their own table
  identities; the enumeration is trusted and complete.
- Boundaries: the calls' commit plans, messages, challenges, rows and their
  consumption by the segment proximity group are not wired yet; no smallness
  or range closure is claimed. The reference shares one forest across both
  calls; these are two independent single-call lookups and the difference is
  reported as ours. Not pushed.
