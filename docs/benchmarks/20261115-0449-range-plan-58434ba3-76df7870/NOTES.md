# range-plan

- Scheme: range-plan (the request segment's range objects: trusted plan,
  identity checks, messages and wire format).
- Based on: the range combination values (code 027cc6ac, trace 76df7870).
- Purpose: give the original compute matrix's packed symbol array and its
  low/high digit combination object a trusted plan binding and a plan-gated
  place in the session message, so the segment's tie and read relations have
  their objects and message fields before the ordering work.
- Change:
  - `RangePlan` (request_chain): the packed matrix geometry, the array
    oracle and one-row tree, the combination plan, oracle and tree, the
    group and the two plan opening counts. `fault()` requires each array to
    fit one main row of `E_DEGREE` symbols, the combination column count to
    equal the matrix's, the stored digits to fit the tree width, an opening
    list on each object and distinct identities.
  - `RangeMessage` carries its tree; the combination digits sit at the
    plan's offset; `verify_range_message` refuses a tree mismatch and any
    position before the offset or past the stored window.
  - `HiddenSendPlan.range`: validation binds the range geometry to the
    trusted matrix rows and columns and keeps the range oracles and trees
    disjoint from the per-position W rows, the dot map, the index mask, the
    segment mask table, the request W, the source message and the member
    witness. A chain plan that names the range objects is refused until the
    segment commits them.
  - `HiddenSendMessages.packed_array` / `.range`: plan-gated messages with
    tree, root, planned positions and opening; the decoder checks the tree,
    a non-zero root, the position count against the plan and the opening
    against the tree, and a plan carrying the objects refuses a message
    without them.
  - `REQUEST_CHAIN_VERSION` 2 -> 3; the decoder refuses the previous
    version before parsing the section.
- Key numbers: focused tests 12 passed (request_chain 6, packed_matrix 5,
  range message round trip 1); workspace 1103 passed / 0 failed
  (workspace.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the range objects have a trusted plan binding, disjoint
  identities and a plan-gated wire form; no challenge is drawn in this step
  and no full table is carried.
- Boundaries: the segment ordering (packed array before the section, the
  read table's root after the section's points, the digit object after the
  read table, the proximity last), the tie and read rows with their
  coefficients, the signed cell representation, the coverage and
  relation-count checks and the real consumer entry are not in this step.
  The old paths keep their bytes (no presence byte when the plan carries no
  range objects).
- Not pushed.
