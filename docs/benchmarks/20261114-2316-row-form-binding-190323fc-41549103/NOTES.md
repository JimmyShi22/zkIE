# row-form-binding

- Scheme: row-form-binding (B2 request chain, TURN 540-C1 V2: the sub-proof row
  forms land on their owning committed members).
- Purpose: convert a masked sumcheck's row claim (hiding coefficients, counted
  mask slots, kept-map coefficients, public constant) into one plain row on the
  owners the trusted binding names, so the token kept, stack terminal
  recombination and merge remaining relations are checked by the owning group.
- Change: segment_slots::RowBinding (mask member, virtual entry per hiding
  entry, virtual entry per mask slot, owner and entry per kept value) and
  row_form_row, which emits mask units for the hiding and slot coefficients,
  main units for the kept coefficients and sets the form constant as the row
  target; an uncovered hiding entry or slot is refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; module
  tests 4 passed; workspace numbers in workspace.log.
- Conclusions: every coefficient of a row form now has an explicit (member,
  entry) owner; no slot or kept index can be read against another object.
- Open items: fill the bindings from the trusted layouts (sub-proof mask member
  vs request index-mask; reduction-table member for the stack sources), then
  the first-chain segment order (V1), the transcript and version bump (V3) and
  member_rows_fit (V4).
