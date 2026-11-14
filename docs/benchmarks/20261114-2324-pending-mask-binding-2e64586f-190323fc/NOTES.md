# pending-mask-binding

- Scheme: pending-mask-binding (B2 request chain, TURN 540-C1 V2/E: the pending
  claim's mask slots bind to the index-mask owner through the trusted layout).
- Purpose: one binding for the pending mask terms so neither the request rows
  nor the pending relation reads a bare slot against another object.
- Change: pending_mask_entries(plan, members) checks each term names the plan
  member index oracle, resolves the owner member by that oracle and the slot
  through the allocation layout's virtual index; request_rows_bundle consumes
  the binding for its mask units.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace numbers in workspace.log.
- Conclusions: the pending claim and the request rows share one (member, entry)
  per slot; a term on another oracle is refused; the fixture binding matches
  the allocation layout.
- Open items: the sub-proof mask member owning the stack/token end slots (its
  commitment, message and ledger), then the first-chain segment order and the
  transcript/version.
