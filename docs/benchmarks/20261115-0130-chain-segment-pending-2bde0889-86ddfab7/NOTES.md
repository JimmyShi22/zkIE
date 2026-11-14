# chain-segment-pending

- Scheme: chain-segment-pending (the request chain section runs inside the
  first chain and its pending claim becomes the request member).
- Based on: the first-chain segment wire (previous commit 86ddfab7); this run
  builds the caller bundle and binds the segment's output on both sides.
- Purpose: exercise the section positively with a caller bundle matched to the
  merged plan, and check that the member row is the segment's pending claim on
  both sides before the seal.
- Change: prove_request_chain now derives the pending claim from the merge's
  row weights and masked final value and returns it; the section entry and the
  first-chain segment pass it on; prove_first/verify_first refuse a plan member
  that differs from the segment output, and a section without a member is
  refused by the plan. The merged fixture runs one request chain section
  positively (a read panel carries the start constant, one weight panel leaves
  the merge value, the committed table's matrix cells are zero so the Eval
  identity holds for every draw) and adds four section-level negatives.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; merged
  tests 111 passed; workspace 1093 passed / 0 failed (workspace.log).
- Conclusions: the pending claim is fixed by the transcript, bound to the
  member after the segment and before the first W commit and the seal on both
  sides; a changed section changes the first chain value at that position. The
  prover and the verifier derive the same pending row.
- Deviation: the fixture uses one read panel plus one weight panel instead of a
  single panel. The stack's target is the start form's hidden constant; a
  single weight panel whose value is the merge claim can only match it when the
  claim is challenge-dependent. The read panel carries the constant and the
  weight value stays zero.
- Open items: the four remaining relation rows (token kept, A0-Alin, B0-Blin,
  merge remaining) need the sub-proof mask member (commitment message, sample,
  opening, read ledger, capacity) and its trusted owner mappings (hiding-entry
  regions, slot-to-entry list, kept-map owners, stack source owners). The mask
  slot/opening layout is still to be re-derived, and the W layout freeze stays
  open with it.
- Not pushed.
