# segment-slots

- Scheme: segment-slots (B2 request chain, TURN 540-C1-D first building block:
  the verifier-side slot ledger of one sub-proof segment).
- Purpose: carry a sub-proof segment's hidden values as shown scalars and
  public factors, and settle the segment's one linear equation into mask
  coefficients plus a public constant without ever reading the private mask
  entries.
- Change: new module proof::segment_slots. hide registers a slot's shown value
  and public factor once; shown returns it; public_part evaluates a form on the
  shown values; push accumulates a weighted form into the segment equation;
  settle yields mask coefficients (-coefficient*factor) and the constant
  (segment constant plus coefficient*shown), refusing a coefficient on an
  unfilled slot.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  module tests 2 passed; workspace numbers in workspace.log.
- Conclusions: the settled equation equals the accumulated forms evaluated on
  the hidden values at any mask entries (unit test), and no path recovers a
  hidden value; this is the engine the first-chain segment relations use.
- Open items: carry the segment's remaining relations (token kept, stack
  terminal recombination, A0-Alin/B0-Blin sources, merge pscRow kept items) as
  BAT rows bound to the owning committed members; the first-chain segment
  order; transcript/version; member_rows_fit; TURN 536 C negatives.
