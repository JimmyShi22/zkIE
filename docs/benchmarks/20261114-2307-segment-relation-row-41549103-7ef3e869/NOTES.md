# segment-relation-row

- Scheme: segment-relation-row (B2 request chain, TURN 540-C1 V2 first step:
  carry a settled segment relation as one owner-bound row).
- Purpose: convert the segment slot ledger's settled equation into a plain BAT
  row whose units carry the committed member's mask entries, so the segment's
  remaining relation is checked by the owning group's proximity test instead of
  a private mask evaluation.
- Change: segment_slots::relation_row(member, entries, mask_coefficients,
  constant) emits one unit per nonzero mask coefficient on the given member's
  mask part, with the row target equal to the negated constant; the entries are
  the trusted owner layout's virtual indices, one per mask slot. A mapping of
  another length is refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; module
  tests 3 passed; workspace numbers in workspace.log.
- Conclusions: the row carries the settled equation by (member, entry) identity;
  the unit test shows the settled equation equals the hidden form at the slot
  mask entries and the row target is the negated constant.
- Open items: map the token kept, stack terminal recombination, A0-Alin/B0-Blin
  sources and merge pscRow kept items onto their owning members (sub-proof mask
  vs request index-mask), then the first-chain segment order and transcript.
