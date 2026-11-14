# first-chain-segment

- Scheme: first-chain-segment (B2 request chain, TURN 540-C1 V1: the segment
  runs on the request first chain transcript).
- Purpose: give the token/stack/merge segment its first-chain runner and fix
  the whole section in the chain before the seal.
- Change: ChainSegmentChallenges (chain-backed draws and absorbs with the
  segment's own tags 24/25/26), prove_first_chain_segment /
  verify_first_chain_segment (same order on both sides) and
  absorb_request_chain_section (tag 27, shape-checked section bytes). The
  fixation test shows equal sections reproduce the chain value and a changed
  stack proof changes it; the plan entry still refuses request_chain until the
  caller inputs are plumbed.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: the segment has a first-chain execution order and transcript
  fixation; no proof byte chooses the order.
- Open items: caller ChainInputs plumbing into prove_first/verify_first, the
  pending-to-member order before the LIN first W and seal, the negatives,
  W layout freeze, batch report.
