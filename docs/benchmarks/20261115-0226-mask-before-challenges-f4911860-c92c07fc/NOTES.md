# mask-before-challenges

- Scheme: mask-before-challenges (TURN 542 R2: the segment mask and the
  at_start index-mask are fixed and absorbed before the PSC/merge challenges
  that read them).
- Based on: the request layout freeze (commit 29c442bf, trace c92c07fc).
- Purpose: correct the first-chain order so the mask commitments precede the
  challenges they bind, and split the segment source cells out of the mask
  table so the mask commitment carries no segment output.
- Change: prove_first/verify_first absorb the at_start index-mask root before
  the segment; prove_first_chain_segment/verify_first_chain_segment commit and
  absorb the segment mask table (root, sample seed/answer, positions and
  opening on the verifier) before running the token/stack/merge. The segment
  source cells (stack read cells and the kept owner entry) are a separate
  committed value message: committed after the segment and before the section,
  with its own root/sample, own member, sample row and tree source. The mask
  table carries only mask entries; its slot layout is derived from the chain
  inputs and the table capacity. The relation rows take A0/B0/kept from the
  source member and the mask coefficients from the segment mask member.
- Key numbers: workspace 1094 passed / 0 failed (workspace.log); merged tests
  112 passed; hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb.
- Conclusions: the mask commitments are fixed before the challenges; the
  index-mask root and the segment mask root are both absorbed before the
  segment; the source cells are a separate committed object fixed before the
  section and the seal.
- Boundaries: R1/R3-R5 remain open (the verifier's direct-callback path, the
  stack rowS relation, the real source-object equality binding, and the real
  diagnostic path); the source object carries the fixture's read cells until
  R4 binds them to the committed source objects.
- Not pushed.
