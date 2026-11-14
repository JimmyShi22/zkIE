# first-chain-segment-closure

- Scheme: first-chain-segment-closure (TURN 542 R3 and the TURN 544 state
  handoff ruling).
- Based on: the public-verifier-and-rows step (code 56509950, trace
  a4221f3f).
- Purpose: close the request segment's relation rows inside the first chain
  with the segment's own proximity test, and continue the same first-chain
  transcript into the member binding and the second chain on both tails.
- Change:
  - The segment runs on the first chain in the fixed order: group, claim,
    reduction root, request index-mask root, segment mask table root/sample/
    opening, request sub-proof challenges, source-cell root/sample, segment
    proximity over the five relation rows, section absorb. The segment mask
    table and the source-cell table are two separate committed objects; the
    mask table carries no segment output and is fixed before every challenge
    that reads its slots.
  - HiddenSendPlan gains `segment_mask_tree` and HiddenSendMessages gains a
    plan-gated `segment_proximity` section carried by the existing fold-start
    and tail codecs. The segment mask tree is its own fresh tree; validation
    ties its log length to the chain mask code, its row count to the padded
    chain mask rows and its tree number to the other registered trees.
  - The segment members are the committed chain mask table and the committed
    source-cell table; sample rows are the received sample answers and the
    plain rows use public coefficients plus committed mask/source units only.
    The five relation rows are the token kept row, the stack remaining row,
    the A0 and B0 source rows and the merge remaining row.
  - ProximityProof and ProximityVerification carry the advanced `Chain`.
    BaseFinalSession::finish_with_chain and LastBaseSession::into_chain hand
    the same transcript back, and the multi-stage prove/verify entries return
    it as well. No digest replacement, no second chain and no new seal or
    binder message are introduced.
  - The check-state fields are renamed `state_rounds`/`state_value` and
    documented as the chain state at the point this proximity check completed,
    not the first chain's seal.
  - Fixture: segment mask tree 203 with four rows is enabled in the segment
    and layout tests; every other plan carries None. New negatives refuse a
    changed fold start, a changed fold round, a changed segment seal, a missing
    segment proximity message and a changed source sample answer. Two new
    tests cover the staged-tail handoff (prover and verifier continue the same
    chain value and round count) and the sealed chain handoff (an unfinished
    session returns an error, a sealed session returns a chain whose value is
    the seal).
- Key numbers: workspace 1096 passed / 0 failed across 51 suites
  (workspace.log); merged tests 112 passed (merged.log); focused segment test
  1 passed (segment.log); base-last tests 16 passed (last-base.log);
  multi-stage tests 28 passed (multi-stage.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998bc870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the segment closure and the member continuation run on one
  transcript; the single BASE and the staged IOPP tails both hand the advanced
  chain back; unfinished, rejected or failed paths return no chain; the
  message order, field semantics, checks, parameters and final seal position
  of the original second chain are unchanged.
- Boundaries: the second chain keeps its own-group copy of the same relation
  rows. The verifier still receives ChainInputs, but the segment rows use only
  its public coefficients and the committed units. A staged plan combined with
  a request segment is not added. R4 (source-cell equality and coverage
  against the real committed objects) and R5 (real diagnostic plan) remain
  open.
- Not pushed.
