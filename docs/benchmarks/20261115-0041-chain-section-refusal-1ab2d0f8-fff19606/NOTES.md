# chain-section-refusal

- Scheme: chain-section-refusal (B2 request chain, TURN 540-C1 V1/V4: refuse
  the unwired first-chain segment and pin the pending mask owner).
- Purpose: no plan is silently proved without the first-chain sub-proof segment
  it asks for, and the pending mask terms stay on the index-mask object.
- Change: HiddenLinPlan::validate refuses plan.send.request_chain Some with
  UnsupportedPrimitive; the fixture test also asserts a pending mask term on
  another oracle is refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: the session cannot pretend to carry the chain section before it
  runs it; V4's index-mask rule is enforced at the plan.
- Open items: first-chain segment order wiring, the negatives, W layout freeze,
  batch report.
