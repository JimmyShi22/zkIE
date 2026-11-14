# chain-segment-entry

- Scheme: chain-segment-entry (B2 request chain, TURN 540-C1 V1: the session
  entry surface for the first-chain segment inputs).
- Purpose: fix the public surface the caller uses to hand the segment inputs,
  with an explicit refusal until the order is wired.
- Change: prove_hidden_lin_hidden_with_chain_segment and
  verify_hidden_lin_hidden_with_chain_segment take ChainInputs and refuse a
  plan with request_chain Some (UnsupportedPrimitive); the entry test builds a
  full input bundle and pins the refusal.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: the caller surface exists; no plan is silently proved without
  its section.
- Open items: run the segment in prove_first/verify_first with the pending/
  member ordering before the LIN first W and the seal, negatives, W layout
  freeze, batch report.
