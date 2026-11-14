# digit-lookup-messages

- Scheme: digit-lookup-messages (the digit lookups in the segment plan and
  message).
- Based on: the digit lookup commit plans (code 128a1e51, trace f5595923).
- Purpose: the plan and the session message must name and carry the range
  object's two digit lookups before the segment can prove and verify them.
- Change:
  - `HiddenSendPlan.digit_lookups`: empty, or exactly the two planes in order
    (low then high). Each lookup plan is checked against the range geometry;
    a plan with lookups but no range objects is refused, and the lookup
    identities and trees stay disjoint from every other object of the session
    (per-position W rows, dot map, index mask, chain mask, request W, source
    message, member witness, range messages).
  - `HiddenSendMessages.digit_lookups`: one entry per lookup with the
    multiplicity-bit root, the mask root, the proof and the claims. The wire
    form is plan-gated: the encoder refuses a zero root and writes the proof
    and the unchecked claims; the decoder reads them under the plan's lookup
    and validates the counts.
  - Every plan and message literal in the session, the output-head path, the
    projection fixtures and the LIN session carries the empty default, so the
    existing bytes are unchanged.
- Key numbers: workspace 1108 passed / 0 failed (workspace.log); digit lookup 3
  passed (digit-lookup.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: both calls are plan objects with a plan-gated wire form and the
  identity checks are in place.
- Boundaries: the prove/verify wiring, the sample answers of the lookup
  messages and the proximity consumption of the claims are not in this step;
  no smallness or range closure is claimed. Not pushed.
