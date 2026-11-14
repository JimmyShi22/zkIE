# digit-lookup-segment

- Scheme: digit-lookup-segment (the two digit lookups proved and verified in
  the request segment).
- Based on: the digit lookup messages (code 99861821, trace 21937417).
- Purpose: the digit lookups' messages were carried but not proved; the
  segment proves both calls over the committed digit message, in the ruled
  order after the digit object and before the batching element.
- Change:
  - `DigitLookupMessages` carries the bits sample, the mask message's tree,
    sample, positions and opening; the plan-gated codec writes the tree and
    the opening with the message's own plan and the decoder checks the tree
    against the plan's and reads the opening under the drawn positions.
  - New chain helpers: the bits and mask roots enter as their own rounds with
    their shapes and trees, each SEND answer enters as its own round, the mask
    opening positions are drawn from the chain, and the mask message's runtime
    tree is rebuilt from the committed rows and salts.
  - Prover: after the digit object's sample and before the batching element,
    each plane takes its own digit cells (the plane's places only), computes
    the multiplicity bits over the real keys, pads the key vector with the
    public zero, commits the two messages with private randomness, absorbs
    the roots and sample answers, draws and opens the mask positions, and
    proves the lookup against `0..1023` (low) or `0..255` (high). The verifier
    replays the absorptions, compares the seeds, answers and positions,
    verifies the mask opening against its root, and requires the same claims.
  - The key count is per plane (`combinations * cell_rows`), and the sources
    enumerate only the plane's places; the earlier both-planes count is fixed.
  - Tests: the digit-lookup layer proves and verifies both planes through the
    chain and refuses a key outside the table; the full chain-segment session
    test enables both lookups and passes with the same chain, section and
    seal assertions.
- Key numbers: workspace 1109 passed / 0 failed (workspace.log); digit lookup
  4 passed (digit-lookup.log); chain segment 1 passed (segment.log); hard gate
  45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: both calls are committed, sampled, proved and verified inside
  the segment with the chain transcript and the ruled order.
- Boundaries: the lookup claims are carried and verified but not yet consumed
  by the segment proximity group; the sources-to-BAT scatter still needs the
  digit message's own layout; no smallness or range closure is claimed. Not
  pushed.
