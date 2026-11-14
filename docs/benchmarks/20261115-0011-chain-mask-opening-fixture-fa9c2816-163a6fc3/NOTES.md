# chain-mask-opening-fixture

- Scheme: chain-mask-opening-fixture (B2 request chain, TURN 540-C1 V1: the
  sub-proof mask table opening positive path).
- Purpose: exercise the committed mask-only tree and the root check end to end
  in the fixture.
- Change: the test commits a mask-only runtime tree (four mask rows of the
  short columns, extension-field symbols), opens two columns, verifies the
  opening against the root, refuses another root and a broken plan.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: the opening root path is exercised positively and negatively;
  mask rows use the short column width the tree expects.
- Open items: four row bindings, first-chain segment order, transcript/version,
  member_rows_fit, negatives, W layout freeze.
