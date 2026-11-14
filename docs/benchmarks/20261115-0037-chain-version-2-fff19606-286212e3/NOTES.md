# chain-version-2

- Scheme: chain-version-2 (B2 request chain, TURN 540-C1 V3: transcript/version
  upgrade with old-version rejection).
- Purpose: move the request chain section to the new layout version and refuse
  the previous one at the decoder.
- Change: REQUEST_CHAIN_VERSION 1 -> 2; the version test asserts the current
  version decodes and a header of one (the previous version) is refused before
  any proof byte is parsed; other values remain refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: old sections cannot be parsed under the new layout; the shape
  checks are unchanged.
- Open items: first-chain segment order, member_rows_fit, the negatives, W
  layout freeze, batch report.
