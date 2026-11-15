# diagnostic-entry-wiring

- Scheme: diagnostic-entry-wiring (the ignored real-entry diagnostic runs the
  request segment over the real activation; this commit also sizes the digit
  message from its real entry count).
- Based on: the request-layout freeze and the digit lookup segment work;
  previous baseline commit 015653c0 (the commit before this one).
- Purpose: enable the real request chain (range objects, both digit lookups,
  segment mask and source messages) on the diagnostic plan and record the
  current full-entry failure. The commit body separates the digit dense
  layout from the request wiring.
- Change:
  - Digit dense layout: a digit source address is the dense value index read
    through the message's own row count; the row count derives from the width,
    the code and the sample pad; the packed arrays and the reduced table keep
    their own widths.
  - Request wiring: the diagnostic plan carries the request chain shape, the
    segment mask plan, the source data plan, the request W plan and both digit
    lookups on the real activation; the pending claim is derived from the
    segment before the run and bound to the member row.
- Numbers: workspace 1110 passed / 0 failed over 51 suites (workspace.log);
  hard gate 45,080 B, sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256, hard-gate.size). The full diagnostic entry
  FAILED: `hidden prove: InvalidProof` at the original entry
  (diag-full-failure.log). Stage probes on the same tree localise it to the
  second chain: first and shared pass, prove_second's own proximity BASE
  linear check fails (diag-probe-stage.log, diag-probe-substep.log). The
  request-row residual is nonzero: the merge claim is a placeholder with
  constant factors and zero maps, so its pending target equals only the mask
  term while the real table request W evaluates nonzero
  (diag-probe-request-row.log).
- Conclusions: the segment path itself runs, but the full consumer entry is
  not closed because the claim source is synthetic. The local segment
  evidence enters in the next commit and is labelled local evidence, not R5.
- Boundaries and memo items (not claimed closed): real claim/maps derivation
  for the isolated L0 K projection (expert memo); kept owner mapping (expert
  memo); the weight unit class and the weight bit bound (expert memo); the
  shared-forest lookup shape (two single-call lookups only); tau restricted to
  powers of two; openings_per_oracle=2 remains the transitional W read; the
  hidden-input verifier still receives the caller mask closure on the
  prover/fixture side.
- Not pushed.
