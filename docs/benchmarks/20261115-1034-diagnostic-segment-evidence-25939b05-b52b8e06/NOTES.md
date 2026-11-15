# diagnostic-segment-evidence

- Scheme: diagnostic-segment-evidence (a first-chain segment evidence entry
  runs the request segment over the real activation and is measured
  separately; this is local evidence, not R5).
- Based on: diagnostic-entry-wiring (b52b8e06); previous baseline commit
  b52b8e06.
- Purpose: provide the local segment evidence the ruling allows: the real
  activation, the range objects, both digit lookups, the tie and X4 rows, the
  segment mask and source messages, and the segment proximity test, each with
  the formal first-part verifier on the same chain state.
- Entry: `prove_first_part_segment_evidence` /
  `verify_first_part_segment_evidence` (partial path: the first chain up to
  its seal; the shared and second chains are outside it). The diagnostic test
  runs it under `ZKIE_SEGMENT_EVIDENCE=1`.
- Numbers, real gpt2-seq16 L0 K activation:
  - tensor limbs: 3
  - pending discovery (separate, not proof time): 26.71 s
  - segment message bytes (encoded first-part message, not a full proof):
    2,032,888
  - segment prove: 26.35 s
  - segment verify: 11.13 s
  - process peak memory at the segment run (VmHWM): 8,957,068 kB
  - negatives: 9 attempted / 9 rejected by the formal first-part verifier
    (changed digit lookup bits root; changed digit lookup multiplicity;
    changed packed array root; changed reduced table root; changed digit
    object root; changed segment mask root; changed source cell root;
    changed merge end value; changed segment proximity seal)
  - workspace: 1110 passed / 0 failed over 51 suites (workspace.log)
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
    (hard-gate.log, hard-gate.sha256, hard-gate.size)
- Full entry record (kept, not turned into success): the unchanged diagnostic
  entry still fails with `hidden prove: InvalidProof`; the request row's
  residual is nonzero because the merge claim is a placeholder with constant
  factors and zero maps (full-failure.log). The failure is recorded in the
  previous trace (diag-full-failure.log, diag-probe-*.log).
- Conclusions: the segment mechanisms (range messages, two single-call digit
  lookups, tie/X4, segment masks, segment proximity) prove and verify on the
  real activation with independent verification and rejection of every
  attempted tamper. This is local segment evidence; it is not a closed
  consumer chain and not R5.
- Boundaries and memo items (not claimed closed): real claim/maps derivation
  for the isolated L0 K projection (expert memo); kept owner mapping (expert
  memo); the weight unit class and the weight bit bound (expert memo); the
  shared-forest lookup shape (two single-call lookups only); tau restricted to
  powers of two; openings_per_oracle=2 remains the transitional W read; the
  hidden-input verifier still receives the caller mask closure on the
  prover/fixture side; the segment message bytes are the first-part packet,
  not a full three-stage proof.
- Not pushed.
