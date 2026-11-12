# Norm committed messages and PSC verifier skeleton

## Scheme and baseline

- Scheme: `norm-psc-skeleton` (T-003 batch 2b.1 step 2).
- Code: `8bbb5137`; previous commit `3a7829cd`.
- Purpose: add the five committed norm groups (x, sum, squares, mean, variance), their
  first and second message codecs and a commitment-free product-sumcheck verifier
  skeleton on top of the shared committed-message layer.
- This is an internal slice, not the 2b.1 close. The BAT/FOLD/CS/BASE row binding is
  not implemented; no soundness or privacy claim is made.

## Results

- `cargo test --workspace`: 48 suites, 820 passed, 0 failed, 17 ignored.
- Focused `hidden_norm_psc` tests: 5 passed, 0 failed.
- No existing proof path changed and no proof was generated in this round. The
  output-head proof bytes were already shown unchanged in the preceding extraction
  trace; this slice only adds a new module and its tests.

## What the skeleton checks

- Plan shape, group ids, oracle bases, opening counts, roots and group order.
- First-message group declarations, samples, positions and runtime openings.
- Second-message version, instance count, PSC round counts and shown end values.
- Chain start bound to the plan digest and the encoded first message; one challenge
  domain per position/relation instance.
- Each shown point value equals the transcript's shown end value.

## Explicitly deferred

- The five group roots are trusted plan inputs; the proof does not derive the
  sum/squares/mean/variance roots from the x rows.
- BAT/FOLD/CS/BASE row binding is absent. The transcript checker is commitment-free:
  arbitrary transcript or challenge tampering is not rejected.
- The one-time PSC masks are prover-private and not committed or verified.
- No owner/consumer binding, no ledger integration, no output commitment.
- Consequently the skeleton is not a hidden norm relation and cannot support a
  soundness or privacy conclusion.

## Open items

- Expert/TL design confirmation for x binding, group row declarations, PSC endpoint
  binding, private opening encoding, challenge order and ledger placement.
- After confirmation: implement the row binding and then the negative matrix over
  transcript/challenge tampering.

## Artifacts

- `workspace.txt`, `focused.txt`, `change.stat`, `NOTES.md`.
- The four-column requirement/landing/gap mapping is in the task channel report,
  outside this trace.
