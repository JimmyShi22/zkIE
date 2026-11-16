# Consumer opening deletion, failed verification round

- Scheme: `asset-consume` (first projection weight consumer-authenticated).
- Code: `77517ed9`; previous commit `d00a26e6`; branch `refactor/v2`.
- Purpose: first deletion round. Commit and prove completed; the production
  verify entry refused the accepted request with `statement_mismatch`, so no
  evidence was recorded.
- Result: the trace no longer carries the consumer opening and its size is
  1,879,423,254 B, but verification failed at trace step 366
  (`L0_fc_w`, a flat projection weight).
- Cause: the port-form refactor moved the flat-asset coverage check out of the
  external-pool branch. A flat weight materialized from its own verified
  opening (no proof-supplied pool) was therefore refused before its relation
  ran. This is a verification regression, not a consumer-chain failure.
- Fix: the check is guarded by the external pool again and the no-pool flat
  form is covered in the plan checks (`2d4eae29`). The definitive evidence run
  is the next round's directory.
- Artifacts: `run/commit`, `run/proof` and `run.log` remain in this directory;
  the large binary artifacts stay gitignored. The failed proof was reused for
  the instrumented local diagnosis of the failing step.
