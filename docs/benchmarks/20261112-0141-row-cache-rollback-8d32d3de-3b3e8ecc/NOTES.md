# Row-cache rollback: bounded asset reads

## Scheme and baseline

- Scheme: `row-cache-rollback`.
- Code: `8d32d3de`; before state `3b3e8ecc` (the retained row cache).
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.
- Same machine, same fixture and release build for the before and after runs.

## Purpose

Remove the retained table-row cache from the prove path. Every row is read
through the bounded asset reader again, so the asset is never held as a whole
and the three row passes keep their own reads. The shared closure point vector
from the same earlier change remains; no proof relation, commitment, challenge
order, parameter or hiding boundary changes.

## Results

| Quantity | Before (`3b3e8ecc`) | After (`8d32d3de`) |
|---|---:|---:|
| hidden prove | 362,908 ms | 357,506 ms |
| hidden verify | 52,635 ms | 50,450 ms |
| whole fixture test | 701.02 s | 644.48 s |
| process maximum resident | 207,138,524 KiB | 169,337,668 KiB |
| hidden proof bytes | 9,939,144 | 9,939,144 |
| hidden proof sha256 | f8f512c0...3215f5 | f8f512c0...3215f5 |

Proof sha256:
`f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5`.
The proof is byte-identical.

The instrumented after run records `table_bytes=805,306,368` on the formal
`total.prove_output_head_lin_hidden` scope, i.e. the three table passes are
back. The cached state's final profile recorded 268,435,456 bytes (one pass).

## Notes

- The before run is the instrumented run in
  `docs/benchmarks/20261111-2342-verifier-path-profile-759db42b/`; its proof
  dump is `/tmp/p24-before-hidden.bin`. The after no-profile run is
  `/tmp/b1_after.log`; the after profile is `/tmp/b1_profile.log`.
- The before/after difference is within the run-to-run spread of this fixture;
  no speed gain is claimed. The table-read restoration is the intended change
  and is recorded directly.
- Peak RSS is lower in the after run, but this single pair is not a
  controlled memory attribution.
- GiB values, where used, are KiB / 1048576.

## Artifacts

- `b1-before.txt`: before metrics, source lines and proof hash.
- `b1-after.txt`: complete after no-profile log.
- `b1-profile.txt`: complete after instrumented log with the table-byte
  scope.

`proof.bin` is not committed; the proof files remain on the test machine.
