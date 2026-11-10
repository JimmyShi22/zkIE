# Trusted switch targets and model identity in the verifier

## Scheme

- Scheme: `s7b-rework-r3`.
- Current code: `912fafb7` (the code change is `9244a314`).
- Previous code: `9244a314`.

## Purpose

R3 first step: the independent verifier now rejects a public switch plan whose
target differs from the trusted compiled targets, and a statement whose model
commitment differs from the trusted policy's model id. A wrong-target negative
is added to the real-chain test.

## Results

- Unit tests: 55 passed, 0 failed, 3 ignored.
- Real local chain: commit 29.598 s; prove 9.507 s / 603,796 KiB VmHWM;
  artifact 5,578,904 B; base proof 63,464 B; independent verify accepts and
  returns the same seal.
- Every existing negative still rejects, including the new wrong-target case.

## Boundary

- The switch runtime-tree plans and stage-fold mask plans are still taken from
  the public handoff. Their shape and faults are checked; their derivation from
  the private session master is not independently rebuilt. This is the
  remaining R3 item, together with the table-to-asset opening tie.
- The production CLI does not carry this chain.

## Logs

- `unit-tests.txt`, `real-chain.txt`.
