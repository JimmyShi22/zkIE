# oracle ports - b4af4698 / eaae60b0

Trace of the relation port and tree-shape hardening plus a full three-stage
rerun on the final code.

## Change

- The shared-trace verifier now checks every step against the trusted plan
  (opcode, config, input and output port counts, external and source maps)
  before the relation verifier consumes any proof port. The external value map
  is built with a checked index, so an oversized map cannot index past the
  trusted input count.
- Relation verification rejects a proof whose output tensor count differs from
  the opcode's exact count, whose input tensor list is empty, whose raw group
  list is empty for an operator that reads raw rows, or whose quotient group
  length differs from the input row count. Raw and auxiliary accesses are
  guarded before indexing.
- A row-wise group must have exactly the padded count produced by the commit
  rule: `max(1, ceil(len/width))` rounded up to a power of two, with checked
  overflow. Extra all-zero leaves with a valid Merkle root and paths are
  rejected.
- Head concatenation requires every part to have the same length before any
  slice is taken.
- New tests cover empty and oversized external maps, empty input, output and
  raw port lists, an extra-zero-leaf group with an independently built valid
  root and paths, and mismatched head-concat part lengths. The shared-trace
  entry is exercised for the external-map and missing-output cases.

## Gate

`cargo test --workspace`: 137 passed, 0 failed. Log: `logs/cargo-test.txt`.

## Full three-stage verification

All five cases ran commit, prove and verify once on the final binary with the
registered anchors and policies. The table reports the single run per stage;
the baseline values are the warm medians from the baseline trace for prove and
verify, and the cold commit run for commit.

| case | commit ms | prove ms | baseline prove ms | verify ms | baseline verify ms | proof bytes | baseline proof bytes | prove VmHWM KiB | verify VmHWM KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 3,389 | 21,710 | 21,576 | 25,202 | 24,360 | 4,161,786,553 | 4,161,786,553 | 8,317,232 | 5,405,436 |
| gpt2-seq512 | 3,383 | 166,840 | 165,973 | 232,603 | 230,274 | 38,992,550,944 | 38,992,550,944 | 78,570,220 | 47,018,280 |
| gemma3-seq16 | 9,382 | 45,222 | 45,076 | 45,736 | 44,401 | 8,766,306,050 | 8,766,306,050 | 17,523,512 | 13,350,856 |
| gemma3-seq512 | 9,371 | 419,974 | 417,040 | 831,845 | 835,721 | 101,812,159,615 | 101,812,159,615 | 207,849,064 | 120,691,540 |
| timesfm-16patch | 5,416 | 35,380 | 35,308 | 43,423 | 42,356 | 6,865,533,616 | 6,865,533,616 | 13,824,016 | 7,322,308 |

All outcomes PASS; verify output binding is `expected` for every case; proof
bytes are identical to the baseline for every case. Total three-stage wall
time was 1907 s across the five cases; the per-case wall times are in
`full3.log`.

## Artifacts

- `prod/<case>/commit-run/`, `prod/<case>/prove/`, `prod/<case>/verify/`:
  `result.json` and `progress.jsonl` for each stage.
- `prod/<case>/commit/`: commit artifacts, kept for later profile reuse.
- `full3.log`: harness log with the per-case wall time and stage records.
- `logs/cargo-test.txt`: full workspace test log.
- `parameters.toml`, `policy.toml` and `policy-source.json` per case.
- `20261109-0907-oracle-ports-b4af4698-eaae60b0/`: an aborted launch retained
  as a record. It failed before producing any proof because the case selector
  was absent; its `result.json` files are the failure evidence.
- Proof binaries are deleted after this ledger entry; commit directories are
  kept. The inventory and freed bytes are recorded in the cleanup entry below.

### Cleanup

After this ledger entry was committed, the five `proof.bin` files were
deleted: 160,598,040,964 bytes (149.57 GiB). The commit directories are
retained. The trace directory now holds 1.8 MiB and 103 files; no `*.bin` file
of at least 1 MiB remains. `/data` used space moved from 1,058,984,255,488 to
898,386,153,472 bytes as reported by df.

## Open items

- The gpt2-seq16 verify delta in the previous trace is pending a boss decision.
- timesfm verify is at +0.06% in the previous trace; no optimization card is
  open.
