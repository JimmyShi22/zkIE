# Consumer chain rework, first evidence round (eleven negatives)

- Scheme: `asset-consume` whole-chain rework (R1-R3).
- Code: `35c76e4e`; previous head `e776aabf`; branch `refactor/v2`.
- Purpose: run the production request chain after the R1-R3 wiring. The
  harness still carried the previous eleven negatives; the twelfth (plan member
  target) was added afterwards in `460dd79a`, so this round is kept as the
  eleven-negative round and the definitive evidence is the next directory.
- Result: accepted. gpt2-seq16, release, one process: commit 254,505 ms (key
  proof 191,925 + key self-verify 58,708 inside), prove 90,958 ms, verify
  77,860 ms (key re-verification 58,640 inside); trace 1,879,423,254 B (195
  assets, no consumer opening); removed opening 8,413,688 B; trace with the
  opening 1,887,836,942 B; consumer sidecar 2,326,312 B; key bundle 4,540,968
  B; VmHWM 35,405,828 KiB. Eleven negatives refused.
- Artifacts: `run.log`, `consumer-evidence.json/.log`, `run/` (the large
  proof binaries remain on disk and are gitignored).
