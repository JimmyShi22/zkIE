# Five-case acceptance after the consumer weight switch

- Scheme: `asset-consume` (acceptance batch after the first projection weight
  moved from a trace opening to the request consumer chain).
- Code: `8b548474`; previous baseline `d00a26e6`; branch `refactor/v2`.
- Purpose: the same production three-stage pipeline accepts the five registered
  cases after the consumer-source switch: the three seq16 cases with one cold
  and three warm rounds (warm medians into the ledger, cold excluded), the two
  seq512 cases end to end, and the unchanged 45,080 B hard gate.

## Basis

- The change concerns the registered consumer table of gpt2. The three seq16
  cases (gpt2-seq16, gemma3-seq16, timesfm-16patch) run one cold and three warm
  rounds each; the two seq512 cases run one end-to-end round each. Every run
  carries commit, prove and verify as separate CLI processes, so the stage wall
  time and VmHWM are per-stage values.
- The durable column budget of one table instance permits the key batch and
  exactly one request batch (52 + 52 of 104 columns), so the prover cannot
  reuse one commit for several requests. Each acceptance round therefore
  provisions its own commit (a fresh table instance). The warm rounds are warm
  on the machine (binaries, weights and public tables are cached); their table
  instance is new by design. The cold round is excluded from the medians.
- The verifier policy is provisioned per round from that round's commit public
  bundle: the model/case/expected-output fields come from the model policy
  template; the anchor identity and (for gpt2) the limited table identity are
  read from the commit. The CLI matrix's static gpt2 policy predates the
  registered table plan, and its commit-reuse mode cannot serve the one-request
  instance, so the rounds drive the same `zkie commit/prove/verify` stages
  directly and aggregate their results. This is the hit basis for not using
  the single-invocation matrix path for gpt2.
- Only the seq16 cases carry warm medians (the TL's seq16 basis); the seq512
  cases are end-to-end single rounds, so no seq512 warm median is claimed.
- Large `proof.bin` files are deleted after their stage results are recorded;
  `summary.json` keeps their byte counts. Every textual artifact stays. The
  hard-gate artifact (45,080 B) stays on disk and is gitignored like the other
  benchmark binaries.
- `proof_bytes` below is the CLI's proof-artifact byte count (the whole proof
  output directory, including the request consumer sidecar and the statement
  files), not the trace file alone; the consumer sidecar varies slightly with
  its fresh per-request seed, so the gpt2-seq16 warm proof bytes differ within
  a few ten thousand bytes.

## Results

All five cases PASS. seq16: one cold plus three warm rounds each; cold and
warm medians (ms) with the warm-stage VmHWM medians (KiB):

| case | proof bytes (cold / warm median) | commit cold | commit warm med | prove cold | prove warm med | prove HWM warm med | verify cold | verify warm med | verify HWM warm med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-seq16 | 1,881,751,888 / 1,881,737,265 | 243,945 | 245,217 | 92,909 | 92,563 | 5,741,716 | 78,465 | 77,990 | 7,439,700 |
| gemma3-seq16 | 5,725,337,524 / 5,725,337,524 | 9,288 | 9,282 | 33,802 | 33,490 | 12,166,660 | 29,380 | 29,317 | 10,386,740 |
| timesfm-16patch | 3,647,375,061 / 3,647,375,061 | 5,397 | 5,384 | 23,020 | 23,064 | 7,860,116 | 23,235 | 23,198 | 4,040,136 |

seq512 end to end (one round each):

| case | proof bytes | commit ms / HWM KiB | prove ms / HWM KiB | verify ms / HWM KiB |
|---|---:|---:|---:|---:|
| gpt2-seq512 | 6,794,076,580 | 243,992 / 27,295,424 | 164,937 / 27,611,300 | 103,090 / 14,421,176 |
| gemma3-seq512 | 10,316,824,499 | 9,316 / 1,086,124 | 87,131 / 40,278,688 | 68,118 / 22,044,040 |

The gpt2-seq512 run exercises the request consumer chain at 512 positions
(prove 164,937 ms, verify 103,090 ms, both accepted); the gpt2-seq16 rounds
exercise the same chain at 16 positions. gpt2 commit includes the one-time key
qualification. The per-round anchors differ for gpt2 (fresh table randomness);
gemma3 and timesfm anchors equal their static policy anchors.

## Hard gate

The norm-free toy output-head proof was dumped from
`merged_request_accepts_and_round_trips` and removed from the source again:

- bytes: 45,080 (unchanged)
- sha256: `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged)

`hard-gate.log` records the command and the values; `hard-gate.bin` stays on
disk in this directory.

## Tests

- `cargo test --workspace`: 55 suites, 1,292 passed, 0 failed, 29 ignored,
  EXIT 0 on this tree (`workspace-tests.log`).
- The acceptance driver `zkie/tests/acceptance_rounds.rs` is an ignored test;
  each invocation runs one case for its rounds through the CLI stages.

## Round log

`driver.log` (first batch: gpt2-seq16 and timesfm-16patch PASS; the gemma3-seq16
invocation failed on an existing round directory and the two seq512 invocations
on the missing child case id) and `driver2.log` (the fixed rerun of
gemma3-seq16, gpt2-seq512 and gemma3-seq512) are kept. The fixed rerun cleared
its round directories before re-running, so the first batch's failed-stage
files were replaced; the logs are the record of both attempts, and the passing
results live in each case's `summary.json`.

## Open items

- The single-invocation CLI matrix aggregation is not used for gpt2 while the
  table instance is one-request and the static policy is stale; the per-round
  acceptance driver replaces it.
- No production total acceptance claim.
