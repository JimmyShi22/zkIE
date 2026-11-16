# Five-case acceptance on the closed consumer chain

- Scheme: `asset-consume` five-case acceptance on the protocol version that
  closes R1-R3 (Eval closes the PSC pending, committed-raw claim row, target
  absorbed before the first challenge).
- Code: `35c76e4e` + `460dd79a` (evidence head `9bf0f30f`); previous head
  `e776aabf`; branch `refactor/v2`.
- Purpose: rebuild the five-case acceptance on the current protocol version.
  The five-case figures recorded at `e776aabf` belong to the previous protocol
  version and are not reused here.
- Basis: the three seq16 cases run one cold plus three warm rounds each (warm
  medians into the ledger, cold excluded); the two seq512 cases run one
  end-to-end round each. In total this batch is 14 end-to-end rounds
  (3 x 4 + 2 x 1) and 42 stage executions (commit, prove and verify per
  round); the five-case group exits 0. Every round provisions its own commit
  (the durable
  column budget is the key batch plus one request batch, so one table instance
  serves one request), the verifier policy is provisioned from that round's
  commit public bundle, `ZKIE_CASE_ID` is propagated to every stage, and the
  large `proof.bin` is deleted after its measured bytes are recorded while the
  textual artifacts stay. The stages run as separate CLI processes, so stage
  wall time and VmHWM are per-stage values.
- Reporting positions: the "proof" figures are whole proof artifacts (trace
  plus consumer sidecar, statement, public inputs and manifest), not
  `proof.bin`; the key bundle is accounted separately in the commit bundle;
  the 3% figure is the regression tolerance, not a proven noise interval; no
  overall performance claim is made from a local comparison.

## Results

All five cases PASS across 14 end-to-end rounds (3 x 4 + 2 x 1) and 42 stage
executions; the five-case group exit is 0. The seq16 cases ran one cold and
three warm rounds; the table gives the cold value and the warm median (ms),
with the warm-stage VmHWM medians (KiB). The seq512 cases ran one end-to-end
round each.

| case | proof artifact cold / warm median (B) | commit cold | commit warm med | prove cold | prove warm med | prove HWM warm med | verify cold | verify warm med | verify HWM warm med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-seq16 | 1,881,722,126 / 1,881,742,222 | 245,070 | 242,468 | 92,728 | 92,652 | 5,826,668 | 78,869 | 78,126 | 7,439,656 |
| gemma3-seq16 | 5,725,337,524 / 5,725,337,524 | 9,296 | 9,309 | 33,618 | 33,511 | 12,167,196 | 29,474 | 29,504 | 10,411,304 |
| timesfm-16patch | 3,647,375,061 / 3,647,375,061 | 5,392 | 5,376 | 23,031 | 23,082 | 7,860,824 | 23,044 | 23,176 | 4,041,100 |

seq512 end to end (one round each):

| case | proof artifact bytes | commit ms / HWM KiB | prove ms / HWM KiB | verify ms / HWM KiB |
|---|---:|---:|---:|---:|
| gpt2-seq512 | 6,794,103,744 | 243,839 / 27,295,484 | 165,761 / 27,660,200 | 102,688 / 14,476,448 |
| gemma3-seq512 | 10,316,824,499 | 9,305 / 1,086,100 | 87,421 / 40,278,252 | 68,609 / 22,043,232 |

The gpt2 cases exercise the closed consumer chain (Eval closes the PSC pending,
committed-raw claim row, target absorbed before the first challenge) at 16 and
512 positions. The proof artifact figures include the consumer sidecar, the
statement, public inputs and the manifest; the trace file alone for gpt2-seq16
is 1,879,423,254 B and the key bundle is accounted separately in the commit
bundle. The seq16 numbers are single rounds on this machine (one cold plus
three warm), not a stability claim; the seq512 cases are single end-to-end
rounds with no warm median.

## Hard gate

The norm-free toy output-head proof was dumped from
`merged_request_accepts_and_round_trips` on this protocol version:

- bytes: 45,080
- sha256: `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (byte-identical to the established gate)

`hard-gate.log` records the command and values; `hard-gate.bin` stays on disk
in this directory and is gitignored like the other benchmark binaries.

## Tests

Workspace: 55 suites, 1,292 passed, 0 failed, 31 ignored, EXIT 0 on the code
tree (`workspace-tests.log`). The acceptance driver is an ignored test. The
release evidence for the closed chain is the separate request-evidence round
(`20261117-0219-...`), not repeated here.

## Open items

- No push and no flag in this round.
- The numbers are single rounds on this machine; the seq16 warm medians are the
  ledger values, the seq512 cases are end-to-end single rounds.
