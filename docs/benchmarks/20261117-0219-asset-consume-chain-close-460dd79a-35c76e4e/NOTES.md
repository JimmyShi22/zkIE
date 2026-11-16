# Consumer relation chain closed at the PSC pending and the raw claim

- Scheme: `asset-consume` rework of the whole-chain review (R1-R3).
- Code: `35c76e4e`; previous head `e776aabf`; branch `refactor/v2`.
- Purpose: close the three relation/challenge wiring gaps the read-only review
  found, keeping the approved boundaries: the product sumcheck's pending row is
  the same claim the W Eval row checks; the raw claim target is bound to the
  committed raw messages under the request coefficients; and the fixed product
  sumcheck fixes its target in the transcript before the first challenge.
- Boundary: no production total acceptance claim in the sense that the security
  replacement is complete. The CLI combined acceptance path exists and passes;
  the gaps below are what this round closes, and the five-case figures of the
  previous head belong to the previous protocol version.

## Change

- R1, Eval closes the pending: the first-stage Eval row is built from the
  checked pending row itself: the member point and weights, the member target,
  and the member mask oracle/slot/coefficient. The independent eval slot, its
  sampled value, the draft's eval-unit list and the separately chosen Eval
  target are removed. The wire target must equal the plan member target.
- R2, raw claim row: the first-stage group now carries one row over the
  committed raw messages of every head with the request coefficients
  `a[p] * theta^h * b_h[j]`, target equal to the same raw claim target the
  product sumcheck fixes. The verifier builds this row from the chain claim and
  the plan, not from the proof. The group row count grows by one.
- R3, target before the first challenge: both fixed product-sumcheck sides
  absorb the caller-fixed target through `PositionPscChallenges::send_claim`
  before the sumcheck starts, so the target is fixed in the canonical prefix
  before the first challenge that multiplies it. The member digest stays a
  post-PSC absorption; no PSC end point or member is pre-inserted.

## Negatives

- Component (release group runs, real group verifier):
  - a verifier Eval row whose target is not the pending the committed W
    satisfies is refused;
  - a raw claim row carrying another weighted target is refused;
  - the existing changed answer/seal negatives stay refused.
- Production (release request evidence): the plan member target changed with an
  internally consistent manifest is refused, beside the existing twelve
  negatives (request file, plan member, plan target, mask root, raw root,
  container byte, residual opening, residual opening row, deferred output,
  missing/changed key bundle).

## Real run

Definitive round (`20261117-0219-asset-consume-chain-close-460dd79a-35c76e4e`):
gpt2-seq16 through the production provider, release, one process for all
stages.

| item | value |
| --- | --- |
| commit | 253,667 ms (key proof 191,484 ms + key self-verify 58,372 ms inside) |
| prove | 90,591 ms |
| verify (accepted run) | 77,588 ms (key re-verification 59,123 ms inside) |
| proof.bin (trace, no consumer opening) | 1,879,423,254 B (195 assets) |
| removed consumer opening | 8,413,688 B |
| trace with that opening | 1,887,836,942 B |
| consumer_proof.bin | 2,317,640 B |
| key bundle | 4,549,512 B |
| VmHWM after prove / verify | 35,405,284 KiB |
| negatives refused | 12 |

The first evidence round (`20261117-0201-asset-consume-chain-close-35c76e4e-
e776aabf`) ran the same chain with the previous eleven-negative harness
(accepted; commit 254,505 ms, prove 90,958 ms, verify 77,860 ms, sidecar
2,326,312 B) and is kept as its own directory; the twelfth negative was added
by `460dd79a`. The trace file is unchanged by the rework (the consumer sidecar
and the key bundle carry the chain changes).

## Reporting positions (corrected, no separate topic)

- The 8,413,688 B opening deletion is the deletion increment, not the
  end-to-end net gain of the whole asset-consume work. The 78,661 to 78,314 ms
  comparison has key and consumer chains on both sides; it only measures the
  final opening deletion.
- The five-case "proof" figures are whole proof artifacts including sidecars
  and statements, not `proof.bin`. For gpt2 seq16 the trace file alone is
  1,879,423,254 B and the warm artifact median is 1,881,737,265 B; the key
  bundle is accounted separately in the commit bundle.
- The 3% figure is the current regression tolerance, not a statistical noise
  interval proven by a single comparison. No claim is made that this round's
  overall performance held; key, consumer and old-trace parts stay separately
  accounted with their total cost, and the local opening-deletion comparison
  does not replace a topic-level before/after.
- The CLI has a real combined acceptance path and PASS; the security
  replacement is not declared complete because R1-R3 were open, not because
  the CLI was unwired.

## Tests

- `cargo test --workspace`: 55 suites, 1,292 passed, 0 failed, 31 ignored,
  EXIT 0 on this tree (`workspace-tests.log`).
- `cargo test -p zkie-ops`: 7 suites, 738 passed, 0 failed, 6 ignored.
- Release group negatives: `cargo test --release -p zkie-ops --lib
  consumer_first -- --ignored` — 6 passed, including the pending/W mismatch and
  the different weighted raw target.
- Release evidence: `cargo test --release -p zkie --test
  consumer_request_evidence -- --ignored` — 1 passed in 1,167.87 s (`run.log`),
  the accepted run plus twelve negatives.

## Open items

- The five-case acceptance figures must be rebuilt on this protocol version
  before any performance or acceptance statement reuses them.
- No push and no flag in this round.
