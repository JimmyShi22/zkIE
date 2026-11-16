# Consumer weight authentication without the trace opening

- Scheme: `asset-consume` (the first projection weight is proved by the
  request consumer chain; its trace opening is removed).
- Code: `2d4eae29` (fix commit; the first code commit is `77517ed9`, whose
  failed evidence round is `20261116-2321-asset-consume-delete-77517ed9-
  d00a26e6`); previous commit `d00a26e6` (the qualified request proof);
  branch `refactor/v2`.
- Purpose: make the request consumer chain the actual proof of the `L0_k_w`
  projection weight. The production plan marks that weight input as a
  consumer-authenticated source; the trace no longer materializes its asset
  opening; the twelve projection steps' operator relations are deferred to the
  request chain, which the production verifier runs before acceptance. A trace
  that still carries the old opening, or any leftover of it, is refused.
- Boundary: no production-level total acceptance is claimed. The trace
  verifier defers the twelve relations; the production entry closes them by
  verifying the request consumer chain over the committed table and the
  verified trace outputs before it binds the output.

## Change

- Plan source: a direct projection weight input can be promoted from a traced
  asset to a consumer-authenticated source. Promotion is trusted-plan work: a
  path that occurs only inside a wrapper, or not at all, is refused, so a plan
  cannot keep the old opening form for a registered consumer asset.
- Trace verification: a consumer source is accepted only as the weight input of
  a projection with the column-slice form. Its port must carry no external
  opening and no edge; the structural checks of the step still run. The step's
  operator relation is not verified in the trace pass; the caller must verify
  the request consumer chain, which the production entry does immediately after
  the trace and before any output binding.
- Residual openings: any opening for an asset that the plan marks consumer-
  authenticated is refused, even when the opening itself is valid for the
  commitment. A changed row in such a residue is refused by the opening
  authentication as before.
- Producer: the production executor stops opening the twelve projection
  weights into the trace; the request consumer chain keeps reading its
  committed rows from the asset store through the same committed table. The
  promoted plan is the only production plan of the model.
- Verification regression fixed before this round: the port-form refactor had
  moved the flat-asset coverage check out of the external-pool branch, so an
  ordinary flat weight materialized from its own trace opening was refused.
  `2d4eae29` restores the guard and covers the no-pool flat form.

## Real run

Production pipeline through the provider: commit, prove and verify on
`gpt2-seq16` from `/data/i1-cases/gpt2/case.toml` over `/data/zkie-assets`,
freshly committed for this run. The release evidence test runs all stages in
one process; the memory high-water mark below is cumulative across those
stages, not a per-stage figure.

| item | value |
| --- | --- |
| case | gpt2-seq16 |
| commit | 252,566 ms (key proof 190,885 ms + key self-verify 57,814 ms inside) |
| prove | 91,063 ms |
| verify (accepted run) | 77,675 ms (key re-verification 58,842 ms inside) |
| proof.bin (trace, no consumer opening) | 1,879,423,254 B |
| trace assets | 195 |
| removed consumer opening | 8,413,688 B |
| trace with that opening | 1,887,836,942 B |
| consumer_proof.bin | 2,319,480 B |
| VmHWM after prove | 35,498,376 KiB |
| VmHWM after verify | 35,498,376 KiB |
| model id | `c2971c0179463bf12aecaefc3488417228008c1314e6905c49a55078242c1268` |
| anchor id | `be5732a7bf52001065781a40ba0299d288f7a678308c8d39c5ba41669abeff6f` |
| table root | `bac3383c405bd06a82e391d09c308f02a9327b4b072b073fb7993ea7903772a9` |
| table record digest | `44a26ec7dd10261e172bbea14d403a32df7d34dd483fe2e583ae856ae7cea774` |
| table configuration | `94dd0177b5a08d3e328a51222a4ab5aee7d6a982b91964b072eea7e75ae0e925` |
| table shape | log_n=12, ell=2048, r=128, v=10, answers=1, salt=40, m_once=m_later=284, k_lo=0 |

The request proof file and the key bundle are per-request/per-instance
material (fresh request seed, fresh table randomness), so their byte sizes are
not part of the deletion comparison.

## Deletion comparison

Same machine, same case, same release pipeline, one process per run, same
measurement basis. Before is the qualified request run of the previous commit
(`20261116-2225-asset-consume-qualify-3012979b-57e4a8dc`), after is this run.
The trace carries no table material, so the trace byte basis is comparable
across the two commit instances.

| item | before (with opening) | after (no opening) | delta |
| --- | --- | --- | --- |
| proof.bin | 1,887,836,942 B | 1,879,423,254 B | -8,413,688 B |
| trace assets | 196 | 195 | -1 |
| prove | 91,530 ms | 91,063 ms | -467 ms (-0.51%) |
| verify | 77,180 ms | 77,675 ms | +495 ms (+0.64%, inside the 3% noise band) |
| VmHWM (process, commit+prove+verify) | 35,497,036 KiB | 35,498,376 KiB | +1,340 KiB (+0.0038%) |

The reconstructed trace-with-opening size is taken in the after process by
serializing the removed opening and adding it to the accepted trace bytes:
`1,879,423,254 + 8,413,688 = 1,887,836,942 B`, exactly the before size. Putting
the removed opening back into the accepted trace with an internally consistent
manifest also lets the previous commit's verifier accept that file, so the two
traces are identical apart from the opening.

Controlled same-instance check of the changed path (same commit instance,
statement, public inputs and request proof file; the previous commit's verifier
on the trace with the removed opening vs this commit's verifier on the accepted
trace; one fresh process each):

| item | with opening (old code) | without opening (new code) | delta |
| --- | --- | --- | --- |
| accepted | yes | yes | - |
| verify wall | 78,661 ms | 78,314 ms | -347 ms (-0.44%) |
| VmHWM (verify process) | 7,423,104 KiB | 7,414,872 KiB | -8,232 KiB (-0.11%) |

The end-to-end peak (about 35.5 GiB) is reached in the unchanged commit/key
stage; the verify-only peak is about 7.4 GiB. The end-to-end +1,340 KiB is
therefore the unchanged stage's cross-instance run-to-run variance, not a cost
of the deletion: on the same instance the changed path is strictly smaller in
both wall time and VmHWM. No memory increase is attributable to the change.

## Negatives

The production verify entry was driven eleven ways after the accepted run;
every one was refused:

| injection | result |
| --- | --- |
| request proof file removed (manifest lists it) | refused |
| plan member changed, proof manifest internally consistent | refused |
| plan target changed, proof manifest internally consistent | refused |
| group mask root changed, proof manifest internally consistent | refused |
| one committed raw root changed, proof manifest internally consistent | refused |
| one container byte changed, proof manifest internally consistent | refused |
| the removed consumer opening put back, proof manifest internally consistent | refused |
| the same residual opening with a changed row | refused |
| one deferred projection output changed, proof manifest internally consistent | refused |
| key bundle removed from the commit public bundle | refused |
| key bundle byte changed, commit manifest internally consistent | refused |

The changed-value negatives rewrite the manifest and its `COMPLETE` marker, so
the refusal comes from the protocol checks (plan member/target/roots, the
residual-opening rule, the consumer chain, key qualification), not from the
container digest alone. The deferred-output negative is rejected only by the
request consumer chain: the trace pass defers that step by design.

## Tests

- `cargo test --workspace`: 54 suites, 1,292 passed, 0 failed, 28 ignored,
  EXIT 0 on this tree (`workspace-tests.log`); the hard-gate test passes with
  the unchanged 45,080 B artifact.
- Release evidence: `cargo test --release -p zkie --test
  consumer_request_evidence -- --ignored` — 1 passed in 1,097.51 s (`run.log`),
  the accepted run plus eleven negatives, each re-running the per-request key
  verification and the trace verification.
- Plan checks: 48 passed, including the consumer-source positives and
  negatives and the no-pool flat asset regression test.

## Open items

- seq512 with the consumer chain and the deletion has not been run.
- The other projection weights still travel as trace asset openings; only the
  registered consumer weight is covered by this round.
- No production-level total acceptance claim; the remaining source relations
  and the larger model instances stay open.
