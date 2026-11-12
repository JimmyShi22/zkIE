# rework9 final: sampling real-path evidence and the v7 whole-chain conditions

## Scheme and baseline

- Scheme: `rework9-final`.
- Code: `<HEAD>`; previous round head `854c408a`.
- The three seq16 runs were taken on this tree with the env-gated sampling
  counter enabled. The production proof relations, commitments, challenge
  order, parameters and encoding are unchanged; the change is the
  test-only forced-candidate hook, the env-gated counter and the document.
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  ZKIE_SAMPLE_COUNTS=1 cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.

## Purpose

Record the H1-H6 batch of the ninth rework round: real-path rejection
evidence, measured accept/reject counts, the whole-chain conditional view and
BASE elimination, the full hash simulator, the per-entry round/bind
correspondence, the local compiled-plan soundness table (with the prior
eight-round ledger term withdrawn), the base-order correction and the fixture/
deployment wording.

## Validation basis

`seq16x3` means three no-profile reruns of the same GPT-2 formal fixture
(`gpt2-seq16`), not three cases; only `gpt2-seq16` is covered. The five-case
set is the full acceptance basis and is not run here because no seq512 logic
or all-case structural path changed. The evidence is the three run logs, the
workspace log, the engine/ops logs and the sampling tests in this directory.

## Runs

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 | accepts | rejects |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 358,567 | 50,444 | 644.91 | 169,375,968 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 2 | 357,631 | 50,662 | 643.93 | 169,342,396 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 3 | 358,622 | 50,475 | 645.33 | 169,501,140 | f8f512c0…3215f5 | 276,343,251 | 0 |

The proof length and hash are recorded per run. The counter proves that the
rejection branch was not hit in this fixture (`rejects=0` measured); the
real-path forced-candidate tests exercise the rejection branch itself. No
best-run selection is used.

## Workspace

`cargo test --workspace`: 48 test binaries, 751 passed, 0 failed,
17 ignored. The complete log is `workspace.txt`; `engine-tests.txt` is the
zkie-engine library log and `ops-tests.txt` the zkie-ops library log.

## Compatibility

The encoding and hidden schema versions are unchanged; the message structure
is unchanged and there is no deployed compatibility surface. The historical
proof from the previous round remains verifiable under the same plan.

## Security and asset documents

- `docs/lin-bind-closure-security.md` v7: real-path rejection evidence,
  measured counts, whole-chain conditional view, BASE elimination, full hash
  simulator, per-entry round/bind table, local compiled-plan soundness table,
  base-order correction and fixture/deployment assumptions.
- `docs/lin-bind-asset-lifecycle.md`: unchanged, fixture-scoped.

## Open items

The local proximity distance/query numeric bound, a mechanized composition,
the symbolic determinant proof and the exact final-verifier tree-reuse
negative remain `[UNPROVEN]`. The prior eight-round 2^-128 ledger term is
withdrawn rather than imported. Deployment randomness and the single-fixture
scope are stated as assumptions.

## Artifacts

- `run1.txt`, `run2.txt`, `run3.txt`: complete no-profile logs with counters.
- `budget-and-runs.txt`: key lines.
- `workspace.txt`, `engine-tests.txt`, `ops-tests.txt`: complete test logs.

`proof.bin` is not committed; the three proof dumps remain on the test machine.
