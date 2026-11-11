# rework7 final: whole-chain random-variable inventory

## Scheme and baseline

- Scheme: `rework7-final`.
- Code: `341590af`; previous round head `ce698ff7`.
- The three seq16 runs were taken on this tree. Production proof code is
  unchanged from `1f80e682`; the test path gained the real-path joint tests.
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.

## Purpose

Record the F1-F5 batch of the seventh rework round: evidence grades, real-path
SEND/Eval/shown/BASE extraction, the closure mask joint object, actual-k FOLD
and actual-q opening checks, the constant-determinant hiding-map derivation,
the RO/CR hash model correction, the tree-reuse bound and the whole-chain
security document version 5. The round rule requires a fresh seq16x3 and
workspace log after the test-path change.

## Validation basis

`seq16x3` means three no-profile reruns of the same GPT-2 formal fixture
(`gpt2-seq16`), not three cases; only `gpt2-seq16` is covered. The five-case
set is the full acceptance basis and is not run here because no seq512 logic
or all-case structural path changed. The evidence is the three run logs and
the workspace log in this directory.

## Runs

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 |
|---|---:|---:|---:|---:|---|
| 1 | 355,378 | 50,891 | 642.21 | 169,421,784 | f8f512c0…3215f5 |
| 2 | 356,036 | 50,918 | 643.15 | 170,029,832 | f8f512c0…3215f5 |
| 3 | 356,269 | 50,360 | 642.69 | 169,422,920 | f8f512c0…3215f5 |

The proof is 9,939,144 B in every run; sha256
`f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5`.
The runs are the full raw set; no best-run selection is used. The previous
round range remains the performance record; these runs are within the
recorded spread and are not a new baseline.

## Workspace

`cargo test --workspace`: 48 test binaries, 742 passed, 0 failed, 17 ignored.
The complete log is `workspace.txt`; `ops-tests.txt` is the zkie-ops library
log (204 passed).

## Security and asset documents

- `docs/lin-bind-closure-security.md` v5: whole-chain random-quantity ->
  public-position -> sharing -> hiding/binding -> exception -> basis format;
  evidence grades; constant-determinant PSC/FOLD maps; closure mask joint
  object; RO/CR hybrid correction; tree-reuse bound; assumption support table.
- `docs/lin-bind-asset-lifecycle.md`: unchanged, fixture-scoped.

## Open items

The modulo-reduction bias of the mask stream is a conditional statistical
premise; a strict statistical hiding claim would need rejection sampling (a
same-topic candidate, not changed here). A mechanized composition, a symbolic
determinant proof and an exact final-verifier tree-reuse negative remain
`[UNPROVEN]`. Chain-based commitment verification, the trusted X source and
policy distribution are long-term direction records, not debts of this round.

## Artifacts

- `run1.txt`, `run2.txt`, `run3.txt`: complete no-profile logs.
- `budget-and-runs.txt`: key lines.
- `workspace.txt`: complete workspace log.
- `ops-tests.txt`: complete zkie-ops library test log.

`proof.bin` is not committed; the three proof dumps remain on the test machine.
