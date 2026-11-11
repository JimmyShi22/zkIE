# rework5 final: linear-cover hiding, corrected constants and seq16x3

## Scheme and baseline

- Scheme: `rework5-final`.
- Code: `1887b320`; previous `610f8559`; production proof basis `1f80e682`.
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.

## Purpose

Record the `seq16×3` rerun after the test-path addition of the changed-message
negative and the second group-table shape, together with the rework5 security
and asset documents. The production proof code is unchanged from `1f80e682`;
the test path changed, so the rerun is required by the round rule.

## Validation basis

`seq16×3` means three no-profile reruns of the same GPT-2 formal fixture
(`gpt2-seq16`), not three cases; only `gpt2-seq16` is covered. The five-case
set is the full acceptance basis and is not run here because no seq512 logic or
all-case structural path changed. The judgment basis is the test-path change in
the rework5 batch; the final evidence is the three run logs and the workspace
log in this directory.

## Runs

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 |
|---|---:|---:|---:|---:|---|
| 1 | 356,337 | 50,523 | 643.21 | 169,906,264 | f8f512c0…3215f5 |
| 2 | 356,404 | 50,458 | 642.22 | 169,376,228 | f8f512c0…3215f5 |
| 3 | 355,107 | 50,187 | 641.28 | 169,409,472 | f8f512c0…3215f5 |

The proof is 9,939,144 B in every run; sha256
`f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5`.
The runs are the full raw set; no best-run selection is used. The earlier
355 s / 161.5 GiB reference remains the performance record; these runs are
355.1-356.4 s and 161.5-162.0 GiB, within the recorded spread and not a new
baseline. No numeric threshold is preset.

## Workspace

`cargo test --workspace`: 48 test binaries, 726 passed, 0 failed, 17 ignored.
The complete log is `workspace.txt`.

## Security and asset documents

- `docs/lin-bind-closure-security.md` v3: linear-cover criterion, Vandermonde
  opening submatrix, PSC/FOLD/BASE/closure coverage, hash-model hybrid,
  corrected field constants and union bound.
- `docs/lin-bind-asset-lifecycle.md`: simultaneous asset-related allocation
  bound below 64 MiB; no whole-matrix copy.

## Open items

Full simulation/adaptive composition, all-challenge PSC/FOLD rank and a
mechanized composition theorem remain `[UNPROVEN]`. Chain-based commitment
verification, the trusted X source and policy distribution are long-term
direction records, not debts of this round. The approved output boundary keeps
winner/valid/tie public; the full `Y` row is hidden by the conditional
linear-cover argument.

## Artifacts

- `run1.txt`, `run2.txt`, `run3.txt`: complete no-profile logs.
- `budget-and-runs.txt`: key lines.
- `workspace.txt`: complete workspace log.

`proof.bin` is not committed; the three proof dumps remain on the test machine.
