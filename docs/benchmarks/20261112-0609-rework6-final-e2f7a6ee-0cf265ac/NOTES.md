# rework6 final: joint-view hiding evidence and corrected accounting

## Scheme and baseline

- Scheme: `rework6-final`.
- Code: `e2f7a6ee`; previous round head `0cf265ac`.
- The three seq16 runs were taken on the tree whose code/test commit is
  `dccabd23` (the security document commits followed). The only post-run edit
  is a test-only correction in `crates/zkie-ops/src/proof/joint_view_tests.rs`
  (closure claim-mask columns and the table message length); the `zkie`
  hidden-session test links `zkie-ops` as a normal dependency and does not
  compile that module, so the three runs exercise the same production code.
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.

## Purpose

Record the E1-E5 batch of the sixth rework round: joint-view rank evidence,
bad-challenge accounting for the PSC/FOLD coefficient maps, oracle/hash
conditions, corrected composition error accounting and the engineering
wording. Production proof code is unchanged from `1f80e682`; the test path
gained the joint-view rank module, so the round rule requires the seq16x3
rerun and a fresh workspace log.

## Validation basis

`seq16x3` means three no-profile reruns of the same GPT-2 formal fixture
(`gpt2-seq16`), not three cases; only `gpt2-seq16` is covered. The five-case
set is the full acceptance basis and is not run here because no seq512 logic
or all-case structural path changed. The judgment basis is the test-path
change in the rework6 batch; the evidence is the three run logs and the
workspace log in this directory.

## Runs

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 |
|---|---:|---:|---:|---:|---|
| 1 | 355,883 | 50,210 | 642.08 | 169,363,776 | f8f512c0…3215f5 |
| 2 | 357,024 | 50,079 | 642.78 | 169,385,776 | f8f512c0…3215f5 |
| 3 | 354,659 | 50,906 | 641.55 | 169,387,412 | f8f512c0…3215f5 |

The proof is 9,939,144 B in every run; sha256
`f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5`.
The runs are the full raw set; no best-run selection is used. The rework5
range (355.1-356.4 s, 161.5-162.0 GiB) remains the performance record; these
runs are within the recorded spread and are not a new baseline.

## Workspace

`cargo test --workspace`: 48 test binaries, 734 passed, 0 failed,
17 ignored. The complete log is `workspace.txt`.

## Security and asset documents

- `docs/lin-bind-closure-security.md` v4: shared-variable attribution table,
  joint matrix rank table, Vandermonde opening argument, PSC/FOLD/BASE
  coefficient maps with nonzero-determinant degree bounds, oracle encoding and
  hash-order table, corrected constants and the full composition expression.
  The closure PSC is counted as one instance per position (16) and the closed
  path has two FOLD instances.
- `docs/lin-bind-asset-lifecycle.md`: simultaneous asset-related allocation
  below 64 MiB, no whole-matrix copy; the inventory is scoped to the stated
  GPT-2 fixture path.

## Open items

A full simulation, the adaptive composition of the hash and algebraic
arguments, a mechanized symbolic proof of the PSC/FOLD determinant
nonvanishing and an end-to-end 128-bit theorem remain `[UNPROVEN]`. Chain-based
commitment verification, the trusted X source and policy distribution are
long-term direction records, not debts of this round. The approved output
boundary keeps winner/valid/tie public; the full `Y` row is hidden by the
conditional linear-cover argument.

## Artifacts

- `run1.txt`, `run2.txt`, `run3.txt`: complete no-profile logs.
- `budget-and-runs.txt`: key lines.
- `workspace.txt`: complete workspace log.
- `joint-tests.txt`: complete `zkie-ops` library test log with the corrected
  joint-view module.

`proof.bin` is not committed; the three proof dumps remain on the test machine.
