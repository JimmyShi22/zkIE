# rework3 final: bounded reads, evidence tests and security accounting

## Scheme and baseline

- Scheme: `rework3-final`.
- Code: `1f80e682`; previous record commit `d705d28b`; closed relation basis
  `3b3e8ecc`, bounded-read rollback `8d32d3de`.
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.

## Purpose

Close the rework batch: the bounded-read rollback, the implementation-shape,
ownership, tree-association and parallel-equivalence tests, the corrected
memory and validation records, the per-oracle tail-budget report, and the
closure security accounting. No proof relation, commitment, challenge order,
protocol parameter or hiding boundary changes.

## Validation basis

`seq16×3` means three no-profile reruns of the same GPT-2 formal fixture
(`gpt2-seq16`), not three cases. Only `gpt2-seq16` is covered. The five-case set
was not run because no seq512 logic and no all-case structural path changed;
the five-case set belongs to the full acceptance basis. The judgment basis is
that the batch changes implementation, tests and documentation only; the final
evidence is the three run logs and the workspace log in this directory. All
GiB values are KiB / 1048576.

## Runs

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | max RSS GiB | proof sha256 |
|---|---:|---:|---:|---:|---:|---|
| 1 | 356,977 | 50,341 | 643.99 | 169,352,476 | 161.5 | f8f512c0…3215f5 |
| 2 | 355,254 | 49,889 | 641.17 | 169,436,568 | 161.6 | f8f512c0…3215f5 |
| 3 | 355,496 | 50,435 | 641.36 | 169,346,408 | 161.5 | f8f512c0…3215f5 |

The proof is 9,939,144 B in every run; sha256
`f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5`.
The three runs are the full raw set; no best-run selection is used.

## Ledger budget

- code-bearing oracles: 72;
- minimum tail remaining: 252;
- total tail rank: 558,080;
- total unique positions on code-bearing oracles: 879;
- exhausted tail oracles: 0.

The full ledger records 143 events, 887 logical openings and 887 unique
positions; the 8-position difference from the code-oracle total is on the
star messages, which have no code tail. Table reuse is zero.

## Bounded-read check

The B1 instrumented run records `table_bytes=805,306,368` on the formal total
scope, i.e. the three table passes are present after the rollback; the cached
state recorded 268,435,456 (one pass). The B1 trace is
`docs/benchmarks/20261112-0141-row-cache-rollback-8d32d3de-3b3e8ecc/`.

## Workspace

`cargo test --workspace`: 48 test binaries, 725 passed, 0 failed, 17 ignored.

## Conclusions and open items

The batch keeps the proof byte-identical and restores the bounded asset read.
The security document `docs/lin-bind-closure-security.md` records the
joint-view argument, the opening-union budget, the challenge order and the
composition accounting; full joint-view hiding, an opening-union rank proof, a
mechanized composition theorem, output confidentiality (if required), the
trusted X source and the production CLI wiring remain open or `[GAP]`.

## Artifacts

- `run1.txt`, `run2.txt`, `run3.txt`: complete no-profile logs.
- `budget-and-runs.txt`: key lines for the three runs.
- `workspace.txt`: complete workspace test log.

`proof.bin` is not committed; the three proof dumps remain on the test machine.
