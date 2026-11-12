# rework8 final: unbiased mask sampling and the v6 whole-chain document

## Scheme and baseline

- Scheme: `rework8-final`.
- Code: `142daaa5`; previous round head `82251fdc`.
- The three seq16 runs were taken on this tree. The production proof code
  changed only in the mask sampling source; the protocol relations,
  commitments, challenge order, parameters and encoding are unchanged.
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.

## Purpose

Record the G1-G5 batch of the eighth rework round: unbiased rejection sampling
for every mask word, the audit of all modulo reductions, the sampling
boundary/replay/domain tests, the historical-artifact verification, the
production-parameter closure joint object, the BASE star elimination analysis,
the per-entry determinant derivation, the hash-hybrid simulation and the v6
whole-chain document. The sampling change alters the proof content; the
protocol structure and semantics are unchanged.

## Validation basis

`seq16x3` means three no-profile reruns of the same GPT-2 formal fixture
(`gpt2-seq16`), not three cases; only `gpt2-seq16` is covered. The five-case
set is the full acceptance basis and is not run here because no seq512 logic
or all-case structural path changed. The evidence is the three run logs, the
workspace log and the sampling tests in this directory.

## Runs

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 |
|---|---:|---:|---:|---:|---|
| 1 | 357,017 | 50,484 | 643.71 | 169,373,740 | f8f512c0…3215f5 |
| 2 | 356,741 | 50,476 | 644.21 | 169,390,348 | f8f512c0…3215f5 |
| 3 | 355,955 | 50,798 | 642.29 | 169,425,044 | f8f512c0…3215f5 |

The proof length and hash are recorded per run. The sampling change only
alters a draw when a candidate word is at or above the modulus; in this
deterministic fixture no such word occurred, so the proof content is
byte-identical to the previous round. The rejection path is exercised by the
boundary/retry tests, not by this fixture sample. No best-run selection is
used.

## Workspace

`cargo test --workspace`: 48 test binaries, 747 passed, 0 failed,
17 ignored. The complete log is `workspace.txt`; `engine-tests.txt` is the
zkie-engine library log and `ops-tests.txt` the zkie-ops library log.

## Compatibility

The encoding and hidden schema versions are unchanged. A proof produced
before the sampling change (9,939,144 bytes) was decoded and verified by the
new verifier through the bytes entry: `loaded_proof_ok=1`. The old artifact
therefore verifies under the same plan and public statement; the verifier does
not re-derive the prover's mask values.

## Security and asset documents

- `docs/lin-bind-closure-security.md` v6: unbiased sampling proof, evidence
  grades, production-parameter closure joint object, BASE star elimination,
  per-entry determinant table, hash-hybrid simulation, itemized tamper bound
  and assumption support.
- `docs/lin-bind-asset-lifecycle.md`: unchanged, fixture-scoped.

## Open items

The ideal-word sampling is exact; the real finite-seed statement is
PRG/RO-conditional and the concrete hash's advantage is not quantified. A
mechanized composition, a symbolic determinant proof and an exact
final-verifier tree-reuse negative remain `[UNPROVEN]`.

## Artifacts

- `run1.txt`, `run2.txt`, `run3.txt`: complete no-profile logs.
- `budget-and-runs.txt`: key lines.
- `workspace.txt`, `engine-tests.txt`, `ops-tests.txt`: complete test logs.
- `historical-proof.txt`: the old-proof verification log.

`proof.bin` is not committed; the three proof dumps remain on the test machine.
