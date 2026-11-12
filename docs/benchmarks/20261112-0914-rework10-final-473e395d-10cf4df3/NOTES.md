# rework10 final: BASE target source, adaptive simulator and unknown terms

## Scheme and baseline

- Scheme: `rework10-final`.
- Code: `<HEAD>`; previous round head `10cf4df3`.
- The three seq16 runs were taken on this tree with the env-gated sampling
  counter. The production proof code is unchanged from the previous round;
  the changes are the I5 test-bound alignment and the v8 document.
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  ZKIE_SAMPLE_COUNTS=1 cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.

## Purpose

Record the I1-I6 batch of the tenth rework round: the BASE target source and
joint mask/star derivation, the lazy-sampled adaptive RO simulator, the
explicit `eps_proximity = UNKNOWN` with corrected sub-term merges, the kernel
binding derivation, the counter-wording fixes and the wording corrections.

## Validation basis

`seq16x3` means three no-profile reruns of the same GPT-2 formal fixture
(`gpt2-seq16`), not three cases; only `gpt2-seq16` is covered. The five-case
set is the full acceptance basis and is not run here because no seq512 logic
or all-case structural path changed. The evidence is the three run logs, the
workspace log, the engine/ops logs and the tests in this directory.

## Runs

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 | accepts | rejects |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 359,802 | 50,538 | 646.80 | 169,394,600 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 2 | 358,777 | 50,224 | 644.51 | 169,315,672 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 3 | 358,806 | 49,829 | 644.82 | 169,404,380 | f8f512c0…3215f5 | 276,343,251 | 0 |

The proof length and hash are recorded per run. The counter is process-level
and includes `redraw` and repeated evaluations; the runs carry the atomic
overhead and are not exactly comparable to the no-counter baseline. No
best-run selection is used.

## Workspace

`cargo test --workspace`: 48 test binaries, 751 passed, 0 failed,
17 ignored. The complete log is `workspace.txt`; `engine-tests.txt` is the
zkie-engine library log and `ops-tests.txt` the zkie-ops library log.

## Security and asset documents

- `docs/lin-bind-closure-security.md` v8: BASE target source and joint
  mask/star derivation, lazy-sampled adaptive RO simulator, `eps_proximity =
  UNKNOWN` with corrected merges, kernel binding derivation, counter wording.
- `docs/lin-bind-asset-lifecycle.md`: unchanged, fixture-scoped.

## Compatibility

The encoding and hidden schema versions are unchanged; the production proof
code is unchanged. A proof produced before the sampling change remains
verifiable under the same plan and public statement. Old proof verification
does not prove the old prover sampled unbiasedly; that is the honest-prover
assumption. No total security bit count is claimed: `eps_proximity` is
unknown.

## Open items

The local proximity/base-position numeric bound (`eps_proximity = UNKNOWN`),
a mechanized composition, the symbolic determinant proof, the exact
final-verifier tree-reuse negative and a concrete-hash CR bound remain
`[UNPROVEN]`. Deployment randomness and the single-fixture scope are stated
assumptions.

## Artifacts

- `run1.txt`, `run2.txt`, `run3.txt`: complete no-profile logs with counters.
- `budget-and-runs.txt`: key lines.
- `workspace.txt`, `engine-tests.txt`, `ops-tests.txt`: complete test logs.

`proof.bin` is not committed; the three proof dumps remain on the test machine.
