# t001 plan check - 13fac801 / 999f0e6

- Scheme: `t001-plancheck`
- Code base: `13fac801`
- Previous baseline: `999f0e67` (trace `docs/benchmarks/20261108-2348-v2baseline-dd7d198-f485c4d`)
- Date: 2026-11-09
- Purpose: bind shared trace verification to static model data-flow plans and an
  operator policy fixed outside the proof bundle, then measure the verification
  cost on the unchanged case set.

## Change

- Static plans for the three shared models mirror the executor construction
  order without reading a proof or executing the model. The token models cover
  both the per-row (`m <= 16`) and batched (`m > 16`) regimes; the
  time-series model has a single branch.
- The plan check binds the step sequence, opcode and config, every input source
  (previous output or slice, asset or asset slice, trusted constant), the
  external asset identity and values, and the output closure (per-row argmax
  derived from verified logits, final output tensor).
- Verification rebuilds the trusted asset table identities, leaf records,
  registry root and anchor identity from the model asset specifications,
  recomputes the input commitment from the statement tokens, and compares the
  policy, statement, public anchor and CLI model identity.
- The policy is strict TOML fixed outside the bundle. It also fixes the
  expected output: mode, digest, digest algorithm, type, encoding and element
  count. Verification strictly decodes public-inputs, confirms it against the
  authenticated computed output, re-encodes the values canonically as i32
  little endian and compares SHA-256 with the registered GT digest. The
  no-expectation mode returns the authenticated output without a user
  expectation.
- Plan checking no longer materializes flat external assets that the verified
  openings already provide, and caches remaining materialized values by
  authenticated asset identity, layout and model version.
- `run` and `matrix` copy the policy into the run directory and record its
  path, SHA-256 and registration time in `policy-source.json`; they no longer
  fill the model identity from the pending commitment.
- No statement field, proof field or CLI flag was added. `--policy` carries
  the policy.
- `zkie anchor` is the operator-side registration command: it selects the case
  entry by the declared input path, validates the input shape, reads every
  asset through the trusted model specifications, rebuilds the complete
  commitment with the commit algorithm, and prints the anchor identity,
  input-bound model identity, parameters digest, specification version and
  audit counters. It never reads a proof bundle.

## Gate

- `cargo test --workspace`: **126 passed, 0 failed** (includes 5 output-binding, 6 policy, 2 anchor, 7 plan-check and 1 cache-key test).
- Graph tamper tests in `crates/zkie-ops/tests/plan_check.rs` (6 tests) inject
  each mutation with a serialize round-trip, then require the full relation and
  plan verification path to reject: missing, extra, reordered and empty traces;
  cross-step edge with equal values; wrong port, wrong step and missing source;
  cross-step slice; wrong asset and tampered asset row; replaced per-row argmax
  output; replaced final output; changed config.
- Anchor and asset mutations in `crates/zkie-ops/tests/anchor_check.rs` cover
  anchor identity, model commitment, parameters digest, asset id, row count,
  start, root, leaf root, table identity, leaf id, path, missing/extra/duplicate
  assets, and spec path/encoding/count/row width.
- `zkie/tests/verify_policy.rs` (7 tests) covers missing, malformed, unknown
  model, ambiguous case, input replacement, model version mismatch and anchor
  mismatch, all with non-zero exit.
- `zkie/tests/run_policy.rs` (3 tests) covers missing policy in `run`,
  missing `case.policy` in `matrix`, and the policy digest/registration
  record.
- Policy parser unit tests (5 tests) cover parsing, selection, unknown fields,
  duplicate keys, bad shapes, versions and duplicate case ids.
- Model plan step counts: gpt2 4462/2798, gemma3 2090/1506, timesfm 6683.

## Registration

The operator-side registration command is `zkie anchor`, added as a
subcommand of the existing `zkie` entry. It takes `--case`, `--input`,
`--asset-root` and optional `--out`, selects the case entry whose declared
input path matches `--input`, validates the input shape, requires the asset
root to be the one implied by the case root (`<case root>/models` for gpt2 and
`<case root>` for the other models), reads every asset through the trusted
model specifications with the commit algorithm, and cross-checks the resulting
model identity, parameters digest and asset count against the model plan
functions. It never reads a proof bundle.

Each registration prints the `anchor_id`, input-bound `model_id`,
`parameters_digest`, model/specification version, registered-at time,
`assets_checked`, `input_elements` and the case shape. The five registrations
reproduce the baseline public identities exactly:

| case | anchor_id | model_id | assets checked |
| --- | --- | --- | ---: |
| gpt2-seq16 | `dd08df8bcbba41adea6182709e00adb2f897fde4abc241d1ba09c552fd77ae36` | `c2971c0179463bf12aecaefc3488417228008c1314e6905c49a55078242c1268` | 199 |
| gpt2-seq512 | `79091b77db16312ae9d9bc4fdb5d2de93fd87caba4be1158e5de303aa660477d` | `6d35b05739f33e2d2c969f55fafd79c22a96d2264683a4d9a63ecb842438d271` | 199 |
| gemma3-seq16 | `567b30e6712444a9060b5a13d0f011faf50406e15babb6b51a5bd45844d645d3` | `39b607c9f6156a8df155ca1a97e1fa70f0adf74730cc670ec67b5c8fd6410e34` | 245 |
| gemma3-seq512 | `fef99a167d1d31b80c4f8baf1ee746be18d279f7113c4e77e015bff1fa623216` | `d5114631bccfb6e029cfeb1ad2c7b2a65d8a23827551f69424b07f87cbe332ec` | 245 |
| timesfm-16patch | `359e03b223eed95bd9078049954c172909e0ce5d6c7f7a2c96e67c539c1d35b2` | `c87a951844e832fcfc8022688ab86f6c7f634c86874a6b6758c1c88104460742` | 241 |

The generated policies are committed at `models/gpt2/case.policy`,
`models/gemma3/case.policy` and `models/timesfm/case.policy` for review. They
are what `matrix` reads next to the case file and what the production runs
below use.

**Trust caveat.** The registration was executed locally against the local
asset tree. Until the registration run itself is independently attested (the
operator and the verifier are distinct parties, and the policy is transferred
through a channel outside the prover's bundle), these policies are a
non-trusted baseline, not a security acceptance. This note is the required
label for that status.

## Expected-output binding

The policy fixes the expected output in addition to the anchor. For gpt2 and
gemma3 the expected output is the per-position argmax file; for timesfm it is
the complete integer output file. `expected_output_sha256` is taken from the
case `integer_gt` registration value by `zkie anchor`, which verifies the GT
file length and SHA-256 first; the GT files are never modified.

The policy declares the mode and the canonical form explicitly:
`output_mode` (`expected` or `none`), `expected_output_sha256`,
`output_digest_algorithm` (`sha256`), `output_type` (`argmax` or `int_output`),
`output_encoding` (`i32-le`) and `output_elements`. Partial declarations are
rejected.

Verification strictly decodes the public-inputs artifact (u64 fixed-width
length prefix followed by fixed-width elements): truncation, trailing bytes,
count mismatch, wrong shape and illegal values are rejected. The decoded
values must equal the authenticated computed output. The verifier then
re-encodes them canonically as little-endian i32, computes SHA-256 and compares
it with the policy digest. Argmax values are range-checked against the model
vocabulary; the time-series output is checked against the model output shape
and i32 range. The computed output comes from the verified relation trace and
the trusted inputs in both modes.

`output_mode = "none"` skips the digest comparison but still performs the full
relation verification and the statement/public-inputs consistency checks. It
returns the authenticated output digest and values (the canonical bytes are
written as `authenticated-output-i32le.bin` in the run directory); statement
values are never treated as a user expectation. The mode is fixed by the
trusted policy, so a proof cannot downgrade it.

### Binding evidence

All cases below ran through the production `zkie verify` path on the
gpt2-seq16 production bundle. The registered expected-mode policy produced an
authenticated output file byte-identical to the GT file
(`135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280`,
16 elements).

| mode / injection | exit | outcome | error code | authenticated digest |
| --- | ---: | --- | --- | --- |
| expected, correct | 0 | PASS | - | `135c4d1c...39280` |
| none | 0 | PASS | - | `135c4d1c...39280` |
| wrong policy digest | 1 | FAIL | - | `135c4d1c...39280` (computed) |
| tampered value | 2 | FAIL | `statement_mismatch` | - |
| truncated file | 2 | FAIL | `invalid_encoding` | - |
| trailing byte | 2 | FAIL | `invalid_encoding` | - |
| count mismatch | 2 | FAIL | `shape_mismatch` | - |
| illegal argmax value | 2 | FAIL | `statement_mismatch` | - |

The unit tests additionally cover a wrong element width, a non-canonical
encoding, illegal computed argmax values, missing vocabulary context and
negative time-series values. The public-inputs artifact is bincode while the GT
registration is raw i32 little endian; the accepted expected-mode case is the
positive "different encoding" check, because the canonical re-encoding
reproduces the GT bytes exactly.

## Measurement basis

- The five proof bundles produced by the baseline code are reused read-only from
  `docs/benchmarks/20261108-2348-v2baseline-dd7d198-f485c4d/matrix/<case>/cold-0`.
  The proof format and prove path are unchanged; a current-code gpt2-seq16 prove
  reproduced the baseline proof bytes and `proof.bin` SHA-256 (see below).
- Each case has one cold validation run under `cold/<case>/` and three warm
  runs under `warm/<case>/run-<i>/`. The table reports warm medians; cold
  numbers are not in the ledger.
- `stage_wall_ms` and `vmhwm_kib` come from each run's `result.json`.
- The supplementary cold and warm runs on archived bundles used policies
  generated from each bundle's public commitment and statement for measurement
  only. They are self-certified test policies, not a trust anchor: they
  exercise the policy path but cannot demonstrate resistance to a malicious
  bundle. The production-path runs below use the registered
  `models/<model>/case.policy` files instead; the fail-closed tests above
  cover policy and bundle mismatch.

## Verification results

These are the first-implementation numbers, kept as the iteration record.
The final optimized numbers are in the performance rework section.

Warm medians of three runs; baseline values are the warm medians recorded in
the previous trace. gemma3-seq512 has one completed warm run; the two
remaining repeats were stopped after the registered production-path runs
were requested, to free the machine. The interrupted repeat's progress is
archived under `warm/gemma3-seq512/run-2-partial/`; the completed run is in
`warm/gemma3-seq512/run-1/`.

| case | proof bytes | baseline verify wall ms | t001 verify wall ms | delta | baseline verify VmHWM KiB | t001 verify VmHWM KiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 4,161,786,553 | 24,360 | 43,512 | +78.6% | 5,385,024 | 5,917,464 | +9.9% |
| gpt2-seq512 | 38,992,550,944 | 230,274 | 238,536 | +3.6% | 47,016,984 | 47,018,956 | +0.0% |
| gemma3-seq16 | 8,766,306,050 | 44,401 | 79,873 | +79.9% | 13,357,036 | 15,275,928 | +14.4% |
| gemma3-seq512 | 101,812,159,615 | 835,721 | 878,432 | +5.1% | 120,670,388 | 120,678,156 | +0.0% |
| timesfm-16patch | 6,865,533,616 | 42,356 | 85,110 | +100.9% | 7,329,656 | 7,387,768 | +0.8% |

## Current-code prove check

The current code re-proved gpt2-seq16 from the baseline commit artifacts:

- `proof_bytes`: 4,161,786,553 (equal to the baseline value).
- `proof.bin` SHA-256: `2ff98e299f5da7eb7a9bd1f26755742dc887713eb33e208362ced894c4c638fc`
  (equal to the baseline file).
- prove wall: 22,500 ms; VmHWM: 8,316,060 KiB (single run, not a ledger
  number; the baseline warm median was 21,576 ms / 8,315,824 KiB).

## Production path

These production prove numbers remain valid. The verify numbers here are
from the first implementation and are superseded by the performance rework
section.

Five cases ran through `zkie run --commit-policy reuse` against the baseline
cold commit directories, using the registered `case.policy` next to each case
file. Each run produced a fresh proof and verified it; all five returned PASS.
Proof bytes are identical to the baseline for every case, and prove wall and
memory stay within noise, as expected because the change is on the verification
path only.

### Prove

| case | prove wall ms | baseline prove wall ms | delta | proof bytes | prove VmHWM KiB | baseline prove VmHWM KiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 21,689 | 21,576 | +0.5% | 4,161,786,553 | 8,317,576 | 8,315,824 | +0.0% |
| gpt2-seq512 | 165,818 | 165,973 | -0.1% | 38,992,550,944 | 78,570,176 | 78,569,540 | +0.0% |
| gemma3-seq16 | 44,858 | 45,076 | -0.5% | 8,766,306,050 | 17,524,108 | 17,523,028 | +0.0% |
| gemma3-seq512 | 422,783 | 417,040 | +1.4% | 101,812,159,615 | 207,845,808 | 207,848,492 | -0.0% |
| timesfm-16patch | 35,356 | 35,308 | +0.1% | 6,865,533,616 | 13,823,696 | 13,821,988 | +0.0% |

### Verify

Baseline verify values are the warm medians recorded in the previous trace.
Each production verify reads the proof just written by the same run.

| case | verify wall ms | baseline verify wall ms | delta | verify VmHWM KiB | baseline verify VmHWM KiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 43,186 | 24,360 | +77.3% | 5,897,988 | 5,385,024 | +9.5% |
| gpt2-seq512 | 240,469 | 230,274 | +4.4% | 47,017,292 | 47,016,984 | +0.0% |
| gemma3-seq16 | 79,080 | 44,401 | +78.1% | 15,290,060 | 13,357,036 | +14.5% |
| gemma3-seq512 | 868,985 | 835,721 | +4.0% | 120,670,972 | 120,670,388 | +0.0% |
| timesfm-16patch | 86,050 | 42,356 | +103.2% | 7,388,652 | 7,329,656 | +0.8% |

The plan check adds about 19-44 seconds of absolute verification time on the
small proofs and 4-5% on the largest proof, where deserialization and relation
verification dominate. Peak memory rises only on the small cases, where the
materialized asset slices are a larger share of the peak.

## Performance rework

Instrumentation of the first verified implementation on gpt2-seq16 showed the
plan check alone at 21.8 s of a 43 s verify: 3,462 asset materializations
copied 6.06 billion values (about 48 GB), even though the same assets were
already supplied as fully verified external openings.

The rework removes that duplicate work:

- An asset source supplied through an external pool with a flat layout no
  longer materializes the expected values. The planned asset id, start, root,
  row count and row coverage are checked against the verified opening. Because
  the opening covers the full asset and is Merkle-verified against the trusted
  anchor, value equality is implied. Edge and coverage obligations are
  unchanged.
- The remaining materialized asset values are cached per verification call.
  The cache key binds the model version, anchor id, model commitment, asset id,
  path, start, root, leaf root, element count, row count and layout. Only
  sources that the trusted plan uses more than once are cached. The cache is
  created after all openings have been verified and is never shared across
  models, requests or layouts.
- After the first change, the gpt2-seq16 plan check fell from 21.8 s to 1.25 s
  and materializations to 1,119 calls / 0.71 billion values. The cache showed
  no measurable additional gain on the five traces because the remaining
  materializations are distinct; this is recorded as a no-gain optimization.

### Final five-case verification

All five cases verified the production bundles with the registered policies
and the final binary. The ledger table reports warm medians of three runs;
baseline values are the warm medians recorded in the previous trace. The
single first run of each case is listed above it as context only.

| case | single run wall ms | delta vs baseline | single run VmHWM KiB |
| --- | ---: | ---: | ---: |
| gpt2-seq16 | 22,485 | -7.7% | 5,415,992 |
| gpt2-seq512 | 263,507 | +14.4% | 47,020,108 |
| gemma3-seq16 | 53,163 | +19.7% | 13,347,228 |
| gemma3-seq512 | 821,571 | -1.7% | 120,673,912 |
| timesfm-16patch | 52,780 | +24.6% | 7,321,944 |

| case | verify wall ms | baseline wall ms | delta | verify VmHWM KiB | baseline VmHWM KiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 22,334 | 24,360 | -8.3% | 5,397,344 | 5,385,024 | +0.2% |
| gpt2-seq512 | 231,803 | 230,274 | +0.7% | 47,037,560 | 47,016,984 | +0.0% |
| gemma3-seq16 | 43,912 | 44,401 | -1.1% | 13,346,360 | 13,357,036 | -0.1% |
| gemma3-seq512 | 821,644 | 835,721 | -1.7% | 120,672,512 | 120,670,388 | +0.0% |
| timesfm-16patch | 52,126 | 42,356 | +23.1% | 7,321,056 | 7,329,656 | -0.1% |

The duplicate materialization removal brings verification back to the baseline
range on every case; the expected-output binding adds only a strict decode,
comparison and SHA-256 over the output bytes. The authenticated output digest
for each case equals the registered case GT digest.

## Open items and production limitations

- Verification cost is back to baseline on gpt2 and gemma3 (within 2%) after
  the materialization removal. timesfm remains 23.1% above baseline (+9.8 s on
  42.4 s); this residual is reported as a performance compromise rather than
  optimized further.
- The expected output is bound through the policy digest registered from the
  case GT file. The registration trust caveat above still applies: until the
  registration is independently attested, the policies are a non-trusted
  baseline. In `output_mode = "none"` the verifier returns the authenticated
  output digest and does not claim a user expectation.
- `policy-source.json` is an audit record, not an authenticated one: it
  records the source file name, SHA-256 and file modification time.
- Unused model assets require no row openings; their metadata, table identity
  and anchor binding are still checked.
- The confidential local rules directory and collaboration material are outside
  the repository and are not part of this trace.

## Artifacts

- `cold/<case>/`: one validation verify per case (`result.json`,
  `progress.jsonl`).
- `warm/<case>/run-<i>/`: warm verify repeats. gemma3-seq512 has `run-1` and
  `run-2-partial` (the repeat was stopped when the production-path runs were
  requested); all other cases have `run-1` through `run-3`.
- `registration/anchor/`: anchor JSON output and stdout for the five cases.
- `registration/policies/<model>/case.policy`: archived copies of the
  registered policies.
- `prod/<case>/`: prove and verify results, the copied policy and its
  `policy-source.json` record. `proof/proof.bin` stays on disk in place and is
  gitignored.
- `prove-gpt2-seq16/`: current-code proof reproduction; `proof.bin` stays on
  disk and is gitignored, and `prove-gpt2-seq16-run/` holds its text results.
- `binding/`: expected, none, wrong-digest, tampered, truncated, trailing,
  count-mismatch and illegal-value runs; `inputs/` and `policies/`.
- `final/<case>/`: first run with the final code.
- `final-warm/<case>/run-<i>/`: warm repeats used for the ledger.
- `registration/`: anchors and policies regenerated with the output fields.
- `logs/verify-warm.txt`, `logs/verify-prod.txt`, `logs/verify-final.txt` and
  `logs/verify-final-warm.txt`.
- The trace directory is 154G on disk; `/data` has 281G free after the runs.
- No benchmark artifact was deleted.
