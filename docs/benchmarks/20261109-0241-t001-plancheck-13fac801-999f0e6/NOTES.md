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

- `cargo test --workspace`: **133 passed, 0 failed** (includes 5 output-binding, 6 policy, 2 anchor, 7 plan-check, 5 relation-opening, 1 opening-structure, 1 signed-input and 1 cache-key test).
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

## Opening row integrity (P0)

`verify_opening`, the single entry for asset openings, now derives the expected
shape from the trusted asset metadata and enforces, before any consumer reads
the rows:

- full coverage: `opening.row_count == meta.rows` and
  `rows.len() == meta.rows`;
- position order and uniqueness: `rows[i].row == i` for every position;
- row width: every row has exactly `meta.row_width` values;
- padding: rows at or beyond `ceil(element_count / row_width)` are all zero,
  and the tail of the last data row beyond `element_count` is zero;
- Merkle depth: `path.len() == log2(meta.rows)` for the trusted perfect tree;
- the existing link and per-row Merkle path verification.

All index and element-count arithmetic is checked. The plan checker, the flat
external fast path, the relation verifier and the cache consume only openings
that passed this check; the cache and fast path run after opening verification
and never before it.

### Structural negatives

Each case below is serialized and verified through the full relation and plan
path; duplicate and swap keep the original rows and Merkle paths valid, so only
the position check can reject them. Error codes are asserted, so rejection is
not an accidental failure from an unrelated condition.

| injection | error code |
| --- | --- |
| duplicate row in another position | `invalid_index` |
| swap of two valid rows | `invalid_index` |
| same-count missing row | `invalid_index` |
| row width change | `shape_mismatch` |
| non-zero padding value | `statement_mismatch` |
| Merkle path length change | `invalid_proof` |
| partial opening coverage | `statement_mismatch` |
| relation input replacement / wrong asset | `statement_mismatch` |

Duplicate, swap and width cases run through `verify_shared_trace` with a
serialize round-trip. The padding and path-length cases use the same
`verify_opening` entry on a correctly built two-row fixture; the original root
cannot contain a wrong-width leaf with a valid path, so that width case is a
structural test rather than a claimed valid-path case.

## Asset hash registration

`zkie anchor` emits a per-asset file SHA-256 list. Each digest covers the
complete asset file bytes in order and encoding through the same mapping that
supplies the Merkle leaves, and the anchor is built from that same read. The
list is published only after the commitment anchor, model identity, input
digest, shape and, when `--policy` is given, the registered policy fields are
checked. `--asset-hashes <previous-anchor.json>` compares the list item by item
with a previous registration and refuses to publish on any mismatch. All 1,129
list entries across the five case registrations match the actual files
(recursive recomputation in the registration log).

**Trust caveat.** The first list is a self-built change-detection baseline. It
does not prove that the first read weights are the expected model and has not
been checked against externally published model material. Asset source
authorization is deferred to the on-chain commitment phase; it is not addressed
in this round.

## Operator oracle coordinate authentication

Relation proofs commit each logical group as an oracle tree. `verify_group`
previously checked only that every opening shared the group oracle and passed a
per-opening Merkle check; `decode`/`decode_wide` then concatenated the openings
in list order and truncated. A prover could reorder, duplicate or omit valid
rows with valid paths and change the vector consumed by relation verification.

The fix binds every group to its commitment construction before decoding:

- purpose: input, output, raw, quotient, helper and auxiliary groups are
  distinguished by the verifier from the relation structure, not from the
  opening list; raw groups use the single-row wide form, the others use the
  row-wise width form;
- coordinates: `opening.index == list position`, rows cover exactly
  `0..count-1`, and the group oracle root count equals the opening count;
- tree shape: count is a power of two and every path length is
  `log2(count)`;
- width: normal groups use the opcode/config/declared-length width rule
  (`commit_group_width`) and every row has exactly that width; wide groups
  have exactly one row of `2*len` words and an empty path;
- padding: rows at or beyond `ceil(len/width)` are zero, matching the zero
  rows added by `commit_group_width`; wide groups have no padding;
- decode: after these checks, decode walks openings in authenticated
  coordinate order, rejects short coverage and no longer concatenates in list
  order with blind truncation. `proof_input_values` and
  `proof_output_values` return `Result` and use the same checked decode;
- all index and multiplication arithmetic is checked; the empty opening list
  is reserved for external slots, whose values come from the authenticated
  external mapping.

Audit: the only `OracleOpening` consumers are `verify_group`, `decode`,
`decode_wide`, `proof_input_values` and `proof_output_values`. The model
executors call `proof_output_values` while proving. The verifier calls
`proof_input_values` in `registry.rs` before relation verification in the same
call that then fully verifies the relation, and after verification in the plan
and shared-trace layers. No other reachable acceptance path consumes openings
without the checks above.

### Negative coverage

All cases use real rows and original Merkle paths where the mutation permits;
the relation verifier is reached through the production entry
`verify_op_relation_with_external`, with asserted error codes.

| injection | error code |
| --- | --- |
| duplicate row in another position | `invalid_index` |
| swap of two valid rows | `invalid_index` |
| same-count missing row | `invalid_index` |
| row width change | `shape_mismatch` |
| non-zero padding row | `statement_mismatch` |
| path depth change | `invalid_proof` |
| coverage loss | `statement_mismatch` |
| length change | `shape_mismatch` |
| wide-row width change | `shape_mismatch` |
| wide-group index change | `invalid_index` |
| wide-group path change | `invalid_proof` |
| wide-group duplicate | `shape_mismatch` |

Width, padding and tree-shape cases that cannot be reached by mutating a valid
commitment without breaking its root are exercised on independently
constructed fixtures. The bounds test uses a `usize::MAX` length and returns a
shape error without panicking.

On the production gpt2-seq16 proof bundle the real proof artifact was
re-serialized after duplicating and after swapping two openings of an internal
input group (step 0) and an internal output group (step 4), keeping the
original rows and Merkle paths; `zkie verify` rejected every run with
`invalid_index` (exit 2, no panic). Raw output: `logs/oracle-cli-tamper.txt`.

### Signed time-series input

`models/timesfm/src/plan.rs` asserts that `model_commitment` over u32 tokens
reproduces the commitment over the same values encoded as signed i64
little-endian, covering the u32 -> i32 -> i64 conversion used by the model.

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
- `--policy` is mandatory and carries trusted content; there is no lenient
  default path, and old invocations migrate to the new contract.

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

The rework removes that duplicate work and keeps the authentication
obligations:

- An asset source supplied through an external pool with a flat layout no
  longer materializes the expected values. The planned asset id, start, root,
  row count and row coverage are checked against the verified opening. Because
  the opening covers the full asset and is Merkle-verified against the trusted
  anchor, value equality is implied. Partial-row assets keep the materialized
  comparison. Edge and coverage obligations are unchanged.
- Flat previous-step outputs are compared by reference instead of being copied.
  Slice and layout results still materialize.
- Remaining materialized asset values are cached per verification call. The
  cache key binds the model version, anchor id, model commitment, asset id,
  path, start, root, leaf root, element count, row count and layout. Only
  sources that the trusted plan uses more than once are cached, the cache is
  created after all openings have been verified, and it is never shared across
  models, requests or layouts.
- After the materialization change the gpt2-seq16 plan check fell from 21.8 s
  to 1.25 s. The cache and the borrow change showed no measurable additional
  gain on these traces; both are recorded as no-gain optimizations.

### Final five-case verification

All five cases verified the production bundles with the registered policies
and the final binary. The table reports warm medians of three runs; baseline
values are the warm medians recorded in the previous trace.

| case | verify wall ms | baseline wall ms | delta | verify VmHWM KiB | baseline VmHWM KiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 25,520 | 24,360 | +4.8% | 5,387,788 | 5,385,024 | +0.1% |
| gpt2-seq512 | 233,515 | 230,274 | +1.4% | 47,019,360 | 47,016,984 | +0.0% |
| gemma3-seq16 | 45,093 | 44,401 | +1.6% | 13,353,424 | 13,357,036 | -0.0% |
| gemma3-seq512 | 821,263 | 835,721 | -1.7% | 120,693,040 | 120,670,388 | +0.0% |
| timesfm-16patch | 42,380 | 42,356 | +0.1% | 7,323,620 | 7,329,656 | -0.1% |

The largest gpt2/gemma3 delta is 4.8% (gpt2-seq16,
25,520 ms vs 24,360 ms,
+1,160 ms); gpt2-seq512,
gemma3-seq16 and gemma3-seq512 are within 1.6%. timesfm verify median is
42,380 ms vs 42,356 ms: +24 ms
(+0.06%), at parity with the baseline. Profiling after the
materialization removal shows `full_values` at 0.08% of samples; the remaining
overhead is the independent plan-comparison pass and the oracle coordinate
checks over every step, which are authentication obligations rather than
redundant materialization. Reducing it further would require fusing plan
comparison with relation verification, which was not attempted.

The gpt2-seq16 delta exceeds the 3% noise band. The same case measured -7.9%
in the previous group with the rework but before the oracle coordinate checks,
so the group-to-group spread on the smallest case is larger than the band; the
final group is the ledger for the final code. This item is outside the
timesfm acceptance and is recorded as pending a boss decision. The boss
temporarily accepted the earlier group's timesfm residual (+23.1% / +9.8 s)
and deferred timesfm optimization; the current timesfm result is +0.06%
(42,380 ms vs 42,356 ms), the final group does not reproduce a residual, no
optimization was made, and no new card is opened. The earlier +23.51% figure
remains in the record only as the earlier record.

## Open items and production limitations

- Final verify medians: gpt2-seq16 25,520 ms
  (+4.8%), gpt2-seq512 233,515 ms
  (+1.4%), gemma3-seq16 45,093 ms
  (+1.6%), gemma3-seq512
  821,263 ms (-1.7%),
  timesfm-16patch 42,380 ms (+0.06%). gpt2-seq16 exceeds the
  3% band, is recorded as-is and is pending a boss decision; the oracle
  coordinate checks are authentication obligations. The boss accepted the earlier timesfm residual
  (+23.1% / +9.8 s) and deferred timesfm optimization to later rounds; the
  final group shows no measurable timesfm residual and no new card is opened.
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
  `policy-source.json` record. `proof/proof.bin` is gitignored and was deleted
  in the cleanup.
- `prove-gpt2-seq16/`: current-code proof reproduction; `proof.bin` is
  gitignored and was deleted in the cleanup, and `prove-gpt2-seq16-run/` holds
  its text results.
- `binding/`: expected, none, wrong-digest, tampered, truncated, trailing,
  count-mismatch and illegal-value runs; `inputs/` and `policies/`.
- `final/<case>/`: first run with the final code.
- `final-warm/<case>/run-<i>/`: warm repeats of the first rework binary.
- `final2-warm/<case>/run-<i>/`: warm repeats of the previous final binary.
- `final3-warm/<case>/run-<i>/`: warm repeats of the final binary including
  the oracle coordinate checks.
- `registration/`: anchors and policies regenerated with the output fields.
- `logs/verify-warm.txt`, `logs/verify-prod.txt`, `logs/verify-final.txt`,
  `logs/verify-final-warm.txt`, `logs/verify-final2-warm.txt`,
  `logs/verify-final3-warm.txt` and `logs/oracle-cli-tamper.txt`.
- Large proof binaries are deleted after verification and ledger entry; the
  keep list is minimal: textual artifacts, each case's cold-0 commit directory,
  policies, GT files, registrations and assets.

### Cleanup

The final measurements and all textual records were committed first. Then 30
`proof.bin` files totaling 832.10 GiB were deleted from the three trace
directories, together with the temporary smoke copies `/tmp/v2smoke` and
`/tmp/v2smoke512` (40.19 GiB). The files were selected as every regular
`*.bin` file of at least 1 MiB outside the `cold-*` commit directories; the
count and byte total were taken before deletion. `/data/zkie-assets` (769G)
was not touched. After the cleanup the three trace directories hold 152 KiB,
4.2 MiB and 5.1 MiB, and 600 files remain, including the 30 files under the
`cold-*` commit directories. `/data` free space moved from 264.7 GiB
(284,194,062,336 bytes) to 1.07 TiB (1,177,655,009,280 bytes) as reported by
df. Every number in this trace comes from the committed text records; no proof
binary is needed to read them.
