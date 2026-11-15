# Derived raw values for the remaining shared relations

- Scheme: `derived-raw`, the shared trace form also stops carrying the raw
  declaration of the affine scale, softmax, softmax rows, hadamard, gated
  ffn, profile layer norm, profile rms norm, profile rope and profile
  standardize relations. The matrix relations already omit it since the
  previous round.
- Code: `95a236fb`; follow-up test commit `0a00b5cd`; previous commit
  `79a7ea73`; branch `refactor/v2`.
- Baseline binary: built in a clean worktree at `79a7ea73` with
  `cargo build --release -p zkie` and saved as `/data/derived-baseline-zkie`,
  SHA-256 `92d212d5306d5032bb1d7498bdc639912a42e3636163cbeeb799e7518395902b`.
  Before and after run the same commit instance per case, the same
  parameters, input and policy.
- Hit rationale: the change alters the shared proof structure of all three
  models and both softmax paths (the vector softmax on the seq16 cases and
  softmax rows on the long cases), so all five cases are hit.
- Change:
  - the shared prove path still generates the forward record with its raw
    accumulators, but no longer commits or transmits raw, its length entry or
    its dedicated root and opening for the nine listed opcodes; every other
    group keeps its oracle and ordering, and the matrix omission is
    unchanged;
  - every branch still recomputes its full integer relation from the trusted
    inputs and checks all declarations: affine multiplier/addend, rounding,
    conversion and clamp; hadamard/gated product and shift; softmax maximum,
    table lookup, non-negative entries, denominator and zero-denominator
    rejection, the auxiliary exp entry equals the actual table lookup, and
    the rounded division output; profile norm means/variances, rsqrt table
    index and extension selection, valid and padding semantics, scale/bias
    sources, overflow and output checks; rope two-half rotation with the
    public coefficients; standardize source row, valid range, statistics and
    output checks;
  - the shared form is selected by the trusted shared verification entry, not
    by the proof; a proof that still carries raw or a raw length in this form
    is rejected (`statement_mismatch`), and the ordinary relation entry keeps
    requiring raw, so the empty form is not a second accepted form there
    (`shape_mismatch`);
  - no field is added and the serialization format is unchanged; the shared
    input version constant is untouched.
- Raw consumers: `session_baseline.rs::check_relation_shape`,
  `verify_oracle_authentication` and the transcript word collection keep
  reading `proof.raw` through the request/component entry
  (`prove_op_relation_with_inputs`); the shared trace path never enters that
  entry. Hidden chains, requests, challenges, budgets and the 45,080 B object
  are unchanged.

## Opcode accounting

The per-opcode account is read from the actual before proofs: the archived
test `opcode-accounting.rs` deserializes each baseline trace and sums the raw
lengths of the nine listed opcodes; its output is
`opcode-accounting.log`. Raw values are committed as two 64-bit words each
(16 B per value); the per-relation bookkeeping is the root tuple (72 B), the
opening fixed fields (32 B), the inner raw vector length (8 B) and the raw
length entry (8 B), 120 B per relation.

| case | opcode | steps | raw values | value bytes |
| --- | --- | ---: | ---: | ---: |
| gpt2 seq16 | ProfileLayerNorm | 25 | 409,600 | 6,553,600 |
| | AffineScale | 288 | 294,912 | 4,718,592 |
| | Softmax | 2,304 | 36,864 | 589,824 |
| | total | 2,617 | 741,376 | 11,862,016 |
| gpt2 seq512 | ProfileLayerNorm | 25 | 13,107,200 | 209,715,200 |
| | AffineScale | 288 | 9,437,184 | 150,994,944 |
| | SoftmaxRows | 144 | 37,748,736 | 603,979,776 |
| | total | 457 | 60,293,120 | 964,689,920 |
| gemma3 seq16 | ProfileRmsNorm | 163 | 1,564,672 | 25,034,752 |
| | ProfileRope | 90 | 368,640 | 5,898,240 |
| | AffineScale | 72 | 294,912 | 4,718,592 |
| | Softmax | 1,152 | 18,432 | 294,912 |
| | GatedFfn | 18 | 589,824 | 9,437,184 |
| | total | 1,495 | 2,836,480 | 45,383,680 |
| gemma3 seq512 | ProfileRmsNorm | 163 | 50,069,504 | 801,112,064 |
| | ProfileRope | 90 | 11,796,480 | 188,743,680 |
| | AffineScale | 72 | 9,437,184 | 150,994,944 |
| | SoftmaxRows | 72 | 18,874,368 | 301,989,888 |
| | GatedFfn | 18 | 18,874,368 | 301,989,888 |
| | total | 415 | 109,051,904 | 1,744,830,464 |
| timesfm 16-patch | ProfileStandardize | 1 | 1,024 | 16,384 |
| | ProfileRmsNorm | 20 | 655,360 | 10,485,760 |
| | Hadamard | 320 | 409,600 | 6,553,600 |
| | Softmax | 5,120 | 81,920 | 1,310,720 |
| | ProfileLayerNorm | 20 | 655,360 | 10,485,760 |
| | AffineScale | 1 | 32,768 | 524,288 |
| | total | 5,482 | 1,836,032 | 29,376,512 |

All nine listed opcodes occur in the production batch; the seq16 cases use
the vector softmax and the long cases use softmax rows, so both paths are
covered. The measured proof.bin drop equals value bytes plus 120 B per
removed relation in every case: gpt2 seq16 11,862,016 + 314,040 = 12,176,056;
gpt2 seq512 964,689,920 + 54,840 = 964,744,760; gemma3 seq16 45,383,680 +
179,400 = 45,563,080; gemma3 seq512 1,744,830,464 + 49,800 = 1,744,880,264;
timesfm 29,376,512 + 657,840 = 30,034,352.

## Evidence

Single before/after runs on this machine, release builds, same commit per
case, same parameters/input/policy; gemma3 and timesfm use deterministic
commits created once by the after side and reused by the before side; gpt2
uses its registered commits. Wall and memory numbers are single runs and are
not a stability claim.

| case | stage | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| gpt2 seq16 | proof.bin | 1,900,012,998 | 1,887,836,942 | -12,176,056 |
| | prove wall / VmHWM | 14,100 ms / 3,923,496 KiB | 13,996 ms / 3,897,184 KiB | -104 ms / -26,312 KiB |
| | verify wall / VmHWM | 11,752 ms / 3,802,184 KiB | 11,830 ms / 3,793,800 KiB | +78 ms / -8,384 KiB |
| | trace assets | 196 | 196 | 0 |
| gpt2 seq512 | proof.bin | 7,763,700,070 | 6,798,955,310 | -964,744,760 |
| | prove wall / VmHWM | 48,215 ms / 20,795,648 KiB | 43,762 ms / 18,922,808 KiB | -4,453 ms / -1,872,840 KiB |
| | verify wall / VmHWM | 31,580 ms / 15,041,612 KiB | 29,491 ms / 14,088,840 KiB | -2,089 ms / -952,772 KiB |
| | trace assets | 196 | 196 | 0 |
| gemma3 seq16 | proof.bin | 5,770,896,630 | 5,725,333,550 | -45,563,080 |
| | prove wall / VmHWM | 33,939 ms / 12,209,720 KiB | 33,519 ms / 12,166,088 KiB | -420 ms / -43,632 KiB |
| | verify wall / VmHWM | 26,083 ms / 10,416,892 KiB | 25,670 ms / 10,368,920 KiB | -413 ms / -47,972 KiB |
| | trace assets | 237 | 237 | 0 |
| gemma3 seq512 | proof.bin | 12,061,694,822 | 10,316,814,558 | -1,744,880,264 |
| | prove wall / VmHWM | 95,323 ms / 43,688,284 KiB | 88,711 ms / 40,273,584 KiB | -6,612 ms / -3,414,700 KiB |
| | verify wall / VmHWM | 65,545 ms / 23,979,876 KiB | 62,059 ms / 22,276,388 KiB | -3,486 ms / -1,703,488 KiB |
| | trace assets | 237 | 237 | 0 |
| timesfm 16-patch | proof.bin | 3,677,141,420 | 3,647,107,068 | -30,034,352 |
| | prove wall / VmHWM | 23,219 ms / 7,920,320 KiB | 23,119 ms / 7,859,052 KiB | -100 ms / -61,268 KiB |
| | verify wall / VmHWM | 21,427 ms / 4,055,396 KiB | 21,139 ms / 4,021,224 KiB | -288 ms / -34,172 KiB |
| | trace assets | 238 | 238 | 0 |

- Commit stage: only the create side runs it, 9,357 ms / 1,085,748 KiB
  (gemma3 seq16), 9,395 ms / 1,085,760 KiB (gemma3 seq512) and 5,424 ms /
  175,224 KiB (timesfm); commit behavior is unchanged by this round.
- Whole-artifact metric, listed separately from the proof.bin sizes: this
  metric counts the whole proof output directory. gpt2 seq16
  1,900,016,968 -> 1,887,840,911; gpt2 seq512 7,763,710,000 ->
  6,798,965,248; gemma3 seq16 5,770,900,606 -> 5,725,337,524; gemma3 seq512
  12,061,704,766 -> 10,316,824,499; timesfm 3,677,409,411 -> 3,647,375,061.
  The remainder after the proof.bin delta is the size change of the other
  files in the proof directory (statement, public inputs, manifest,
  completion marker), not part of the relation account.
- All ten verifications pass with `output_binding = "expected"` and the
  unchanged expected output digests: gpt2 seq16 `135c4d1c…`, gpt2 seq512
  `eac0949d…`, gemma3 seq16 `8683f5e2…`, gemma3 seq512 `6d1df6b5…`, timesfm
  `cc72805b…`.
- Trace asset counts are unchanged; this round removes per-relation raw
  groups, not asset openings.
- The long-case wall and memory reductions are single-run observations and
  are not a pre-committed performance claim: gpt2 seq512 prove -4,453 ms and
  VmHWM -1,872,840 KiB, gemma3 seq512 prove -6,612 ms and VmHWM
  -3,414,700 KiB. The seq16 cases are within or near run-to-run variation.

## Negatives

All relation tests run through the full verification path after a serialize
round trip unless noted. Authentication failures and numerical rejections
are recorded separately; none of the relation negatives is a broken Merkle
path.

- whitelist positives: `shared_derived_relations_carry_no_raw_declaration`
  covers all nine opcodes through the shared prove path, asserts the raw and
  raw lengths are empty, verifies through the shared entry, and shows the
  ordinary entry rejects the empty form (`shape_mismatch`). The five-case
  batch exercises all nine in production traces.
- authentication-consistent output tamper: the matrix test
  `shared_matmul_output_tamper_…` and the profile norm test
  `rms_extension_consumption_tamper_is_rejected` alter the record before the
  existing prover commits it, so authentication accepts the rebuilt material
  and the numerical relation rejects the altered declaration
  (`invalid_proof`).
- key auxiliary and lookup declarations: `shared_softmax_aux_tamper_…`
  alters the auxiliary exp entry and the denominator; the authentication
  helper first accepts every committed group and the full verification
  rejects (`invalid_proof`). `shared_softmax_table_replacement_…` proves with
  one table while the trusted plan materializes another of the same length;
  the exp entry no longer matches the lookup and the relation rejects
  (`invalid_proof`). A softmax raw length error in the ordinary form is still
  rejected by the existing test.
- input and source binding: the matrix plan input replacement and the
  softmax table replacement tests cover a trusted-plan value that differs
  from the proved input; both reject in the relation (`invalid_proof`).
- legacy raw form: `rms_extension_legacy_raw_form_is_rejected` builds the
  proof with the ordinary input-committed form (raw present) and the shared
  path rejects it (`statement_mismatch`); the matrix equivalent
  `shared_entry_rejects_the_ordinary_raw_form` also shows the ordinary entry
  accepts the same proof. Raw length/opening one-sided books are rejected
  (`statement_mismatch`).
- non-whitelisted control: `shared_unlisted_relations_keep_their_raw_declaration`
  uses a RopeProjection test op, which is outside the list; its shared form
  still carries raw and verifies through the shared entry. AffineScale is not
  used as the control because it is now in the list.
- ordinary-entry raw errors still reject: the existing relation tests for
  the input-committed form (raw width, depth, coverage, ordering) still pass.
- other asset and edge substitution negatives are reused:
  `negative_wrong_asset`, `negative_external_and_missing_ports` and
  `negative_external_opening_must_cover_the_full_asset`; their lines are in
  `targeted-tests.log`.

## Boundaries

- No new field, no serialization change, no new protocol, masking, sampling
  or budget change; the input public/private boundary, private weights,
  anchors, policies and ground truth are untouched; the 45,080 B gate object
  is untouched; no other opcode is expanded or reduced.
- Workspace suite: 1175 passed / 0 failed / 20 ignored (51 suites). Targeted
  plan_check 41/0 and relation_openings 21/0 after the final test-only
  commit; the log is `targeted-tests.log`.
- Logs and scripts: the ten `before-*`/`after-*` run logs; `release-build.log`;
  `workspace.log`; `targeted-tests.log`; `opcode-accounting.rs` and
  `opcode-accounting.log`; `run-script.sh`; `run-console.log`;
  `run-summary.txt`; `proof-assets.py` and `proof-assets.txt`. Per-stage
  `result.json` and `progress.jsonl` stay under each run directory; proof
  binaries stay on disk and are gitignored. Not pushed.

## Round closure

- Expert batch review: the whole round is accepted, no further rework.
- Final commit list, in order: `95a236fb` (production change: the remaining
  shared relations no longer carry the redundant raw declaration),
  `0a00b5cd` (test assertions for the softmax auxiliary and table
  declarations), `e13984d9` (evidence record), and this closing commit (round
  flag, NOTES record only; no code change).
- State at closure: hard gate 45,080 B untouched; no anchor, policy, asset,
  ground truth or commitment registration changed; the workspace suite is
  green (1175 passed / 0 failed / 20 ignored); source and evidence are on
  the branch, not pushed.
