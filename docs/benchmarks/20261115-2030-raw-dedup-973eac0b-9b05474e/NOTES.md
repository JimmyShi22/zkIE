# Shared matrix relations without the redundant raw declaration

- Scheme: `raw-dedup`, the production shared trace no longer carries the raw
  accumulator of the MatMul, Projection and RegressionHead relations; the
  verifier already recomputes the same value from the trusted inputs.
- Code: `973eac0b`; follow-up test commit `9e5549b6`; previous commit
  `9b05474e`; branch `refactor/v2`.
- Baseline binary: built in a clean worktree at `9b05474e` with
  `cargo build --release -p zkie` and saved as `/data/raw-baseline-zkie`,
  SHA-256 `81f395e4111c58f94753d312b5581426dbfeed54ea971c6278e0db27853f5f1a`.
  Before and after run the same commit instance per case, the same
  parameters, input and policy.
- Hit rationale: the change alters the shared proof structure of all three
  models, so all five cases are hit and the batch covers every case.
- Change:
  - the shared prove path still generates the forward record with its raw
    accumulators, but no longer commits or transmits raw, its length entry or
    its dedicated root and opening for the three matrix opcodes; every other
    group keeps its oracle and ordering;
  - the verifier recomputes the matrix product, the checked bias addition,
    the shift and the rounding from the trusted materialized inputs and
    compares the authenticated output, exactly as before; no matrix
    computation, source binding or output check is removed;
  - the shared form is selected by the trusted shared verification entry
    (`verify_shared_trace_steps` calls the shared relation entry), not by the
    proof; a proof that still carries raw or a raw length in this form is
    rejected (`statement_mismatch`), and the ordinary relation entry keeps
    requiring raw, so the empty form is not a second accepted form there
    (`shape_mismatch`);
  - a non-zero bias must have exactly one entry per output (`m * n`) before
    any slot is indexed; a malformed bias length returns an error instead of
    panicking, and the per-output-slot bias semantics are unchanged;
  - no field is added and the serialization format is unchanged; the shared
    input version constant is untouched.
- Raw consumers: `session_baseline.rs::check_relation_shape`,
  `verify_oracle_authentication` and the transcript word collection read
  `proof.raw`. Their only caller is `verify_selected_step`, which is used by
  the request/component session tests with steps produced by
  `prove_op_relation_with_inputs` (raw committed); no production path feeds
  an executor-produced shared trace into that entry. The shared trace path is
  `verify_shared_trace_steps` -> shared relation entry, so those consumers
  keep their original raw.

## Opcode accounting

The per-opcode account is derived from the three model plans with the same
geometry as the production runs; the temporary accounting test and its output
are archived as `opcode-accounting.rs` and `opcode-accounting.log`. Raw values
are committed as two 64-bit words each (16 B per value); the per-relation
bookkeeping is the root tuple (72 B), the opening fixed fields (32 B), the
inner raw vector length (8 B) and the raw length entry (8 B), 120 B per
relation. No case uses RegressionHead.

| case | opcode | steps | raw values | value bytes |
| --- | --- | ---: | ---: | ---: |
| gpt2 seq16 | Projection | 600 | 3,784,704 | 60,555,264 |
| | MatMul | 289 | 1,232,896 | 19,726,336 |
| | total | 889 | 5,017,600 | 80,281,600 |
| gpt2 seq512 | Projection | 600 | 121,110,528 | 1,937,768,448 |
| | MatMul | 289 | 76,021,760 | 1,216,348,160 |
| | total | 889 | 197,132,288 | 3,154,116,608 |
| gemma3 seq16 | Projection | 234 | 3,096,576 | 49,545,216 |
| | MatMul | 145 | 4,507,648 | 72,122,368 |
| | total | 379 | 7,604,224 | 121,667,584 |
| gemma3 seq512 | Projection | 234 | 99,090,432 | 1,585,446,912 |
| | MatMul | 145 | 162,529,280 | 2,600,468,480 |
| | total | 379 | 261,619,712 | 4,185,915,392 |
| timesfm 16-patch | Projection | 86 | 3,473,408 | 55,574,528 |
| | MatMul | 640 | 491,520 | 7,864,320 |
| | total | 726 | 3,964,928 | 63,438,848 |

The measured proof.bin drop equals value bytes plus 120 B per removed
relation in every case: gpt2 seq16 80,281,600 + 106,680 = 80,388,280;
gpt2 seq512 3,154,116,608 + 106,680 = 3,154,223,288; gemma3 seq16
121,667,584 + 45,480 = 121,713,064; gemma3 seq512 4,185,915,392 + 45,480 =
4,185,960,872; timesfm 63,438,848 + 87,120 = 63,525,968.

## Evidence

Single before/after runs on this machine, release builds, same commit per
case, same parameters/input/policy; gemma3 and timesfm use deterministic
commits created once by the after side and reused by the before side; gpt2
uses its registered commits. Wall and memory numbers are single runs and are
not a stability claim.

| case | stage | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| gpt2 seq16 | proof.bin | 1,980,401,278 | 1,900,012,998 | -80,388,280 |
| | prove wall / VmHWM | 14,176 ms / 4,088,844 KiB | 14,097 ms / 3,922,656 KiB | -79 ms / -166,188 KiB |
| | verify wall / VmHWM | 11,886 ms / 3,897,580 KiB | 11,928 ms / 3,803,076 KiB | +42 ms / -94,504 KiB |
| | trace assets | 196 | 196 | 0 |
| gpt2 seq512 | proof.bin | 10,917,923,358 | 7,763,700,070 | -3,154,223,288 |
| | prove wall / VmHWM | 61,019 ms / 26,967,920 KiB | 47,607 ms / 20,807,020 KiB | -13,412 ms / -6,160,900 KiB |
| | verify wall / VmHWM | 38,050 ms / 18,649,780 KiB | 31,831 ms / 15,041,756 KiB | -6,219 ms / -3,608,024 KiB |
| | trace assets | 196 | 196 | 0 |
| gemma3 seq16 | proof.bin | 5,892,609,694 | 5,770,896,630 | -121,713,064 |
| | prove wall / VmHWM | 33,282 ms / 12,260,064 KiB | 34,064 ms / 12,209,296 KiB | +782 ms / -50,768 KiB |
| | verify wall / VmHWM | 25,893 ms / 10,601,108 KiB | 25,758 ms / 10,412,448 KiB | -135 ms / -188,660 KiB |
| | trace assets | 237 | 237 | 0 |
| gemma3 seq512 | proof.bin | 16,247,655,694 | 12,061,694,822 | -4,185,960,872 |
| | prove wall / VmHWM | 112,940 ms / 51,857,928 KiB | 95,641 ms / 43,678,856 KiB | -17,299 ms / -8,179,072 KiB |
| | verify wall / VmHWM | 74,701 ms / 30,176,680 KiB | 65,930 ms / 23,979,840 KiB | -8,771 ms / -6,196,840 KiB |
| | trace assets | 237 | 237 | 0 |
| timesfm 16-patch | proof.bin | 3,740,667,388 | 3,677,141,420 | -63,525,968 |
| | prove wall / VmHWM | 23,500 ms / 8,043,508 KiB | 23,227 ms / 7,919,716 KiB | -273 ms / -123,792 KiB |
| | verify wall / VmHWM | 21,454 ms / 4,120,572 KiB | 21,228 ms / 4,052,700 KiB | -226 ms / -67,872 KiB |
| | trace assets | 238 | 238 | 0 |

- Commit stage: only the create side runs it, 9,419 ms / 1,085,136 KiB
  (gemma3 seq16), 9,446 ms / 1,085,752 KiB (gemma3 seq512) and 5,437 ms /
  175,028 KiB (timesfm); commit behavior is unchanged by this round.
- Whole-artifact metric, listed separately from the proof.bin sizes: this
  metric counts the whole proof output directory. gpt2 seq16
  1,980,405,259 -> 1,900,016,968; gpt2 seq512 10,917,933,294 ->
  7,763,710,000; gemma3 seq16 5,892,613,669 -> 5,770,900,606; gemma3 seq512
  16,247,665,637 -> 12,061,704,766; timesfm 3,740,935,379 -> 3,677,409,411.
  The remainder after the proof.bin delta is the size change of the other
  files in the proof directory (statement, public inputs, manifest,
  completion marker), not part of the relation account.
- The long-case wall and memory reductions are single-run observations and
  are not a pre-committed performance claim: gpt2 seq512 prove -13,412 ms and
  VmHWM -6,160,900 KiB, gemma3 seq512 prove -17,299 ms and VmHWM
  -8,179,072 KiB. The seq16 cases are within or near run-to-run variation.
- All ten verifications pass with `output_binding = "expected"` and the
  unchanged expected output digests: gpt2 seq16 `135c4d1c…`, gpt2 seq512
  `eac0949d…`, gemma3 seq16 `8683f5e2…`, gemma3 seq512 `6d1df6b5…`, timesfm
  `cc72805b…`.
- Trace asset counts are unchanged; this round removes per-relation raw
  groups, not asset openings.

## Negatives

All run through the full verification path after a serialize round trip
unless noted. Authentication failures and numerical rejections are recorded
separately; none of the relation negatives is a broken Merkle path.

- authentication-consistent output tamper (`shared_matmul_output_tamper_…`):
  the test op changes one output inside the record before the existing prover
  commits it, so every group and root is rebuilt over the altered record; the
  authentication helper first re-verifies every committed group against the
  registered root, then the full verification rejects in the matrix relation
  (`invalid_proof`).
- trusted-plan input replacement (`shared_matmul_plan_input_replacement_…`):
  the proof is produced over one activation while the trusted plan
  materializes another of the same length; authentication stays valid and
  the relation rejects (`invalid_proof`).
- legacy raw declaration (`shared_matmul_legacy_raw_declaration_is_rejected`):
  the same relation is proved with the ordinary input-committed form (inputs
  external, raw committed) and fed to the shared path; the shared entry
  rejects the raw form (`statement_mismatch`). The relation-level test
  `shared_entry_rejects_the_ordinary_raw_form` also shows the ordinary entry
  accepts that proof while the shared entry rejects it.
- raw books mismatch (`shared_matmul_raw_books_…`): a raw length without
  openings and an opening without a length are both rejected
  (`statement_mismatch`).
- malformed bias: the real entry rejects a proof config that declares one
  bias entry for four output slots before any relation or slot index
  (`statement_mismatch`); the relation-level test
  `a_short_bias_is_rejected_before_slot_indexing` shows the matrix branch
  returns `shape_mismatch` instead of indexing a short bias.
- non-target opcode: the shared AffineScale relation still carries raw and
  verifies through the shared entry
  (`shared_non_matrix_relations_keep_their_raw_declaration`); a changed raw
  length is rejected.
- dual-form rejection: the ordinary entry rejects the empty shared form
  (`shape_mismatch`), so no relaxed second path exists.
- other asset and edge substitution negatives are reused:
  `negative_wrong_asset`, `negative_external_and_missing_ports` and
  `negative_external_opening_must_cover_the_full_asset`; their lines are in
  `targeted-tests.log`.

## Boundaries

- No new field, no serialization change, no new protocol, masking, sampling
  or budget change; no input public/private boundary change; private weights,
  anchors, policies and ground truth are untouched; the 45,080 B gate object
  is untouched.
- Workspace suite: 1173 passed / 0 failed / 20 ignored (51 suites). Targeted
  plan_check 37/0 and relation_openings 20/0. The workspace run preceded the
  final test-only assertion; the targeted rerun after it is in
  `targeted-tests.log`.
- Logs and scripts: the ten `before-*`/`after-*` run logs; `release-build.log`;
  `workspace.log`; `targeted-tests.log`; `opcode-accounting.rs` and
  `opcode-accounting.log`; `run-script.sh`; `run-console.log`;
  `run-summary.txt`; `proof-assets.py` and `proof-assets.txt`. Per-stage
  `result.json` and `progress.jsonl` stay under each run directory; proof
  binaries stay on disk and are gitignored. Not pushed.

## Round closure

- Expert batch review: the whole round is accepted, no further rework.
- Final commit list, in order: `973eac0b` (production change: the shared
  matrix relations no longer carry the redundant raw declaration),
  `9e5549b6` (test assertion), `030d5d1d` (evidence record), and this closing
  commit (round flag, NOTES record only; no code change).
- State at closure: hard gate 45,080 B untouched; no anchor, policy, asset,
  ground truth or commitment registration changed; the workspace suite is
  green (1173 passed / 0 failed / 20 ignored); source and evidence are on
  the branch, not pushed.
