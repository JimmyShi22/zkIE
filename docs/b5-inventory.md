# B5 plaintext removal inventory (read-only)

- Date: 2026-11-15. Code head: `cdb4d752`; no code change, no mechanism run,
  no new measurement. Every number below is either a code-derived count, a
  case-manifest file size, or a measured value already archived in the B4
  traces.
- Basis: the agreed B5 scope (remove plaintext, ledger budget, five-case
  matrix), the confirmed deletion rule (a table may be deleted only when a
  confirmed public definition exists — trusted digest, entry count, decoding
  fixed before challenges — and every consumer is switched away from the
  external opening), and the fixed non-deletion list (MatMul/Projection/
  RegressionHead weights, embedding/lm_head, norm gain/bias, and the
  consumers whose real claim/maps source is still open).
- Evidence status: the four migrated tables are already out of the proof
  trace as of the B4 round; this inventory separates that realized, measured
  removal from the additional commitment-side deletion that B5 would
  introduce.

## 1. Assets committed and opened by the proof

The proof asset list is `ModelCommitment::assets` (built from each model's
`asset_specs`, opened by `AssetStore::open_asset` in the executors). Counts:

| model | specs | trace assets after B4 | removed in B4 |
| --- | ---: | ---: | ---: |
| gpt2 | 199 | 196 | exp, rsqrt, gelu |
| gemma3 | 245 | 242 | exp, rsqrt, gelu |
| timesfm | 241 | 238 | exp, rsqrt, silu |

The trace counts are read from the B4 proof headers; the spec counts are
from `models/*/src/executor.rs::asset_specs` (gpt2 12×16 + 7; gemma3
18×13 + 11; timesfm 20×11 + 21).

### (a) Confirmed public definition, all consumers switched — B5 candidate

| group | tables | file bytes | proof value bytes (8 B/entry) |
| --- | --- | ---: | ---: |
| exp | gpt2, gemma3, timesfm | 8,388,608 | 16,777,216 |
| rsqrt | gpt2, gemma3, timesfm | 29,360,128 | 58,720,256 |
| gelu | gpt2, gemma3 | 67,108,864 | 134,217,728 |
| silu | timesfm | 4,194,304 | 8,388,608 |

- The fixed identities live in `models/*/src/plan.rs::public_tables`
  (locator, SHA-256, entry count, signed i32 decoding); `trace_plan_public`
  replaces every matching `PlanSource::Asset`, and the executors'
  `public_table_paths`/`public_pool` keep their ports out of the external
  pool. The verifier's `PlanSource::PublicTable` branch reads the file once
  per run after checking size and digest.
- Proof-side status: already removed and measured in B4 (Section 2).
- Commitment-side status: all four files are still hashed by the commit stage
  and still appear in the anchor's leaf set; deleting them from
  `asset_specs` is the remaining B5 action, and it changes the anchor (see
  the risk in Section 4).

### (b) Not removable under the confirmed rule

| group | examples | reason |
| --- | --- | --- |
| MatMul/Projection/RegressionHead weights and biases | per-layer `*_w`, `*_b`, `lm_head`, `proj_*`, `head_*` | no public value definition; the verifier must authenticate the opening |
| embedding / learned position | `embed_tokens`, `lm_head` (gpt2), `pos_emb` | model weights, listed as not migrated |
| norm gain/bias | `ln*`, `*_norm`, `final_norm`, `lnw` | model weights, listed as not migrated |
| masks and structures | gpt2 mask, timesfm `mask_q`, `gather` | structural data, listed as not migrated |
| real claim/maps consumers | the open second-stage consumer path | the real claim/maps source is not closed; its consumers stay |

The non-deletion list is fixed by the earlier confirmation and is not
reopened here.

### (c) Boundary undecided — needs a ruling before any change

| group | assets | bytes | why undecided |
| --- | --- | ---: | --- |
| RoPE cos/sin (gemma3) | `rope_cos_global`, `rope_sin_global`, `rope_cos_local`, `rope_sin_local` | 262,144 each | no public digest/entry/decoding identity is fixed in the plan; consumers still read asset slices; the earlier ruling kept RoPE on its original verification |
| rsqrt extension (gemma3) | `cases/gemma3/rsqrt_ext_i32.bin` | 59,397,052 | no generator or confirmed public definition; the extension selection and range refusal must keep their original verification |
| scaling/config assets (timesfm) | `q_scale`, `scale_i64` | 512 + 16 | no confirmed public definition; `scale_i64` enters the plan as an asset value, not a table |

These are not in the fixed non-deletion list either, so they are listed for
an explicit ruling rather than deleted or kept by inference.

## 2. Volumes and expected net bytes for (a)

The opening value bytes are sequence-independent (the tables do not depend
on tokens); the measured net proof drop differs only by per-instance opening
and relation-proof bookkeeping. Measured B4 values, single before/after runs:

| case | net proof drop | table value bytes | remainder |
| --- | ---: | ---: | ---: |
| gpt2 seq16 | 209,754,240 | 209,715,200 | 39,040 |
| gpt2 seq512 | 209,719,680 | 209,715,200 | 4,480 |
| gemma3 seq16 | 209,738,112 | 209,715,200 | 22,912 |
| gemma3 seq512 | 209,720,832 | 209,715,200 | 5,632 |
| timesfm seq16 | 83,970,272 | 83,886,080 | 84,192 |

- The seq16 and seq512 rows are the two sequence bases for the same model;
  timesfm has only the fixed 16-patch case.
- If B5 only removes the four entries from the commitment, the additional
  proof-byte gain is **zero** (the openings are already absent); the gain is
  commit-side: 104,857,600 B of file hashing for gpt2, 104,857,600 B for
  gemma3, 41,943,040 B for timesfm, plus the removal of those leaves from the
  anchor. These are estimates from the file sizes and the realized trace
  removal, not a new measurement.
- Any B5 claim of further proof-byte reduction would be a new mechanism and
  is not supported by the archive.

## 3. Ledger and budget touch points

| mechanism | entry point | status | B5 impact |
| --- | --- | --- | --- |
| durable per-column budget | `DurableColumnBudget::initialize/open/reserve`, `crates/zkie-engine/src/pcs/request/read.rs:325` | landed in B3, initialized at asset-table commit | keep the limited table's ledger; a removed public table must not create or open a ledger |
| per-table keeper and ledger directory | `AssetTableKeeper`, `column_ledger_dir`, `initialize`, `crates/zkie-ops/src/proof/column_session.rs:463,476,563` | landed in B3 | capacity check `hiding_positions` stays; path derivation unchanged |
| plan read budget | `ReadLedger`, `ReadLedgerCell`, `VerifierReads` (`read.rs`; `hidden_send_session.rs:8591`) | landed in B3 | unchanged unless the plan's read counts change |
| per-tree Omega opening count | `TreeOpeningCount` (`zkie-engine` tree book; `multi_stage_iopp.rs:41`) | landed in B3 | the public tables are not Omega trees in these chains; counts unchanged |
| shared linear bounds | `SharedLinCheck`, `shared_lin_bounds` (`proximity_planner.rs`) | landed in B3 | unchanged |
| provider ledger path | `ProviderConfig::column_ledger_dir` (`zkie/src/provider.rs:61`) | landed in B3 | unchanged |
| anchor leaf accounting | `verify_model_anchor_with_tables` `leaf_count == specs.len() + tables.len()`, `model_commitment.rs:1055` | B5-new surface | deleting assets changes the expected count and the anchor |
| policy counting and registration | project policy fields `anchor_id`, `model_id`, `[case.limited_table_identity].asset_id`, `zkie/src/policy.rs:611-627` | B5-new surface | re-registration required if the asset list changes |
| asset id assignment | sequential ids in `models/*/src/executor.rs::asset_specs` | B5-new risk | removing a non-trailing entry renumbers later assets (gemma3 extension, timesfm exp/rsqrt/input) |

## 4. B5 execution plan draft (change points only)

1. Ruling first: confirm whether B5 deletes the four entries from the
   commitment or keeps them committed but unopened (the current B4 state).
   The deletion variant changes the anchor and needs policy and instance
   re-registration; without that ruling the work cannot start.
2. Executor public-table reads: after deletion the prover can no longer read
   the tables through `AssetStore::value`; it needs a trusted public-root
   read for the four tables (same files the verifier reads), keeping the
   range/mmap discipline.
3. Remove the four entries from `asset_specs`; keep the limited table's
   asset id stable; audit trailing-id shifts (gemma3 extension, timesfm
   exp/rsqrt/input). No whole-file materialization into any artifact.
4. Update the anchor leaf accounting and policy registration; regenerate
   the tracked policy and registered commits with the new identities.
5. Negatives: old policy against the new commit and the reverse; missing or
   extra asset entry; renumbered asset id; a removed table's opening supplied
   by the prover; a removed table's ledger directory requested; swapped or
   short public file (already covered in B4).
6. Five-case matrix: same-machine before/after on the production CLI, with
   the re-registered policy/instances; report commit/prove/verify wall and
   peak memory, proof bytes (expected flat on the proof side), and the
   commit-side hashing drop; archive the trace.

## 5. Risks and boundaries

- The main risk is identity: deleting entries changes the anchor leaf set
  and renumbers later assets, so the tracked policy and every registered
  commit stop matching. This is a trusted-artifact change requiring a
  ruling, not an implementation detail.
- The prover-side public-file read is a new execution path (not a new proof
  mechanism); it needs confirmation that it stays inside the ruled public
  source model.
- RoPE, the rsqrt extension and the timesfm scale/config assets stay on their
  original verification; no deletion without a confirmed public definition.
- Weights, embedding/lm_head, norm parameters, masks/structures and the open
  claim/maps consumers are not removable in B5.
- The hard gate (45,080 B) must stay unchanged; no evidence may claim a
  proof-byte gain that the archive does not show.
- No mechanism was implemented and no measurement was run for this
  inventory; evidence for the undecided groups is incomplete by design.
