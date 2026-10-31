# Legacy verifier migration ledger (work in progress)

Entry points under migration: `verify_committed_shard_dag` (inline) and
`verify_committed_shard_dag_bytes` (bytes). The legacy verifier currently
rejects every proof, so green negative tests that still call it are vacuous and
are not security evidence.

Rule: every migrated test proves an honest fixture at the same entry first,
then exercises the original forgery. Obligations are kept.

Latest full suite: 97 passed, 7 failed (`zkie-ops-lib-full-2.log`); the 7
remaining failures are exactly the pending dependencies below.

## Supported ops: migrated

| Test | Ops | Original obligation | Status |
|---|---|---|---|
| s2_empty_weight_batches_rejected | Projection | Drop every weight batch/opening -> reject | migrated |
| s2_matmul32_empty_commitments_rejected | MatMul | Empty commitments reject; zero-round matmul sumcheck reject | migrated |
| s2_weight_opening_value_and_root_swap_rejected | Projection x2 | Opening value change reject; opening-group swap reject | migrated (model root/mapping replacement is covered separately by `trusted_manifest_rejects_consistent_replacement_model`; group swap is not claimed as root coverage) |
| s2_committed_lookup_roundtrip | Projection+Lookup | Honest non-identity table roundtrip | migrated |
| s2_committed_identity_table_roundtrip | Projection+Lookup | Honest identity table uses the statement-identity path (no table commitment) | migrated |
| s2_identity_flag_forgery_rejected | Projection+Lookup | Forged identity decision reject; decision comes from the statement | migrated |
| s2_table_leaf_point_swap_rejected | Projection+Lookup | Swapped/moved table leaf point opening reject | migrated |

Bytes entry coverage: at least one honest byte proof and one wrong-input byte
proof are exercised in `s2_empty_weight_batches_rejected` and
`s2_matmul32_empty_commitments_rejected`.

## Pending dependency: legacy shard-boundary verifier

| Test | Ops | Original obligation | Status |
|---|---|---|---|
| committed_shard_dag_roundtrip | Projection chain | boundary_tensors/boundary_commitments/cross_binds counts; serialized entry; tampered boundary opening reject | pending dependency: legacy shard verifier migration (not replaceable by a single-graph roundtrip) |

## Pending dependencies: unsupported reductions

| Test | Ops | Original obligation | Status |
|---|---|---|---|
| s2_committed_scatter_gather_roundtrip | Scatter/Gather | Committed perm/gather fractional sections verify end to end | pending dependency: Scatter/Gather reverse reduction |
| argmax_selection_openings_are_individual | Argmax | Selection openings are individual; forged selection reject | pending dependency: Argmax reverse reduction |
| s6_lm_honest_roundtrip | LM chain | Honest prompt/token/argmax statement verifies | pending dependency: LM public token binding |
| s6_wrong_prompt_rejected | LM chain | Wrong prompt reject | pending dependency: LM public token binding |
| s6_response_tampered_rejected | LM chain | Tampered response reject | pending dependency: LM public token binding |
| s6_length_mismatch_rejected | LM chain | Length mismatch reject | pending dependency: LM public token binding |
| s6_out_of_vocab_rejected | LM chain | Out-of-vocab token reject | pending dependency: LM public token binding |
| s6_wrong_position_on_selector_rejected | LM chain | Wrong position on selector reject | pending dependency: LM public token binding |
| s6_wrong_embedding_token_rejected | LM chain | Wrong embedding token reject | pending dependency: LM public token binding |
| s6_selector_root_identity_rejected | LM chain | Selector root/identity reject | pending dependency: LM public token binding |
| s6_tie_rule_rejected | LM chain | Tie rule reject | pending dependency: LM public token binding |
| s6_empty_prompt_roundtrip | LM chain | Empty prompt honest roundtrip | pending dependency: LM public token binding |
| s6_generic_skips_sequence | LM chain | Generic statement skips sequence binding | pending dependency: LM public token binding |

## Other reverse-path work still pending

- LayerNormCentered, Softmax, StableSoftmaxIndex reductions (real GPT-2 chain).
- S8: base-field integer modular wraparound unresolved.
- S9: 90-bit WHIR PCS setting is not a 90-bit whole-protocol claim.
