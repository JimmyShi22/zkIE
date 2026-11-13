# TURN 492 acceptance matrix for the rsqrt index and rstd table lookup

- Scheme: `rstd-acceptance`
- Change: test-only additions on top of code commit `5edf0da8`: selector,
  rounding, remainder and quotient negatives, the table-encoding negative, the
  plaintext scan, and extended message tampering (sample answer, lookup proof
  value).
- Baseline: `5edf0da8` (rstd wiring; trace `18ab68dc`).
- Purpose: map every row of the TURN 492 acceptance and negative matrix to an
  executable test through the real hidden prover/verifier and record the
  coverage and the gaps.

## What changed

- `rstd_selector_tampering_is_rejected`: two active segment selectors and no
  active selector are both rejected.
- `rstd_rounding_remainder_tampering_is_rejected`: a shifted rounding quotient
  and a shifted rounding remainder are rejected.
- `rstd_index_quotient_tampering_is_rejected`: a shifted segment quotient is
  rejected.
- `rstd_table_encoding_tampering_is_rejected`: a table whose encoding differs
  from the trusted plan is rejected by the verifier configuration compare.
- `rstd_message_tampering_is_rejected` also tampers the rstd sample answer and
  a lookup proof value.
- `rstd_proof_does_not_carry_plaintext_values`: the fixed-width lookup proof,
  outcome and sample buffers keep eight-byte alignment, so a plaintext rstd
  value would appear as its little-endian bytes plus sixteen zero bytes at an
  eight-byte boundary; none of the table values appears.

## Evidence

- rstd: 16 passed (`focused.txt`).
- norm: 29 passed; lookup: 24 passed.
- Workspace: 48 suites ok, 894 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Matrix mapping

| # | Row | Test / evidence | Status |
| --- | --- | --- | --- |
| P1 | variance nonnegative, 0 and 1 | `rsqrt_row` boundary unit test (L1) | pass |
| P2 | half rounding parity | boundary unit test (L1) | pass |
| P3 | FINE boundary | boundary unit test (L1) | pass |
| P4 | COARSE_END boundary | boundary unit test (L1) | pass |
| P5 | high carries | boundary unit test (L1) | pass |
| P6 | index N-1 / N | row constructor rejects index >= N (L1); `rstd_index_past_the_real_length_is_rejected` (L4) | pass |
| P7 | large table random point | streaming equals direct fold over the real frozen table (L2, env-gated release test) | pass |
| P8 | normal proof and later checks | `rstd_lookup_proves_and_verifies` (L3/L4) | pass |
| N1 | wrong selector | `rstd_selector_tampering_is_rejected` | pass |
| N2 | wrong raw/remainder/slack | `rstd_rounding_remainder_tampering_is_rejected`, rsqrt slack negative | pass |
| N3 | wrong index | `rstd_committed_index_tampering_is_rejected`, `rstd_index_past_the_real_length_is_rejected`, `rstd_index_quotient_tampering_is_rejected` | pass |
| N4 | wrong value | `rstd_committed_value_tampering_is_rejected` (a real table value at the wrong index) | pass |
| N5 | wrong table row | `rstd_table_content_tampering_is_rejected` | pass |
| N6 | padding | `rstd_table_padding_tampering_is_rejected` (source lies about metadata), `rstd_key_padding_tampering_is_rejected` | pass |
| N7 | identity/length/encoding/version | `rstd_table_identity_tampering_is_rejected`, `rstd_table_length_tampering_is_rejected`, `rstd_table_encoding_tampering_is_rejected`; open-level version negative (L2) | pass |
| N8 | base/extension mapping | SplitRows tests (L2) | pass |
| N9 | changed committed witness | selector/rounding/index/value negatives above (all re-committed) | pass |
| N10 | duplicate or extra key sources | two-column key plan unit test; the plan is trusted input, not a prover message (L1) | pass at L1 |
| N11 | key roots, openings, sample answers | `rstd_message_tampering_is_rejected` (bits root, mask root, key claim, sample answer, proof value) | pass |
| N12 | table point/bit order | byte-table bit-order test, fold adjacency (L1/L2) | pass |
| N13 | wrong challenge | plan digest and table identity bind the transcript; identity and padding negatives reject; no separate cross-instance reorder test | partial with note |
| N14 | proof carries no plaintext | `rstd_proof_does_not_carry_plaintext_values` | pass |

## Boundaries

- The index itself is not scanned as plaintext: the committed index in this
  fixture is zero, indistinguishable from structural zero field elements; the
  rstd values are scanned.
- Cross-instance transcript reordering is not a separate negative; the chain
  replay, the plan digest and the recomputed claims already bind the challenge
  order. Recorded as a gap rather than a tested row.
- L5 (model-level and whole-system acceptance) belongs to 2b.4 and later; it is
  not part of this batch.

## Open items

- Model-level plan assembly, the rounding claim linkage and 2b.4 inputs.
- L5 acceptance after 2b.4.
- No push in this round.
