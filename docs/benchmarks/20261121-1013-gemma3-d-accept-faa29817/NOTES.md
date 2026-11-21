# D acceptance cold round 20261121-1013 — PASS

## Provenance
- Tree: /data/a04-gemma3, HEAD=faa29817 (clean working tree; probes moved out, not committed).
- Binary: target/release/zkie, built 2026-11-21 10:13 (7,452,912 B) from the clean tree.
- Driver: /data/run-d-accept.sh (ZKIE_ACCEPT_ROUNDS=1, KEEP=1, PROFILE_HIDDEN/STREAM/MEM_TRACE=1,
  no ZKIE_AUDIT_ROWS).
- Independent v2 case: /data/a04-cases/models/gemma3/case.policy — the shared
  /data/zkIE template with only the header model_version 1->2; every other byte identical.
  The shared template was not modified. The generated round policy carries
  model_version=2 (header and limited_table_identity).

## Result (harness summary: ROUND 1 commit=274672 prove=3084227 verify=696984 proof_bytes=5808750389)
- commit: PASS, 274,672 ms, VmHWM 17,654,456 kB.
- prove:  PASS, 3,084,227 ms (51.4 min), VmHWM 285,789,604 kB (285.8 GB),
          proof_bytes 5,808,750,389 (proof.bin 5,716,919,862 + consumer_proof 1,852,200
          + norm-segment-carrier 89,973,076 + manifest 4,679 + statement/public-inputs/COMPLETE).
- verify: PASS, 696,984 ms (11.6 min), VmHWM 146,450,184 kB (146.5 GB),
          output_binding=expected, authenticated_output_digest=8683f5e2...
          (bit-identical to the case GT digest).
- Harness test wall 4,058.15 s; DRIVER_EXIT=0.

## Caliber items
- HWM bounded: prove 285.8 GB against the ~423.8 GB anon ceiling (margin ~138 GB).
- Bytes truthful: round-1 = 5,815,140,745 B total (proof 5,808,750,389 + commit 6,146,624
  + run/policy remainder 243,732).
- Semantic hard item: output_binding=expected and GT digest bit-exact on the verifier.
- Budget record (planner unchanged): own_two_stage k=13 and k=14 both give
  stage0 positions=251 work_bits=9 m=454, last_positions=259, last_r=259/259, last_m=450.
- Negative matrix (component level, clean tree, 6/6 pass):
  the_stacked_closure_member_set_plans_at_k_thirteen,
  the_stacked_band_runs_place_entries_on_the_commit_cells,
  a_stacked_closure_plan_needs_its_marker,
  a_nonzero_pad_cell_is_rejected,
  the_merged_closure_shape_is_two_thousand_forty_rows,
  the_norm_group_identifiers_are_pairwise_disjoint.

## Comparison with the audit round (same fix, probe binary + ZKIE_AUDIT_ROWS)
- prove 3,585,706 -> 3,084,227 ms (-14.0%, audit overhead removed);
  HWM 285,748,228 -> 285,789,604 kB (+0.014%).
- verify 697,800 -> 696,984 ms; HWM 146,410,100 -> 146,450,184 kB.
- proof.bin size identical (5,716,919,862 B); the carrier differs slightly across runs
  with fresh commit salts.

## Boundaries
- This round's data is the acceptance performance evidence (clean binary, no row audit).
- The case-template version update (model_version 1->2) follows the earlier
  case-template version bump precedent (c15c11a3): MODEL_VERSION bump, old v1
  evidence moved out, version negative test. It is a close-out hygiene item and
  does not block this round.

## In-tree record
- Recorded at docs/benchmarks/20261121-1013-gemma3-d-accept-faa29817/ as the
  lane's acceptance record; proof binaries (proof.bin, carrier, consumer proof)
  are excluded per the evidence convention; every textual artifact stays.
- Hygiene status in this tree (earlier case-template version bump precedent,
  c15c11a3 and 2a0cd7d0):
  models/gemma3/case.policy already model_version=2, MODEL_VERSION=2, the
  version negative is in zkie/tests/consumer_request_evidence.rs, and the old
  version 1 evidence directory is already out of the tree.
