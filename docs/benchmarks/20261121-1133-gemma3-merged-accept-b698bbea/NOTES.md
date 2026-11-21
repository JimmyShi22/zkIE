# Gemma 3 merged acceptance (relation fix + multiplicity derivation + stacked closure form)

## Provenance
- Tree: /data/a04-gemma3, HEAD=b698bbea at round start. Two later commits landed
  while the rounds ran: 158998f9 (evidence notes wording) and 1531fe16
  (comment-only neutralization, no code path change). The stage binary is the
  clean 10:13 build of the same code paths.
- Driver: /data/run-merged-accept.sh — four rounds (one cold + three warm),
  KEEP=1, PROFILE_HIDDEN/STREAM/MEM_TRACE=1, no row audit.
- Independent v2 case: /data/a04-cases/models/gemma3/case.policy; the shared
  case template is untouched.

## Four-round table
| round | commit ms | prove ms | verify ms | commit VmHWM KiB | prove VmHWM KiB | verify VmHWM KiB | proof bytes |
|---|---|---|---|---|---|---|---|
| 1 cold | 275,710 | 3,077,977 | 694,291 | 17,654,332 | 285,798,424 | 146,441,640 | 5,809,122,791 |
| 2 warm | 276,174 | 3,079,749 | 697,028 | 17,653,900 | 285,833,444 | 146,441,808 | 5,808,912,587 |
| 3 warm | 275,719 | 3,080,205 | 693,217 | 17,654,668 | 285,751,380 | 146,441,652 | 5,809,300,510 |
| 4 warm | 271,454 | 3,074,583 | 693,601 | 17,654,136 | 285,822,384 | 146,435,376 | 5,808,879,388 |

- Warm medians (rounds 2-4): commit 275,719 ms; prove 3,079,749 ms with VmHWM
  285,822,384 KiB; verify 693,601 ms with VmHWM 146,441,652 KiB; proof bytes
  5,808,912,587 B. Cold numbers are not used in the ledger.
- Harness wall: 16,198.64 s; DRIVER_EXIT=0; all twelve stage runs PASS.

## Byte account (per round)
- proof.bin identical in all four rounds: 5,716,919,862 B.
- consumer proof: 1,851,240 / 1,851,368 / 1,852,104 / 1,860,616 B.
- norm-segment carrier: 90,346,444 / 90,136,108 / 90,523,300 / 90,093,668 B.
- manifest ~4,670 B; statement 435 B; public inputs 72 B.
- proof directories: 5,809,122,791 / 5,808,912,587 / 5,809,300,510 /
  5,808,879,388 B; commit directories: 6,136,173 / 6,130,513 / 6,140,475 /
  6,137,121 B.
- Lookup-multiplicity reference: the removed materialization is
  8 * 2^25 symbols * 24 B = 6,442,450,944 B. This is a plan-derived expectation,
  not a measured pre-fix whole round; it is quoted with that dual attribution.

## Semantics (hard item)
- Every round: output_binding=expected and authenticated output digest
  8683f5e2e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562, identical
  across all four rounds and equal to the case ground-truth digest.

## Peak and planner
- prove VmHWM stays at the ~285.8 GB level with no round-to-round growth (max
  285,833,444 KiB against the ~423.8 GB ceiling; margin ~138 GB).
- Planner budget record unchanged: the two-stage own-group candidate gives
  stage0 positions=251, work_bits=9, m=454; last_positions=259,
  last_main_r/last_mask_r=259/259, last_m=450 — identical for both fold-k
  candidates and identical to the earlier candidate report.

## Negative matrix (component level, clean tree, 6/6 pass)
- the_stacked_closure_member_set_plans_at_k_thirteen
- the_stacked_band_runs_place_entries_on_the_commit_cells
- a_stacked_closure_plan_needs_its_marker
- a_nonzero_pad_cell_is_rejected
- the_merged_closure_shape_is_two_thousand_forty_rows
- the_norm_group_identifiers_are_pairwise_disjoint

## Hard gate
- The historical anchor is unchanged and takes precedence: norm-free toy proof
  45,080 B, sha256 7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  as recorded in-tree (20261113-1336-rsqrt-enable-c4577182-3f6020d4 and
  20261114-2052-request-rows-hook-66f06b42-2e65871c). This round neither
  re-measures nor revises it.

## Workspace
- Workspace suite after the rounds: WS_EXIT=0, 56 test binaries, 1346 passed,
  0 failed (log archived as workspace.log in this directory).

## Boundaries
- This merged round is the acceptance performance evidence (clean binary, no row
  audit). Round-1 is cold and not used in the ledger; warm medians above are.
