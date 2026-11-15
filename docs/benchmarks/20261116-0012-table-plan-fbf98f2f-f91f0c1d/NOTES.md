# table-plan: local first+shared prefix over a freshly committed candidate table

## Scheme and provenance

- Scheme: table-plan (public-configuration-derived encoded table plan and the
  local table check prefix).
- Code under test: `fbf98f2f` (derived prefix plan), plus the test-only tamper
  commit `8ddde229` (negatives only; no production path, plan value or proof
  format changes). The trace name carries `fbf98f2f` and its previous commit
  `f91f0c1d`; the last accepted scheme close remains `c54d37c2`.
- Prior candidate instance:
  `docs/benchmarks/20261115-2248-table-plan-4cff3d7f-c54d37c2` (superseded
  first candidate). Its textual artifacts are archived next to this directory;
  its gitignored binaries remain on disk. That instance was committed before
  the local prefix plan was rebuilt from the public configuration; its prefix
  never closed.
- Purpose: run the local first+shared prefix over a table instance committed
  from the same public configuration that derives the plan. Every member,
  message, group stage and budget value is derived from that configuration,
  and the plan is planner-checked, so its validation re-derives the assembly
  from its own public shapes instead of trusting scaffold geometry.
- Boundary: this is a local first+shared prefix proof, not a complete one-time
  authentication. The second chain is not executed. No later-batch capacity,
  asset deletion or single-shot claim is made. The full-model runs below use
  the existing `SharedTrace` prove/verify path with the candidate table
  instance; they are not hidden-chain acceptance.

## Candidate instance

Both cases commit the same public table configuration (shape
`{log_n:12, ell:2048, r:128, v:10, answers:1, salt:40}`, one-time distance
`m_once=284`, `m_later=0`, parameters digest
`f2b58e54…c6c98`); the per-commit random salts and tails make distinct table
roots and distinct ledger instances.

| Case | commit | anchor | table root | ledger |
|---|---|---|---|---|
| gpt2-seq16 | PASS 3,848 ms, VmHWM 356,744 KiB | `1fd8465d…304f` | `f439e4ba…2862` | charged 52 |
| gpt2-seq512 | PASS 3,850 ms, VmHWM 356,672 KiB | `d864b5e6…86cc` | `9b3383b0…798a` | 0 |

The candidate verification configuration is the explicit registration
`instance/case.policy` with the two anchors above. The derived table plan
identity equals the committed `meta.identity` in both cases (asserted at the
local entry).

## Full-model cross checks

- Baseline policy (`models/gpt2/case.policy`, baseline anchors) over the new
  instance: prove PASS (seq16 1,887,840,920 B / 14,020 ms / 3,899,260 KiB;
  seq512 6,798,965,241 B / 44,222 ms / 18,922,412 KiB), verify FAIL
  `statement_mismatch` in both cases.
- New candidate policy over the previous instance's seq16 proof: verify FAIL
  `statement_mismatch` (`cross-verify-old.log`).
- Candidate policy, fresh `run2`: prove PASS and verify PASS with the expected
  output binding.
  - seq16: prove 1,887,840,920 B / 13,992 ms / 3,897,948 KiB; verify
    `135c4d1c…a80` over 16 elements / 11,884 ms / 3,794,148 KiB.
  - seq512: prove 6,798,965,241 B / 43,806 ms / 18,924,808 KiB; verify
    `eac0949d…fdd` over 512 elements / 29,506 ms / 14,089,736 KiB.
- Proof bytes are single-run values and differ from the previous instance by a
  few bytes through the sampled table roots; they are not a stability claim.
  The protected 45,080 B object is untouched by this round.

## Local first+shared prefix (the round's evidence)

The plan is derived from the registered public configuration, then validated
with its planner check on:

- matrix rows 1024, logical columns 1024, shown-map width 1024, W SEND
  openings per message 2; at_start index mask rows 1; own fresh-mask rows 1;
  fresh ids 2/12; BASE positions 66 (shared) and 53 (own); stage positions
  shared `[65, 71, 66]`, own `[65, 66, 53]`.
- First-stage codes: main `{log_n:18, ell:16384, r:130}`, mask
  `{log_n:18, ell:256, r:66}`; column draws = column budget = 52.
- `plan validate ok`: the planner check re-derives the assembly input from the
  plan's public shapes and the plan's stage choices, and matches codes, every
  stage's fold/CS/BASE shapes, BASE counts and fresh-mask row counts.
- Prove: first message 5,992 B, shared message 713,904 B, 122,881 ms.
  Verify: seal `dd4b00ee…8ab17`, 77,731 ms. Whole test 414.01 s including the
  table rebuild from the real weight rows and all negatives. Process VmHWM was
  not captured for this test, so no peak-memory number is claimed for it.

### Table column accounting

- The prefix reserves exactly one batch of the derived column count on the
  original table: one `begin_charge` reserve of 52 before the first message and
  the same charge consumed by the shared stage, not a per-stage charge. After
  the run the seq16 table ledger shows the one batch: `reserved=52 total=52`.
- The sample list length is the same 52; the charge is not reduced by unique
  positions and is not refunded.
- A restarted keeper over the same ledger opens at the spent count and the next
  prove is refused before any read; the count stays 52.
- The seq512 table ledger remains 0 because no prefix ran on it. The previous
  instance's two production ledgers (`46773cc3…`, `6251981d…`) remain at 0;
  nothing was reset. Its failed attempts used temporary ledgers outside the
  asset root.
- Other objects (W messages, Omega, Gu, activation and mask messages) use their
  own codes and random sources; no other object is charged to the table ledger.
  The one-time column count 52 is not the table's physical tail capacity 128.

### Rejections exercised

- Plan identity replacement, plan table-shape replacement, meta identity
  replacement, meta shape replacement (`m_once+1`), and trusted-reader root
  replacement are all rejected at verification.
- The first seal and a shared column opening are tampered; both are rejected.
  Every tamper verify runs on a plan clone whose verifier read ledger starts
  fresh, so a spent ledger cannot produce a vacuous rejection.
- The baseline policy rejects the new instance, and the new policy rejects the
  previous instance's proof (above).

## Conclusions

- The registered production table object feeds the local first+shared prefix:
  the plan assembled from the public configuration validates against the
  planner check, proves, verifies, and charges exactly one derived column batch
  on its own durable ledger. Incompatible plans and tampered proofs or readers
  are rejected before any acceptance.
- The superseded first candidate's plan was still assembled from scaffold
  components (chain mask blocks and fixture index rows); it never reached a
  valid local prefix. The rebuild from the public configuration is what closed
  the gap.

## Tests

- Workspace: 1,182 passed, 0 failed, 20 ignored across 51 test binaries on the
  evidence source (`fbf98f2f` plus the test-only `8ddde229`). The local prefix
  test above ran against the committed instance and is the round's evidence,
  not part of that count.

## Open items

- The consumer's request chain, member claim and second chain over this table
  object; real claim/maps derivation; complete key authentication; later
  batches; asset deletion; single-shot. None is closed or claimed here.
- Prefix process peak memory was not captured; a later round that repeats the
  prefix should record it with the same measurement basis.
