# Assembled proximity plan and session package

## Scheme

- Scheme: `assembled-proximity`.
- Current code: `3e97b834` (this scheme), following `d49e98c4`.
- Previous code: `d49e98c4`.
- Scope: the projection consumer's hidden session over the registered GPT-2
  seq16 artifacts. The trusted planner assembles the production proximity
  configuration and the verifier re-derives it; the canonical session
  encoding enters the verification package as `session.bin`.

## Purpose

Close the parameter-derivation chain: replace the fixed diagnostic constants
in the consumer's proximity plan with values derived from the public
production configuration and the real member layout, check them independently
on the verifier side, and carry the session in the proof package. The run is
not a performance comparison. No protocol message order changed, no plaintext
asset was removed, and no old check was deleted.

## Measurement basis

- Same machine, release build.
- Focused derivation tests:
  `cargo test -p zkie-ops --lib proximity_planner`.
- Real run:
  `cargo test -p zkie --release --test projection_consumer_diagnostic
  real_projection_consumer_closes_in_hidden_input_mode -- --ignored
  --nocapture`, with the session dump path set; timed with `/usr/bin/time -v`
  for the peak memory.
- Workspace gate: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped by the
  merged-request test and reverted afterwards.

## Assembly result

The trusted assembly selects the first candidate of the public production
configuration mapped to the executor's one-main-code layout:

- first main code: `log_n = 18`, message `16384`, random tail `130`;
- first mask code: `log_n = 18`, message `256`, random tail `66`;
- both chains have three stages; the shared group's first stage is
  `k = 1, h = 0` and the own group's is `k = 5, h = 2`, both from the real
  member rows;
- shared stage-0 positions `65`, own stage-0 positions `65`, own stage-0
  capacity `max_zero = 65`; reserved rounds of the frozen plan `28`.
- stages (fold bits, main log/message, mask log/message, positions, main
  tail, mask tail), shared: `(1,0: 18/16384, 18/256, 65, 130, 66)`,
  `(4,2: 14/1024, 12/256, 71, 71, 71)`, `(0,0: 14/1024, 14/1024, 66, 66, 66)`;
  own: `(5,2: 18/16384, 18/256, 65, 130, 66)`, `(4,3: 14/1024, 13/128, 66, 66, 66)`,
  `(0,0: 15/1024, 15/512, 53, 53, 53)`.
- BASE positions: shared `66`, own `53`. The own first-stage main tail is the
  shared list plus the own list; the mask tail is the own list plus the
  index-mask SEND reading.

## Key numbers

| figure | value |
|---|---:|
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |
| session.bin | 1,075,512 B |
| session.bin sha256 | `ee19deb10579a67aa0d0e7f030a3b89291570a9375436ec3db8348effa2b3cab` |
| package manifest total | 2,191,273,523 B |
| public commitment in package | 42,493 B |
| trace proof in package | 2,190,155,518 B |
| real consumer run wall | 77.85 s (test harness); 1:30.58 elapsed including the 12.6 s release rebuild |
| real consumer run peak memory | 8,787,812 KiB |
| workspace | 51 suites, 1014 passed, 0 failed, 19 ignored |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Trusted plan chooses a bounded public configuration and fully checks it | production group and stage lines; parameter parsing | `AssemblyCandidate`, `production_candidates`, `assemble_proximity` in the planner | assembly tests: the production candidate assembles; an unformable group names its point |
| The first choice carries the first stage's code lengths; the first fold bits come from the real layout; later choices are post-first stages | stage-list sizing rules; group row bits | `ChainStart`, `solve_chain_stages`, `SolvedStage` with separate main and mask tails | stage-solve tests: first stage codes vs middle-stage fold bits, derived positions |
| Last codes are the last choice; message lengths, out-of-domain entries and positions are derived | per-stage sizing and position iteration | `solve_stage_positions`, `validate_solved_chain` | chain test asserts every derived stage field |
| `max_zero` is the smallest remaining openable capacity of the own stage-0 objects, main and mask separate, SEND and shared reads charged | joint first-stage reading; per-object hiding conditions | `solve_first_stage_one_mask` computes per-object tails including spends; the assembly checks every object | assembly test: own stage-0 list fits `max_zero`; the real run proves |
| Reserved rounds are the frozen complete plan's exact output | complete-plan round counting | both chains' round counts summed before any run | assembly test recomputes the sum and compares |
| Identifiers follow registration order, collisions are rejected, one physical object keeps its tree, root, run and shared ledger | registration-order identifiers; tree registration | the existing chain derivation allocators are reused; no second numbering system | planner check rejects changed stage codes and fold bits; existing derivation tests |
| The transition two-position W reading is charged, not claimed as proximity | transition allowance | per-W spend in the assembly input and in `max_zero`; samples unchanged | real run; the plan check rejects a changed code tail |
| The index mask stays at the second chain's own proximity | at-start slot allocation | index-mask plan unchanged; treated as an own mask object with its SEND spend | real hidden run with the index mask |
| The formal hidden entry runs with `planner_checked = true` | trusted plan check | `hidden_plan_for` builds codes, trees, stage lists and BASE counts from the assembled values; the verifier re-derives them without searching | real run passes; changed position, tail, code, BASE count or fold bit is rejected |
| `session.bin` enters the proof package after the trusted table | verification package | canonical session encoding written as `session.bin`; manifest roles public-commitment, trace, lin-session; the verifier decodes and verifies from the package | real run prints the session and package bytes; manifest verifies; tampered and trailing sessions are rejected |
| One row rule, no removed plaintext or old check | single implementation | the column-message row rule extracted and shared with the shape constructor | hard gate unchanged; workspace green |

## Negative checks

- The verifier-side re-derivation rejects a plan whose own stage-0 position is
  changed by one, whose main or mask code tail is changed, whose last-stage
  main message length is doubled, whose BASE count is changed, or whose first
  stage fold bits are changed.
- The session package rejects a changed session byte through the manifest and
  a trailing byte through the session decoder.
- The existing hidden-mode checks still reject a missing index-mask message
  and a changed table identity.

## Conclusions

- The production proximity plan is an output of the trusted derivation chain:
  public codes and fold bits, member-derived group bits, joint first-stage
  tails including the SEND and shared reads, per-stage positions, and the
  exact reserved round count.
- The verifier re-derives the same values from the same trusted inputs and
  rejects an inconsistent plan; it never searches for another configuration.
- The session is carried in the proof package and verified from those bytes.

## Open items

- The planner-checked assembly covers the projection consumer's members: the
  W messages, Omega, the index mask and the shown map. The output-closure
  messages and the norm groups are not yet in the assembly input; a
  planner-checked plan carrying them is rejected with a named point and they
  keep their existing checks.
- The executor uses one main code and one mask code for both proximity groups.
  The production configuration's split group lines are therefore mapped onto
  one code; split layouts remain outside the candidate domain.
- The round accounting is the structural stage-list count approved earlier;
  the per-round whole-error and message-byte comparison remains a future
  check.
