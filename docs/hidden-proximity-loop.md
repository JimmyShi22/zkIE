# Hidden proximity loop: binding sources, envelope and compatibility

## Scope and status

This note describes the implemented multi-stage IOPP loop of the hidden
proximity groups and the proof envelope that carries it. The loop, the
envelope and the final-verifier checks are implemented and exercised by
tests. The measured budget and the security parameters of a production run
are recorded by the acceptance run with the same measurement basis as the
benchmark matrix; no number from a small fixture is presented here as final.

- `[FACT]` verified in the code or checked against it.
- `[ASSUMPTION]` an explicit condition the argument needs.
- `[GAP]` a missing check or mechanism outside this construction.

## Trusted stage list

`[FACT]` Each proximity group carries its stage list in the trusted plan:
`shared_stages` and `own_stages`. The list is plan data, never proof bytes.
An empty list keeps the single-stage delivery; a non-empty list must hold at
least two stages. The first stage repeats the group's main and mask codes.
Every later stage is made by a code switch: the folded main message fills the
next stage rows, the mask room grows by the stage's hiding shift, and the
last stage has zero fold rounds and the BASE position count.

`[FACT]` The actual fixture's own group has 8,258 main rows: 16 W messages
(2 rows each), Omega (1), the Dot shown map (1), 16 slack messages (2 rows
each), 16 bits messages (256 rows each, rounded up from 129 by the answer-pad
capacity and the power-of-two rule), 16 map messages (256 rows each) and the
closure mask (0 main rows). Its stage-0 fold is therefore `k=14, h=2`; the
next stage is `k=13, h=13` and the last stage is `k=h=0`. The earlier
`4,162`/`k=13` figure divided by `ell` without the `ell - answers` capacity
and is withdrawn.

`[FACT]` The stage FOLD and switch tree numbers, the fold group, member and
fresh identifiers, the stage seeds and the BASE star plan are derived from
the trusted stage list. Identifiers are allocated in stage order from the
request's bound identifier set, always taking the smallest unused value, and
the allocation uses checked arithmetic. A missing or duplicate identifier, a
stage-0 conflict, or a derived tree number that collides with a bound or
derived tree number is rejected before any proof work.

## Stage 0 and the old mask

`[FACT]` The first switch reads the old mask from the verified stage-0
FOLD basis and layout. The accepted sources are:

- shared group: the `G_u` and `Omega` members are maskless, so the layout
  determines the old mask as zero. The verifier derives the zero claim and
  zero tail from the layout; no zero tree is committed and the proof cannot
  declare the source.
- own group: every committed mask member is opened at the stage-0 positions
  and folded under the layout's gamma into the old mask.
- the stage-0 FOLD fresh mask tree is a separate one-time mask. It is committed
  and opened independently and can never stand in for the old mask.

`[FACT]` The stage-0 FOLD keeps its main tail, old-mask tail, fresh-mask tail,
claim, kept count and pad; the switch records carry the openings the plan
requires. The last stage's BASE openings live in the canonical LastBase
artifact.

## Switch semantics

`[FACT]` Switch `i` opens the rows of stage `i - 1`, not the rows of stage
`i`. Its position count and work bits come from the previous stage entry; the
last stage entry describes the BASE target and its work bits never enter a
switch replay. The verifier rebuilds the same sequence from the trusted plan
and rejects a missing, reordered, duplicated or truncated stage, a target
that differs from the plan, and a stage sequence whose query or work differs
from the previous entry.

`[FACT]` A switch absorbs its target, identity, declared root, sample answer,
nonce and the stated openings into the transcript in a fixed order; the next
stage member is the code-switch output actually registered and root-verified
by the previous switch, and the following FOLD layout must equal the plan.

## Envelope and compatibility

`[FACT]` The hidden output-head proof envelope version is `5`; the session
layout carried inside it is `14`. Version `5` carries each proximity group's
end as a plan-selected union:

- a group without stages ends in the single-stage BASE artifact, which carries
  its member, fresh, star and star-mask openings directly;
- a group with stages ends in the loop proof, whose switch records carry the
  previous stage member and fresh openings and whose canonical LastBase
  artifact carries the final openings.

`[FACT]` The envelope and the session layout are decoded against the trusted
plan. A previous version is rejected with an unsupported-version error and an
unknown version with an invalid-encoding error; there is no silent fallback
to the single-stage tail. A staged plan rejects an artifact whose tails are
the single-stage artifacts, and a plan without stages rejects a loop tail.
The decoder bounds the switch count, the member count, the opening counts and
the loop byte length, and rejects trailing bytes.

`[FACT]` The outer shared and second messages never treat a proof-declared
BASE outcome as trusted: each side replays its loop from the trusted chain and
the stage-zero object, checks the final claim, opens the LastBase artifact and
returns a seal that must match the shared seal and the second chain.

## Parameter derivation

`[FACT]` One-time switch and fold material and the BASE star material are
derived from the session seed under separate domain purposes: one purpose for
each switch index, one for each stage FOLD seed, one for the stage FOLD salts,
and two for the BASE star. Public labels do not carry private seeds and do not
depend on future roots or challenges. A drawn value that must be non-zero is
redrawn a bounded number of times before the derivation fails.

`[FACT]` The BASE star plan is derived from the last stage's main and mask
codes and the BASE read layout; its tree number is allocated for this chain,
stage and purpose rather than copied from another scheme, and every field of
the star plan is bound in the trusted plan.

## Open items

`[GAP]` The joint opening ledger accounts the single-stage basis in the
session harness. The loop moves those openings into the switch and LastBase
artifacts, so the loop's measured opening budget is recorded by the acceptance
run, not by the harness constants.

`[FACT]` The acceptance run measured on the GPT-2 seq16 formal fixture, same
machine, after the loop wiring and the `k=14` stage-0 correction:
hidden prove 744,204 ms, hidden verify 51,142 ms, whole test 1,035.04 s,
hidden proof 9,527,120 bytes and maximum RSS 213,833,272 KiB. The proof
reports 14 own fold rounds. The previous single-stage trace on the same
fixture (scheme `rework10-final`, code `473e395d`) recorded hidden prove
359,802 ms, hidden verify 50,538 ms, maximum RSS 169,394,600 KiB and a
646.80 s test. The loop therefore roughly doubles hidden prove wall time and
raises peak RSS by about 26% on this fixture; the numbers are recorded as a
resource cost of the multi-stage scheme, not as an optimization gain. The
baseline is from a different commit and is cited with its trace.

`[GAP]` The security bound that combines code distance, list size, batching,
hash, work and mask hiding budgets at the production parameters is not
restated here. It is part of the batch review together with the measured
resource budget; small-fixture parameters are not final parameters. The
`positions=2` and `work=0` stage values remain candidate plan values; the
planner-derived proximity positions for a `2^-128` per-stage target are much
larger (shared stage0 155, shared last 129, own stage0 155, own stage1 15,
own last 20).
