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

`[FACT]` The own group now uses the interval-derived two-stage candidate:
stage 0 keeps the frozen `(17,65,536,8,192)/(17,512,256)` codes with
`k=14,h=2`, the trusted planner chooses `positions=252` and 24 work bits by
the conservative search under the mask-tail cap 254 (Johnson m = 616), and
the last stage is `(17,65,536,259)/(17,16,384,259)` with `k=h=0`,
`positions=259` and Johnson m = 595. The planner lives in
`crates/zkie-ops/src/proof/proximity_planner.rs`; the verifier re-derives the
values and `HiddenProximityPlan::fault` rejects a mismatch before the first
query. The same-machine comparison with the same-parameter three-stage chain
is recorded in the trace notes; the interval run is the current candidate.
The planner uses outward-rounded interval arithmetic (`proof/interval.rs`)
and no floating-point decision. The naive sum of all counted round terms is
`2^-123.69`, while the worst single round is `2^-128.000035`; the batch does
not claim the sum meets 128 bits (see the rework18 evidence note).

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

## Opening ledger

`[FACT]` The formal harness books each loop tree from the trusted plan before
proof work: one `OracleKind::Cs` key per switch tree, one `OracleKind::Fold`
key per stage-FOLD mask tree, and `Star`/`StarMask` keys for the plan-derived
BASE star tree. Each key is reserved with the plan's logical draws and tail
rank. The switch member and fresh openings are recorded against those keys
after the prover returns; the LastBase verifier exposes the drawn base
positions, and the harness records the last switch tree and both star trees at
those positions after verification. Recording requires the request to be in
the reserved state, enforces the per-tree union bound and fails closed on
excess, reuse or recovery. In the GPT-2 seq16 two-stage run the ledger reports
139 events / 18,267 logical / 18,265 unique positions before LastBase and 145
events / 19,050 logical / 19,042 unique after it; the total tail rank is
558,595, the minimum remaining is 0 and two oracles are exactly exhausted (not
over). The raw totals are in the trace notes.

The switch sample answer is a single evaluated row carrying the switch pad,
not a tree-opening vector: the canonical switch record carries no opening
positions for it, so it is not a position-union event of the ledger. Its
transcript value and pad are part of the public view covered by the switch
argument.

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

`[FACT]` The joint opening ledger includes the loop switch trees, the
stage-FOLD mask trees and the plan-derived BASE star trees, and records their
actual member, fresh and LastBase openings under the plan keys. The single
non-union event is the switch sample answer described above. See the evidence
note for the measured totals.


`[FACT]` The rework18 interval run on the GPT-2 seq16 formal fixture measured
hidden prove 389,493 ms, hidden verify 64,434 ms, proof 65,432,832 bytes,
maximum RSS 169,342,676 KiB and whole test 1,197.38 s, with 14 own fold
rounds. The earlier 254-position candidate measured hidden prove 372,116 ms,
hidden verify 64,736 ms, proof 66,135,784 bytes, maximum RSS 169,961,812 KiB
and whole test 674.20 s. The Q3 ledger rerun measured hidden prove
377,899 ms, hidden verify 64,966 ms, the same 66,135,784-byte proof and
169,442,488 KiB maximum RSS; its 1,192.65 s test wall includes the enlarged
outer negative matrix. The same-parameter three-stage comparison
measured hidden prove 768,897 ms, hidden verify 74,595 ms, proof 212,523,920
bytes, maximum RSS 213,916,484 KiB and whole test 1,667.56 s (the wall
includes the outer normal/negative matrix). The earlier
positions=2 three-stage run (prove 736,296-744,204 ms, proof 9,527,120 bytes)
is a different security setting and is not compared as a cost baseline. The
numbers are recorded as a resource cost of the safe-parameter loop, not as an
optimization gain; the trace directories carry the raw logs.

`[GAP]` The security bound that combines code distance, list size, batching,
hash, work and mask hiding budgets at the production parameters is not
restated here. It is part of the batch review together with the measured
resource budget. The two-stage candidate uses planner-derived stage-0
positions/work 252/24 and last positions/r 259/259; those are bound in the
trusted plan but are not final parameters. The interval arithmetic and the
largest-m search are implemented; the evidence note records the per-round
table and the union-sum caveat.
