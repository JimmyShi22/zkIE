# gu-combine r2 - 4d909969 / 857e8fcd

Trace of the G_u declaration combination routed through the existing BAT layer.

## Change

Code commit `4d909969` ("Route the G_u declaration combination through the
existing BAT"), based on `857e8fcd`. The production verify entry is unchanged.

- The custom x-power combination and declaration-absorb challenge path is
  removed. G_u declarations are adapted to `BatRow` claims and combined by the
  existing `BatStatement`/`BatRunInput`/`BatSession`/`BatGroupRow` flow.
- The BAT group is frozen with `begin_run`; the layout, fixed class order and
  planned row count come from BAT.
- The G_u oracle/group ids are mapped to BAT's sequential registration ids; the
  original ids are retained in the outcome.
- The mask term is zero-padded to a power-of-two BAT width. The original G_u
  mask code has six entries; BAT requires power-of-two widths, so the adapter
  uses an eight-entry BAT mask code with an empty tail (8 + 1 would exceed the
  code length). The padded coefficients are zero, so the linear functional is
  preserved. This is a documented adapter, not a second BAT.
- The chain snapshot wording is corrected to a **same-prefix forked isolated
  snapshot**; the original FOLDEQ session is not advanced by BAT.
- The BAT round is marked candidate-only: not an exact reference transcript,
  not a permanent proof format.

The previous 3d package `t002-s6b3d-review.tar.gz` sha256
`d75adfdbb4f2e9f3e4e61714ebd8301f2c88781227894a72a0fe7c457bcde76a` is void and
retained unchanged.

## BAT reuse and adapter

| item | adapter |
| --- | --- |
| claim | `GuRelationRow` → `BatRow` with a main term and a mask term |
| sample | BAT `Sample` class; the session answer is a fake value |
| record/column | BAT `Plain` class |
| oracle/group ids | G_u 11/5 mapped to BAT 0/0; originals retained |
| main term | G_u main code, 16 coefficients |
| mask term | padded from 6 to BAT width 8 with zero coefficients and empty tail |
| target | unchanged declaration target |
| freeze | `BatRunInput` via `begin_run` |
| order | BAT fixed class order: plain rows, then sample rows |
| count | `planned_rows` checked by `BatSession::finish_run` |
| combination | BAT `combine`, one `BatGroupRow` |

The custom `combine_declarations` and `draw_combine_challenge` functions are
gone. No second arithmetic path remains.

## Snapshot and transcript

- Cloning `FoldeqSession::chain()` and running BAT on the clone is a same-prefix
  forked isolated snapshot. The original session has not advanced through the
  BAT round and this is not a later FOLD-ready state of that session.
- Future linkage must be one continuous session: fix the declarations in the
  confirmed order, then draw the challenge. Copying state must not create two
  independent protocol paths.
- BAT closes the empty algebraic round and draws `x`; declarations are frozen by
  `BatRunInput` and counted, not re-absorbed as an extra hash layer.
- Binding basis: frozen `BatRunInput` (layout, member order, class order and
  row count). Challenge domain: BAT empty algebraic close. Declaration order:
  BAT fixed class order. No extra protocol round or permanent format is added.
- This is a candidate demonstration, not an exact reference transcript.

## G_u and BAT semantics

- The G_u root is self-reported and the SEND answer is fake; both are flagged.
- Column openings authenticate only against the column root.
- The G_u row is a pending declaration; BAT combines and does not judge truth.
- A wrong declaration on the valid path still combines and remains pending; no
  later proximity/relation check is connected, so this is not a check pass.

## 3c connection boundary

The combined row is not connected to the 3c FOLD source in this step. The 3c
source has its own fixture claim/target relation; rewriting its target to match
this combination would not establish real consistency. `three_c_connected` is
false; the 3b projection `handoff_statement` remains rejected. Fresh, old-mask
and random tails stay on their existing fixture/fail-closed boundaries.

## Tests and measurements

- G_u combine unit: 4 passed.
- GroupRow unit: 6 passed.
- Fold projection unit: 4 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 G_u combination through BAT: 1 passed (56.94 s test, 57.08 s
  wall, VmHWM 4,241,260 kB; combination session 371 ms).
- Workspace: 453 passed, 0 failed, 10 ignored.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

Real session values:

| item | value |
| --- | --- |
| reader root | `83a62c3995b33823d29b5465f7a271d69b2cc9a4f73d4c403ee9e96c01f623b3` |
| G_u root | `4747474747474747474747474747474747474747474747474747474747474747` |
| G_u oracle/group/tree | `11` / `5` / `7` |
| queries | `[4, 7, 7, 4]` |
| sample / record / column declarations | `1` / `1` / `4` |
| declaration names | sample, record, column, column, column, column |
| BAT order | plain, plain, plain, plain, plain, sample |
| BAT group/oracle | `0` / `0` |
| combined main/mask lengths | `16` / `8` |
| challenge x | `Fp3([12392023129221551483, 2908739786673047002, 1072783450561979053])` |
| combined target | `Fp3([2699838696867800944, 16244595859463608400, 12804667154672950552])` |

The roots are local/session demonstration values, not authentic G_u content.
The full asset opening verification and table build are included in the test
cost; this is a single `gpt2-seq16` case and is not a production performance or
memory measurement.

## Negative coverage

| change | layer | observed result |
| --- | --- | --- |
| declaration main/mask shape differs | BAT adapter | deterministic rejection |
| declaration target wrong on the valid path | BAT combination | still pending; combined target changes, no truth decision |
| declaration values changed | BAT combination | same x (BAT close does not absorb declarations); different combined target |
| group shape/width cannot be adapted | BAT adapter | deterministic rejection |
| second run with the same session data | deterministic rerun | same transcript, x and combined row |

The wrong-target case is deliberately not a rejection: BAT does not judge truth.
No random check is written as a must-reject condition; the demonstration uses
fixed session data and the test-only fake SEND answer.

## Validation scope

Only `gpt2-seq16` is used. The session is the same isolated column session used
by the earlier components; the G_u root and SEND answer are demonstration
values. The three seq16 cases beyond the dependency case were not run because
this step does not change the shared three-stage path or seq512 logic. No
five-case three-stage run was made.

## Unclosed items

- Real G_u authenticity/source and a real SEND provider.
- A later proximity/relation check for the pending declarations and the
  combined row.
- A valid 3c FOLD-to-CS/BASE handoff; `handoff_statement` still rejects.
- Fresh-mask handoff and trusted fresh authentication.
- Folding-tail derivation; random tails did not participate in FOLD folding.
- One continuous session for the BAT round; this step uses a forked snapshot.
- Hiding and single-shot semantics.
- Production wiring: `verify_request` unchanged; this is not an accepted proof
  format.
- Multi-oracle SEND and any declaration not listed above.

## Artifacts

- `tests/gu-combine-unit.txt`
- `tests/group-row-unit.txt`
- `tests/fold-handoff-unit.txt`
- `tests/proximity-session-unit.txt`
- `tests/gu-combine-real.txt` (local fixture path replaced with `<fixture>`)
- `tests/regression-foldeq.txt`
- `tests/regression-column.txt`
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The earlier 3c r2 package remains retained unchanged. This is a new frozen
evidence set; the previous 3d package is void.
