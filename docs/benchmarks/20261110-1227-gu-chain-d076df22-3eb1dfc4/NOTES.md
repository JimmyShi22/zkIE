# gu-chain - d076df22 / 3eb1dfc4

Trace of the single-continuous-session chain handoff for the G_u declaration
and BAT demonstration.

## Change

Code commit `d076df22` ("Transfer chain ownership across the FOLD, BAT and next
stages"), based on `3eb1dfc4`. The production verify entry is unchanged.

- `FoldeqSession::take_chain_for_bat` consumes the session after the declaration
  joins and returns the chain. A missing join or an earlier stage is rejected.
  The old session handle cannot advance the chain independently.
- `BatSession::into_chain` consumes the completed BAT session and returns the
  advanced chain. A transfer before a completed run is rejected.
- The G_u combination now uses this ownership transfer: FOLDEQ → BAT → returned
  chain. It is one continuous state handoff, not a copied digest and not a
  forked snapshot.
- The BAT frozen input summary records declaration class order, targets, count,
  oracle/group mapping and table lengths.
- The six-to-eight mask adapter is marked adapter-only; its result is not
  connected to any real FOLD/proximity claim.

All earlier packages remain retained unchanged.

## Chain ownership

| step | ownership behavior |
| --- | --- |
| FOLDEQ declaration joins complete | `take_chain_for_bat(self)` consumes the FOLDEQ session and returns its chain |
| BAT grouping and combine | `BatSession` owns the transferred chain and advances it through the BAT close and `x` draw |
| BAT complete | `into_chain(self)` consumes the BAT session and returns the advanced chain |
| wrong stage / missing join | `InvalidSessionState` |
| BAT transfer before run | `InvalidSessionState` |
| reuse of either old handle | type error (the handles were consumed) |

The tests use the ownership transfers; they do not prove continuity by
comparing two summaries. The engine test `bat_chain_handoff_requires_the_read_stage`
checks the FOLDEQ stage boundary, and `bat_chain_transfer_requires_a_completed_run`
checks the BAT completion boundary.

## Declaration fixing

The verifier takes the declarations from the same session's already-bound table
root/record, G_u message, challenges and authenticated columns. The BAT group is
frozen by `begin_run`; a later replacement is rejected. Only then is `x` drawn.
No arbitrary external declaration is hashed to claim a trusted source, and no
extra declaration-absorption round is added: the declarations depend on the
already-fixed inputs and are frozen by `BatRunInput`.

## BAT frozen input

The outcome records:

- BAT class order: plain, plain, plain, plain, plain, sample;
- declaration count and planned row count;
- BAT oracle/group mapping `0/0` (original G_u ids `11/5` retained);
- per-row targets in frozen order;
- main/mask table lengths `16/8`.

The G_u declaration order remains sample, record, column, column, column,
column; BAT's fixed class order places the plain rows first and the sample row
last. Both are recorded.

## Mask adapter isolation

The G_u mask code has six entries; BAT requires power-of-two widths. The adapter
zero-pads the BAT-only mask term to eight entries and uses an empty tail because
8 + 1 exceeds that code. The padded coefficients are zero and the declaration
target is unchanged. This adapter is for the local algebraic demonstration only:
`adapter_only` is true and `downstream_fold_connected` is false. A result that
depends on the adapter must not flow into a path claiming a real FOLD or
proximity result.

## G_u and BAT semantics

- The G_u root is self-reported and the SEND answer is fake; both are flagged.
- Column openings authenticate only against the column root.
- The G_u row is a pending declaration; BAT combines and does not judge truth.
- A wrong declaration on the valid path still combines and remains pending; no
  later proximity/relation check is connected.

## Negative coverage

| change | layer | observed result |
| --- | --- | --- |
| declaration main/mask shape differs | adapter | `ShapeMismatch` |
| declaration group/oracle differs | adapter source check | `StatementMismatch` |
| group closed, then replacement row | BAT freeze | `InvalidSessionState` |
| BAT transfer before completed run | BAT ownership | `InvalidSessionState` |
| FOLDEQ transfer before read stage | FOLDEQ stage check | `InvalidSessionState` |
| declaration target wrong on the valid path | BAT combination | pending; combined target changes, no truth decision |
| declaration values changed | BAT combination | same `x`; different combined target |
| bound challenge prefix changed | fixed-sample propagation | different `x` for that fixed sample; not a general collision/domain claim |
| same opening rebuilt independently | data-flow assertion | same reader root and table identity |

## Tests and measurements

- G_u chain unit: 7 passed.
- GroupRow unit: 6 passed.
- Fold projection unit: 4 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 39, column 9, edge 2.
- Real gpt2-seq16 chain handoff: 1 passed (56.43 s test, 57.22 s wall, VmHWM
  4,240,984 kB; combination session 368 ms).
- Workspace: 457 passed, 0 failed, 10 ignored.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

Real session values:

| item | value |
| --- | --- |
| reader root | `83a62c3995b33823d29b5465f7a271d69b2cc9a4f73d4c403ee9e96c01f623b3` |
| G_u root | `4747474747474747474747474747474747474747474747474747474747474747` |
| G_u oracle/group/tree | `11` / `5` / `7` |
| queries | `[4, 7, 7, 4]` |
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
- A real FOLD/proximity connection for the adapter-only result.
- A valid 3c FOLD-to-CS/BASE handoff; `handoff_statement` still rejects.
- Fresh-mask handoff and trusted fresh authentication.
- Folding-tail derivation; random tails did not participate in FOLD folding.
- Hiding and single-shot semantics.
- Production wiring: `verify_request` unchanged; this is not an accepted proof
  format.
- Multi-oracle SEND and any declaration not listed above.

## Artifacts

- `tests/gu-chain-unit.txt`
- `tests/group-row-unit.txt`
- `tests/fold-handoff-unit.txt`
- `tests/proximity-session-unit.txt`
- `tests/gu-chain-real.txt` (local fixture path replaced with `<fixture>`)
- `tests/regression-foldeq.txt`
- `tests/regression-column.txt`
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The earlier 3d r2 and 3c r2 packages remain retained unchanged. This is a new
frozen evidence set; old packages are not deleted.
