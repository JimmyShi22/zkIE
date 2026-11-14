# Joint first-stage fixed point of the shared and own groups

## Scheme and baseline

- Scheme: `joint-first-stage` (fixed-point block of the sizing derivation).
- Code: `42701762`; previous commit `ea26d7b7`; branch `refactor/v2`.

## What changed

- `proximity_planner.rs`: the group analysis can now run without mask rows
  (the main-only bound); the existing two-stage path always enables them and
  keeps its behavior. New `with_tail`/`group_with_tails` set each code's
  random tail under the code-length check; `positions_from_below` sizes the
  main code alone on the way up; `solve_first_stage` iterates the shared and
  own groups upward on the joint draw counts (Omega read by the shared test
  and every request, split reading both sides), recomputing tails and sizes
  each round, and reports `StageFailure{stage,point}` for capacity, distance
  and non-settling cases.
- Unit tests: a converged first stage (positions 47/47, Omega 0 for the
  non-split group, tails from the joint draws), a capacity failure and a zero
  batch; the planner's existing tests stay green.

## Mapping (reference rule to this module)

- Per-group sizing: reference size lambda -> `sized_step`.
- Main-only up-bound: reference positions_from_below -> `positions_from_below`.
- Position count: reference plan_positions/scope_positions -> our
  `scope_positions(agree, 0, margin)` with `TARGET_BITS - work_bits`
  corresponding to the target minus the query work and `instances = 1`.
- Margin: reference 2^-(target+30) -> `margin_bound`.
- Load: reference GroupLoad{cube_bits, messages=2, batched_rows} -> same
  fields; batched rows are the ruled per-group values (columns+1+3*samples for
  the folded row, 1+2*samples for Omega, 2+3*samples for the request group).

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Upward position/tail fixed point | per-group size rule; joint draw counts | `solve_first_stage` | unit test |
| Main-only bound on the way up | positions_from_below | no-mask group analysis | unit test |
| Tail recomputed each round | tail rule with draws | `group_with_tails` | unit test |
| Failure points, no generic rejection | StageFailure{stage,point} | capacity/distance/settle points | unit test |
| Capacity mapping, whole-chain round budget | next steps | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib proximity_planner`: 12 passed (including the
  three new first-stage tests and the existing planner tests).
- `cargo test --workspace`: 51 suites, 999 passed, 0 failed, 19 ignored.
- Hard gate: 45,080 bytes, sha256 `7a5da6eb…de4cfb`, unchanged; the two-stage
  path's plan behavior is unchanged (its tests pass).
- No push.

## Boundaries

- The fixed point produces the first stage's positions and random tails for a
  given group configuration. The capacity mapping (spent/remaining lists,
  hiding conditions) and the whole-chain round budget are the next steps.
- The production session still uses the diagnostic fixture parameters; the
  planner check stays disabled; fixture constants remain test-only.
- No protocol object, message or verification condition changed; all existing
  full-weight checks remain; no plaintext is removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `planner-tests.txt`, `NOTES.md`.
