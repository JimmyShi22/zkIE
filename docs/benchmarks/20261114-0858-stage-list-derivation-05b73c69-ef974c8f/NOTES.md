# Configured stage list derivation with failure points

## Scheme and baseline

- Scheme: `stage-list-derivation` (third piece of the sizing derivation).
- Code: `05b73c69`; previous commit `ef974c8f`; branch `refactor/v2`.

## What changed

- `proximity_planner.rs`: public `StageChoice` and `FirstStage`
  configuration types; `derive_stage_list` derives the first stage plus one
  stage per choice, enforcing the last-stage and middle-stage shape rules,
  the stage capacity and random-tail rules and the code-length capacity
  (message plus tail within the code length). A rejected configuration
  reports its stage and the failing point (stage shape, indivisible cut,
  capacity).
- Unit tests: a valid chain, both stage-shape rules, an indivisible cut and a
  code length that does not fit.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Public configuration candidates | first stage plus per-stage choices | `FirstStage`/`StageChoice` | unit test |
| Last stage single row, middle stages 1<=k, h<=k | stage shape rule | validation with stage and point | unit test |
| Capacities and tails per stage | stage rules | `derive_stage_list` | unit test |
| Code-length feasibility per stage | message and tail within the code length | `code_fits` | unit test |
| Rejected configuration names its failing point | failure struct | `StageFailure` | unit test |
| Shared/own joint solve, capacity mapping, round budget | next steps | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib stage_list`: 1 passed (plus the two existing
  stage-list tests of the session module).
- `cargo test --workspace`: 51 suites, 995 passed, 0 failed, 19 ignored.
- Hard gate: 45,080 bytes, sha256 `7a5da6eb…de4cfb`, unchanged.
- No push.

## Boundaries

- The derivation covers one chain's stage list from a given first stage and
  public choices. The first stage itself (group bits, positions, distance,
  random tails) comes from the group layout and the joint solve, which are
  the next steps; the shared and own groups are not yet solved together.
- The capacity mapping and the whole-chain round budget are not implemented.
- The production session still uses the diagnostic fixture parameters; the
  planner check stays disabled. Fixture constants remain test-only.
- No protocol object, message or verification condition changed; all existing
  full-weight checks remain; no plaintext is removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `stage-list-tests.txt`, `NOTES.md`.
