# Stage mask capacity and random tail derivation

## Scheme and baseline

- Scheme: `stage-capacity-tail` (second piece of the sizing derivation).
- Code: `4bd2d321`; previous commit `12ac4042`; branch `refactor/v2`.
- Scope: capacity and tail rules only; no session parameter is wired yet.

## What changed

- `proximity_planner.rs`: `stage_tail(t, circle)` follows the domain parity
  rule; `solve_next_stage` cuts the folded row after a divisibility check,
  carries the previous two mask tails and slots plus its own two tails into
  the mask capacity, takes the power-of-two mask length that holds the
  carried entries, the samples and the pad, and reports the remaining
  out-of-domain entries; `solve_last_stage` is the single-row last stage.
- Unit tests: middle stage rules, last stage rules, indivisible cut and the
  parity rule.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Folded row cut with divisibility | stage recursion rule | `solve_next_stage` | unit test |
| Mask capacity carries tails and slots | stage recursion rule | carried sum and power-of-two length | unit test |
| Out-of-domain entries keep the pad | capacity minus carried | `r_ood` | unit test |
| Last stage is one row | last-stage rule | `solve_last_stage` | unit test |
| Random tail follows the domain | tail rule | `stage_tail` | unit test |
| Configuration candidates, stage lists, joint solve, capacity mapping, round budget | next steps | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib stage_sizes`: 2 passed.
- `cargo test --workspace`: 51 suites, 994 passed, 0 failed, 19 ignored.
- Hard gate: 45,080 bytes, sha256 `7a5da6eb…de4cfb`, unchanged.
- No push.

## Boundaries

- Capacity and tail rules only; the code-length feasibility checks, the
  public configuration candidates, the full stage lists, the shared/own joint
  solve, the capacity mapping and the whole-chain round budget are next.
- The production session plan still uses the diagnostic fixture parameters
  with the planner check disabled; fixture constants remain test-only.
- No protocol object, message or verification condition changed; existing
  full-weight checks remain; no plaintext is removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `stage-sizes-tests.txt`, `NOTES.md`.
