# Joint draw counts of the shared and own groups

## Scheme and baseline

- Scheme: `joint-draws` (first piece of the joint shared/own solve).
- Code: `43b0aa4c`; previous commit `634e02a4`; branch `refactor/v2`.

## What changed

- `proximity_planner.rs`: `joint_draws` accounts the positions the shared
  group's codes and a request's codes must cover, with Omega read by the
  shared test and by every request's test, the split case reading Omega's own
  message in both tests, and the batch term multiplying only the request's
  positions. A single request fixes the public batch size at one.
- Unit test: joined group, split group, batch of two and a zero batch.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Omega read by shared and every request | joint draw rule | `joint_draws` | unit test |
| Split reads both sides | split draw rule | folded/omega mask draws | unit test |
| J=1 is requests=1, not a missing term | public batch configuration | `requests >= 1` enforced | unit test |
| Position/distance fixed point over the counts | next step | **not yet** | gap |
| Capacity mapping, whole-chain round budget | next steps | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib joint_draw`: 1 passed.
- `cargo test --workspace`: 51 suites, 996 passed, 0 failed, 19 ignored.
- Hard gate: 45,080 bytes, sha256 `7a5da6eb…de4cfb`, unchanged.
- No push.

## Boundaries

- Only the draw accounting: the ascending position/tail fixed point over
  these counts (which produces the first stage's positions, distance and
  random tails), the capacity mapping and the whole-chain round budget are
  not implemented.
- The production session still uses the diagnostic fixture parameters; the
  planner check stays disabled; fixture constants remain test-only.
- No protocol object, message or verification condition changed; all existing
  full-weight checks remain; no plaintext is removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `joint-draw-tests.txt`, `NOTES.md`.
