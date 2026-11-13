# Clamp sign-consistency slack slots (carrier)

- Scheme: `norm-output-tail-clamp-slacks`
- Change: code commit `86e28e41` (four per-column slacks of the clamp
  comparison in the clamp widths and slot layout).
- Baseline: previous commit `600fca94` (clamp value and exclusivity
  relations).
- Purpose: carry the committed slacks the selector sign-consistency relations
  consume.

## What changed

- `NormClampWidths` gains the strict and non-strict slack widths of both
  bounds; the tail slots and message width include them.
- Values append the four slacks (zero on the inactive side): strict
  `low - pre - 1`, non-strict `pre - low`, strict `pre - high - 1`,
  non-strict `high - pre`.
- Widths are capacity-checked against the pre-clamp span and the farthest
  bound; the plan encoding absorbs the new widths.
- The tail group grows to 256 columns on the fixture.

## Evidence

- Plan and chain tests passed at this commit (focused filters); the full
  workspace and hard-gate numbers recorded in the following trace
  `20261113-1707-norm-output-tail-clamp-consistency-51024736-86e28e41`
  cover the combined state, because the slacks are inert until the
  consistency relations consume them.

## Four-column mapping

| Requirement (TURN 504 一) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| committed slack slots | four slack digit rows per column | plan/chain tests | the range lookup is next |
| widths from the trusted plan | capacity against the pre-clamp span and bounds | plan rejection cases | - |
| layout version | plan encoding absorbs the widths | plan digest | - |

## Open items

- The selector sign-consistency relations and the slack digit range lookup;
  padding; hd.
- No push in this round.
