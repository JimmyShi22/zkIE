# The own group's request W member

## Scheme

- Scheme: `request-w-member`.
- Current code: `2f36ea80`, following `30b0ceb2`.
- Previous code: `30b0ceb2`.

## Purpose

The own group's member list now takes the request W's member when the plan, the
first message and the witness carry it: its source oracle, code, row and answer
counts come from the plan, its root and tree from the committed entry (checked
against the message), and its message and tails from the witness. The sampling
gives it its own sample rows from the message's sample, one per drawn seed.
Existing plans carry none, so every current member list is unchanged.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: the workspace's hidden-session tests.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| workspace | 51 suites, 1078 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The request W's member joins the group with its committed identity | member rule | `own_members` request W block | the workspace builds and passes |
| Its sample rows come from its message's sample | sampling rule | `own_sample_rows` branch | one row per drawn seed |
| Existing member lists are unchanged | stability rule | gated on the plan's field | hard gate unchanged |

## Conclusions

- The request W has a member with its sampling path.
- No existing path changes; the toy proof is byte-identical.

## Open items

- The callers pass the request rows bundle; a member-carrying fixture exercises
  the positive and negative runs; then the shared and second tests run.
