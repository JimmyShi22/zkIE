# The own group takes the request rows

## Scheme

- Scheme: `request-rows-hook`.
- Current code: `66f06b42`, following `2e65871c`.
- Previous code: `2e65871c`.

## Purpose

The own-group row builder can now take the request's two LIN rows: the evidence
row over the request W member and the dot row over Omega and W, bundled with
their claim data. Every existing caller passes none, so the current row lists
are unchanged; a focused test builds the bundle and sees the two rows with their
targets and terms.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_own_group_can_take_the_request_rows`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused hook test | 1 passed |
| workspace | 51 suites, 1078 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The own-group builder can take the request's two rows | group rule | `RequestRows`, the optional parameter | focused test builds both rows |
| Existing callers' row lists are unchanged | stability rule | every caller passes none | workspace passes; hard gate unchanged |
| The rows carry their targets and terms | row rule | `request_rows` | focused test |

## Conclusions

- The insertion point for the request rows exists behind an optional bundle.
- No existing path changes; the toy proof is byte-identical.

## Open items

- The own group's member list takes the request W's member, the callers pass
  the bundle, then the shared and second tests run.
