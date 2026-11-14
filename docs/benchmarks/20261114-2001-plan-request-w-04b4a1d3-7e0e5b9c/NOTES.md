# The request W's oracle in the plan

## Scheme

- Scheme: `plan-request-w`.
- Current code: `04b4a1d3`, following `7e0e5b9c`.
- Previous code: `7e0e5b9c`.

## Purpose

The plan now names the request-level W's SEND oracle: its message shape, its
oracle number, its group and the padded column count the vector holds. Validation
refuses an oracle inside the per-position range and a vector length that is not
the padded column count, so the request W cannot silently reuse a per-position
identity or a shorter vector.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_plan_names_the_request_w_oracle`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused plan test | 1 passed |
| workspace | 51 suites, 1075 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The plan names the request W's shape, oracle, group and column count | plan layout rule | `RequestWPlan`, the plan field | focused test validates a plan carrying it |
| The oracle may not sit in the per-position range | identity rule | oracle range check | focused test |
| The vector length is the padded column count and a power of two | layout rule | column check | focused test refuses half the columns |
| Existing plans are unchanged | stability rule | empty value in every literal | hard gate unchanged |

## Conclusions

- The request W has its plan identity with its boundaries checked.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The first part commits the request W into the message, the own group takes its
  member, and the two request rows join it; then the shared/second tests run.
