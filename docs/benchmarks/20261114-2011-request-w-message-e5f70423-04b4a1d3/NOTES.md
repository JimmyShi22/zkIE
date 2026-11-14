# The request W's root in the first message

## Scheme

- Scheme: `request-w-message`.
- Current code: `e5f70423`, following `04b4a1d3`.
- Previous code: `04b4a1d3`.

## Purpose

The request-level W's committed root travels in the session message. The plan
names its oracle, shape and group; the message carries the root; decoding reads
it only when the plan carries it, so every existing plan encodes exactly as
before and a plan/message mismatch is refused by the shifted parse.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_request_w_root_travels_in_the_message`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused message test | 1 passed |
| workspace | 51 suites, 1076 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The request W's root travels in the session message | message field rule | `RequestWMessage`, the message field | focused test round-trips it |
| Decoding reads it only when the plan carries it | version-gated rule | plan-gated decode | focused test refuses a plan/message mismatch |
| Existing plans encode exactly as before | byte stability rule | no bytes when absent | hard gate unchanged |

## Conclusions

- The request W's root has its plan-gated message section.
- No proof path produces the root yet; the first part commits it next.

## Open items

- The first part commits the request W over the table and absorbs the root; the
  own group takes its member and the two request rows join it.
