# Start form of the request reduction's stack

## Scheme

- Scheme: `token-start`.
- Current code: `81f14407`, following `4e2a1e28`.
- Previous code: `4e2a1e28`.

## Purpose

The kept token sumcheck leaves one linear claim. The stack's start absorbs it:
the claim's slot is subtracted, and the kept map's value at the token point,
solved from the token row, is added with the weight batching coefficient. The
form is kept as one constant, one coefficient per hiding entry of the token
sumcheck and one coefficient per index-mask slot, so a masked sumcheck can take
it as its target and claim masks. The solve divides by the challenge and the
shown end value, and both must be nonzero.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_start`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 3 passed |
| workspace | 51 suites, 1049 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The start subtracts the token claim's slot and adds the kept value with the weight batching coefficient | start form rule | `token_start`, `StartForm` | focused test rebuilds the batched kept value from the form and compares it with the row's solve |
| The kept value is the token row solved for it | row solve rule | row target, hiding form and claim slots divided by the challenge and the shown end value | focused test solves the row and matches the form |
| The form drives a masked sumcheck's target and claim masks | start-to-stack rule | start terms as target and claim masks | focused integration test proves and verifies a stack sumcheck; another panel value is refused |
| A zero challenge or a zero shown end value is refused | boundary | `token_start` checks | focused negative test |

## Conclusions

- The start form is in place and tested, including its use as a stack target.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The weight fork, the merge's leftover claim and the pending row enter the
  session transcript next.
- The stack's three linear claims and the source-layer reads follow.
