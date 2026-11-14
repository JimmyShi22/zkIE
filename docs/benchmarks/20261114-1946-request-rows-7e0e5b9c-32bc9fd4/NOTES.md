# The request's evidence and dot rows

## Scheme

- Scheme: `request-rows`.
- Current code: `7e0e5b9c`, following `32bc9fd4`.
- Previous code: `32bc9fd4`.

## Purpose

The request's two LIN rows now have their builders. The evidence row places the
request-level W at the claim's column point on the W message's own layout and
adds the claim's mask slots against the claim's target. The dot row places the
row weights on Omega and the W at the shared point, against zero. Both use the
message-layout scatter the leaf rows already use, so coefficients land where
the committed messages hold them, and a point wider than the vector is refused.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_request_rows_use_the_message_layouts`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused request-rows test | 1 passed |
| workspace | 51 suites, 1074 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The evidence row ties W at the claim's point plus the mask slots to the target | (Eval) rule | `request_eval_row` | focused test compares the term with the point's scatter |
| The dot row ties the weights on Omega and W at the shared point to zero | (Dot) rule | `request_dot_row` | focused test compares both terms |
| Coefficients use the messages' own layouts | layout rule | `scatter_message_coefficients` | focused test |
| A point wider than the vector is refused | boundary rule | width check | focused test |

## Conclusions

- The two request rows are built on the committed messages' layouts.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- Add the rows to the first and second groups, run the shared and second IOPP
  tests, then freeze the layout and deliver the batch report.
