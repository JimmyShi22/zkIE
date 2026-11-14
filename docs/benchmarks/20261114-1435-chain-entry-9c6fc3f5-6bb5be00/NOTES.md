# Request chain prove and verify entries

## Scheme

- Scheme: `chain-entry`.
- Current code: `9c6fc3f5`, following `6bb5be00`.
- Previous code: `6bb5be00`.

## Purpose

The request chain now proves and verifies end to end: the kept token sumcheck,
the batched stack whose start absorbs the token claim, and the weight merge,
in transcript order. The prover checks its own row claims before handing them
on; the verifier checks the stack's and the merge's row claims on the
transcript's mask entries and leaves the pending claim built from the merge's
row weights, point, end value and end mask factor. The token's kept map stays
out of the messages and is solved into the start.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_chain`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused chain tests | 3 passed |
| workspace | 51 suites, 1056 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The three sumchecks run in transcript order with one challenge stream | chain order rule | `prove_request_chain`, `verify_request_chain`, `Recorder` | focused test runs the whole chain |
| The token claim is solved into the start and a changed token proof breaks the stack | start rule | `start_form`, `start_terms` | focused test refuses a changed token proof |
| The stack's and merge's row claims are checked on the mask entries | row claim check rule | `RowForm::check` in prove and verify | focused test; a changed merge proof is refused by its row |
| The verifier leaves the pending claim of the merge | pending rule | `pending_row` over row weights, point, end value and end mask factor | focused test checks it on a matrix; another matrix or mask value is refused |

## Conclusions

- The chain's arithmetic entry is in place and tested end to end.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The session entry, the stack's three linear claims (`rowS`, `sA0-Alin`,
  `sB0-Blin`) and the source-layer reads follow.
