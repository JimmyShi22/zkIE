# Row claims of the masked sumchecks

## Scheme

- Scheme: `row-claim`.
- Current code: `9acd6af9`, following `81f14407`.
- Previous code: `81f14407`.

## Purpose

Every masked sumcheck leaves one linear claim on the masks: the hiding form's
coefficients on the sub-proof's hiding entries, the claim slots' coefficients,
the kept maps' coefficients, against the row target. This block keeps that
shape in one place and evaluates it; the start form now reuses the same slot
term instead of its own. A focused test runs a weight merge, builds the row
claim from the verified output and checks it on the proof's entries.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib row_claim` and
  `cargo test -p zkie-ops --lib request_start`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused row-claim tests | 1 passed |
| focused start-form tests | 3 passed |
| workspace | 51 suites, 1050 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The output's row claim is the hiding entries, the claim slots and the kept values against the row target | row claim shape | `RowForm::from_output` | focused test builds it from a merge output |
| The claim is evaluated on entries the caller supplies | row evaluation rule | `RowForm::evaluate`, `RowForm::check` | focused test; another hiding entry or mask value is refused |
| One slot term serves the start form and the row claim | single implementation | `SlotTerm` moved to the row claim module | start-form focused tests stay green |

## Conclusions

- The row claim shape is in one place and tested on a real merge output.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The weight fork's leftover claim and its chain test, then the pending row and
  the session transcript wiring, follow.
