# The plan's member claim is the chain's claim

## Scheme

- Scheme: `member-claim`.
- Current code: `cf29e69f`, following `30145541`.
- Previous code: `30145541`.

## Purpose

The plan's member claim must be the claim the request chain derived from its
verified prefix, not a second source. This block adds the check that the two
are the same object: the same row weights, the same column point, the same
target and the same mask terms, in the same order. A plan claim that differs
in any part is refused instead of being trusted.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_plan_claim_must_be_the_chain_claim`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused member test | 1 passed |
| workspace | 51 suites, 1067 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The plan's member claim is the chain's verified claim | single-source rule | `check_member_claim` | focused test accepts the identical claim |
| Another weight row, point, target or mask coefficient is refused | source rule | term-by-term comparison | focused test |
| No second claim source | input source rule | equality rather than trust | the plan claim must equal the derived one |

## Conclusions

- The member claim has a single source: the chain's verified prefix.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- Wiring the claim into the hidden LIN first and second parts, the at_start
  ledger and the request-level parameters follows.
