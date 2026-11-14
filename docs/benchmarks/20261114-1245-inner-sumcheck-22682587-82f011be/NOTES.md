# Inner-dimension sumcheck of the request reduction

## Scheme

- Scheme: `inner-sumcheck`.
- Current code: `22682587`, following `82f011be`.
- Previous code: `82f011be`.

## Purpose

The request reduction's second dimension: the claim is the equality-weighted
product of the two operand maps at an inner point, the sumcheck runs one
equality weight per inner bit, and the weighted product inner carries the
later variables' equality factors while the verifier weights the bound
variable. Focused tests compute the claim by hand, prove and verify it, and
refuse a pair whose product sum does not match.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_reduction`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 4 passed |
| workspace | 51 suites, 1038 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The inner claim is the equality-weighted product of the two operand maps | inner sumcheck claim rule | `inner_claim` | focused test computes the claim by hand |
| One equality weight per inner bit, in bit order | equality round form | `inner_variables` | focused test; prove and verify agree |
| The inner rounds carry the later equality factors | weighted product inner-sumcheck rule | `prove_inner`/`verify_inner` with the weighted product inner | focused test; a mismatched pair is refused |

## Conclusions

- The inner-dimension sumcheck is in place and tested.
- No session or proof message is touched; the toy proof is unchanged.

## Open items

- The stack rounds, the affine row claims (`rowS`, `A0`, `B0`) and the weight
  claims follow.
- The session transcript wiring waits for those pieces.
