# Kept-map token sumcheck of the request reduction

## Scheme

- Scheme: `kept-token`.
- Current code: `4e2a1e28`, following `68dd9f23`.
- Previous code: `68dd9f23`.

## Purpose

The upstream token sumcheck of a request reduction is affine in maps whose
values are not shown: one factor of each term is a shown map, sent masked as an
end value, while the other factor stays a kept map. This block adds that
summand: the shown end value is the only one sent, the kept map's value stays
out of the messages, and the output row claim carries it with the coefficient
of the shown value as it was sent. The last round includes the kept maps'
contributions through their coefficients, so the round reflects the whole
summand while only the shown value is bound by the end claim.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib psc_general` and
  `cargo test -p zkie-ops --lib request_reduction`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused general-sumcheck tests | 10 passed |
| focused reduction tests | 5 passed |
| workspace | 51 suites, 1046 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| A token sumcheck term is the product of a shown map and a kept map | kept summand form | `Summand::ShownKept` | focused test computes the claim by hand and proves and verifies it |
| The kept map's value is not sent; it stays a term of the row claim | kept-end-value rule | `end_values` length and `row_maps` | focused test sees one shown end value and one kept coefficient |
| The kept coefficient is `eps` times the shown end value as it was sent | output row rule | `row_maps` coefficient | focused test pins it to the first drawn challenge and the shown end value |
| The last round carries the kept maps through their coefficients | last-round rule | `prove_general` last round | focused runs; no kept summand with a shown map existed before, so earlier paths are unchanged |
| A changed round or another kept map is not accepted | negatives | prove/verify refusals | focused test; the verifier leaves another output claim and the prover refuses another kept map |

## Conclusions

- The kept-affine token sumcheck is in place and tested.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The start form (`cstart`: join part, padding, the token claim's mask term and
  the kept terms with their batching coefficient) and its composition follow.
- The merge, pending row and the stack's three linear claims enter the session
  transcript after that.
