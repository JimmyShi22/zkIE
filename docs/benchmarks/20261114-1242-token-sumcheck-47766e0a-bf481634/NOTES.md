# Token sumcheck of the request reduction

## Scheme

- Scheme: `token-sumcheck`.
- Current code: `47766e0a`, following `bf481634`.
- Previous code: `bf481634`.

## Purpose

The request reduction starts with its token dimension: the claim is the
equality-weighted sum of the token map at a token point, the sumcheck runs one
equality weight per token bit, and the inner map rounds carry the equality
factors of the later variables while the verifier weights the variable being
bound. The general sumcheck also gains a sum-of-products summand over shown
maps for the panel combinations of the layer relation.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_reduction` and
  `cargo test -p zkie-ops --lib psc_general`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused token tests | 3 passed |
| focused general-sumcheck tests | 10 passed |
| workspace | 51 suites, 1037 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The token claim is the equality-weighted sum of the token map at a token point | token sumcheck claim rule | `token_claim` | focused test computes the claim by hand |
| One equality weight per token bit, in bit order | equality round form | `token_variables` | focused test; prove and verify agree |
| The inner rounds carry the later variables' equality factors; the verifier weights the bound variable | weighted inner-sumcheck rule | `WeightedMapInner` | focused test proves and verifies the token claim |
| Panel combinations need a sum of products of shown maps | summand forms | `Summand::ProductSum` | focused test checks the form and the out-of-range index |

## Conclusions

- The token sumcheck and the product-sum summand are in place and tested.
- No session or proof message is touched; the toy proof is unchanged.

## Open items

- The inner-dimension sumcheck, the stack rounds, the affine row claims
  (`rowS`, `A0`, `B0`) and the weight claims follow.
- The session transcript wiring waits for those pieces.
