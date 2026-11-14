# Proving the weight-claim merge

## Scheme

- Scheme: `merge-psc`.
- Current code: `651a8d56`, following `c69d7593`.
- Previous code: `c69d7593`.

## Purpose

Compose the weight-claim merge with the one-map sumcheck. The claims' shown
values are fixed before the batch challenge; its powers batch the values into
the sumcheck's target and fold the claims' mask slots. The batched map is
summed over the matrix's column cube, the end value is masked by the
sumcheck's last challenge, and the verified point yields each claim's row
coefficient and the accumulated row weights. The verifier draws the same
batch challenge, rebuilds the target and mask terms from the shown values,
checks the sumcheck and derives the same rows and weights.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_weights`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 8 passed |
| workspace | 51 suites, 1033 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The claims' values are fixed before the batch challenge, whose powers form the sumcheck's start | merge message order | `prove_merge` draws the challenge first, then builds the target and the mask terms | focused test: prove and verify agree |
| The batched map is summed over the matrix's column cube; the end value is masked by the last challenge | merge sumcheck rule | `prove_merge`/`verify_merge` with the one-map summand | focused test: the same rows and weights on both sides |
| The claim's column factor and batch coefficient become its row coefficient; the row weights accumulate | merge-to-row rule | `merge_claims` called from both sides on the verified point | focused test: rows and accumulated weights equal |
| A changed round leaves another output claim for the caller's binding | sumcheck output-claim rule | the verifier returns the output claim | focused test: the changed round's output differs |

## Conclusions

- The merge's arithmetic and its sumcheck are composed and tested.
- No session or proof message is touched; the toy proof is unchanged.

## Open items

- The session transcript (hiding entries, mask slots, chain adapters) is not
  wired; the test uses zero claim masks and synthetic maps.
- The leftover sumcheck claim's chain check and the source-layer reads follow.
