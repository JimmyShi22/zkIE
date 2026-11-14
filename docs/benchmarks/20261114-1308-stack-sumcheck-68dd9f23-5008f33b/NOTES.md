# Stack sumcheck and row ties of the request reduction

## Scheme

- Scheme: `stack-sumcheck`.
- Current code: `68dd9f23`, following `5008f33b`.
- Previous code: `5008f33b`.

## Purpose

The stack's batched sumcheck: both sides' panel values are packed into the
stack cube, the claim is the sum of the packed cell products, and the product
summand runs with plain degree-three rounds. The prover sends each side's
packed end value, and two row ties make each sent end value equal the
verifier's affine recombination of the panels' operand values, so the end
values stay tied to the reads and the shown weights. The layout additionally
requires panels to span whole powers of two, which makes a shown weight one
value per row.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_stack`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 7 passed |
| workspace | 51 suites, 1045 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Both sides are packed cube maps at the panels' aligned offsets; the first side carries the call factor, the batch coefficient and the row weights | packed stack maps | `StackReduction::side_maps` | focused test checks cells at both offsets and refuses mismatched shapes and missing batch coefficients |
| The stack's claim is the sum of the packed cell products | stack summand rule | `StackReduction::sum`, `prove_stack`/`verify_stack` | focused test computes the claim by hand; prover and verifier agree; a changed round leaves another output claim; another target is refused |
| The prover sends each side's packed end value at the stack point | end-value rule | `StackReduction::end_values`, `mle` | focused test reads the values back at the point |
| Each sent end value must equal its affine recombination of the panels' operand values | row ties | `StackReduction::check_row_ties` | focused test; changed reads and a changed shown weight are refused |
| A shown weight is one value per row | whole-power-of-two panels | `panel_layout` validation | focused test refuses a three-row panel |
| Equality factors pair bit `j` with the coordinate of its range | low-bit pairing | `eq_low`, `eq_low_at` | focused test (unchanged from the layout block) |

## Conclusions

- The stack sumcheck and both row ties are in place and tested.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The start form (`cstart`: join part, weight-term constant and mask terms)
  and the weight panels' extra shown values precede the session wiring.
- The weight fork (`wsForkEnd`/`wsForkMerge`), the leftover PSC claim
  (`wsForkLin`), and the pending row enter the session transcript after that.
