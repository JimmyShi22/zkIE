# Panel stack layout and affine recombination

## Scheme

- Scheme: `stack-affines`.
- Current code: `5008f33b`, following `0ecc59f4`.
- Previous code: `0ecc59f4`.

## Purpose

The layer relation reduces its panels together. This block fixes the panel
layout: every panel gets its width and an aligned offset, largest panel first,
and the stack's width follows from the total. It computes the equality factors
of a range of the stack challenge coordinates with the low-bit pairing, and
builds the verifier's affine recombination of the panels' operand values:
the batch coefficient, the call factor and the row weights fold into the
first side's terms, while the second side's terms are either reads or shown
weight values.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_stack`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 3 passed |
| workspace | 51 suites, 1041 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Panels are aligned and laid out largest first; the stack's width covers them | panel layout rule | `panel_layout` | focused test checks the widths, offsets and total |
| Equality factors of a range pair bit `j` with its coordinate | stack equality-range rule | `eq_low`, `eq_low_at` | focused test checks every entry of a range |
| The first side's terms fold the batch coefficient, the call factor and the row weights; the second side's terms are reads or shown weights | affine recombination rule | `stack_affines`, `AffTerm`, `StackAffines` | focused test checks one coefficient and the shown terms |

## Conclusions

- The stack's layout and affine recombination are in place and tested.
- No session or proof message is touched; the toy proof is unchanged.

## Open items

- The stack sumcheck's rounds and the row claims (`rowS`, `A0`, `B0`) follow.
- The session transcript wiring waits for the relation reduction's full chain.
