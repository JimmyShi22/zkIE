# The request layout's named registration

## Scheme

- Scheme: `layout-registry`.
- Current code: `b67f736c`, following `b9547cf2`.
- Previous code: `b9547cf2`.

## Purpose

The request-level re-derivation needs one table for every quantity the layout
fixes. The chain gains a named registration: mask rows and slots, end values,
stack and merge rounds, opening budget, sample plan, tree layout and stage
plan, all unset in the skeleton until the layout is frozen. Unknown names are
refused and the table never grows, so a re-derived value replaces a named
entry instead of a scattered constant. A zero is explicitly the unset state,
not a final parameter.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_layout_registers_every_quantity_by_name`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused layout test | 1 passed |
| workspace | 51 suites, 1068 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Every layout quantity has one named entry | registration rule | `RequestLayout`, `LayoutValue` | focused test sees the fixed name list |
| Values are set by name and read by name | single table rule | `set`, `get` | focused test |
| Unknown names are refused and the table does not grow | boundary rule | name lookup | focused test |
| The skeleton is explicit, not a final parameter | skeleton rule | zero-valued entries, `is_skeleton` | focused test |

## Conclusions

- The layout has one named registration to be filled when the layout is
  frozen; today it is a skeleton.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The frozen values wait for the W opening's ruling (TURN 538): the W's oracle
  count, encoded rows, sample and mask slots and opening capacity are not fixed
  yet.
