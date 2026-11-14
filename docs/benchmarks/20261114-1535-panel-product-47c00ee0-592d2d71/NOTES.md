# Weighted end-group rows

## Scheme

- Scheme: `panel-product`.
- Current code: `47c00ee0`, following `592d2d71`.
- Previous code: `592d2d71`.

## Purpose

The stack's end group weights its weight values the way the anchor does: each
weight panel's row coefficient is the offset factor times the row's equality
factor at the end point. A panel-product summand carries only the panels'
offsets, widths and row counts; the weight values stay the unpacked row values
the row claims use, so the summand and the group ties share one convention
instead of mixing packed and unpacked values.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib psc_general`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused general-sumcheck tests | 13 passed |
| workspace | 51 suites, 1061 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Each weight panel row's end coefficient is the offset factor times its equality factor at the end point | row end coefficient rule | `PanelEnd`, `Summand::PanelProduct` | focused test computes the coefficient by hand |
| The weight values stay the unpacked row values the row claims use | one-convention rule | the summand takes the group rows as shown values | focused test; the same values feed `group_bases` |
| Point width, group size and panel geometry are checked | boundary rule | `at` validation | focused refusals |

## Conclusions

- The end group's row weighting is in place and tested with the anchor's
  convention.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The group's maps and the chain entry's stack segment follow.
