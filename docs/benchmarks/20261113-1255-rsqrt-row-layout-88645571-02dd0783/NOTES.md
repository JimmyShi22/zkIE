# Rsqrt index witness row and committed slots

- Scheme: `rsqrt-row-layout`
- Change: code commit `88645571` (RsqrtRow, round-layout digit blocks and
  value slots, digit lookup coverage, no-wrap bound).
- Baseline: previous commit `02dd0783` (range table trace).
- Purpose: prepare the seg-mented rsqrt index relation by deriving every
  witness value from one committed variance and by reserving committed slots
  for them, without wiring the relation rows yet.

## What changed

- `RsqrtRow` in the norm core derives from a nonnegative variance and the
  frozen table real length:
  - `raw = round_half_up(variance, 2^18)` and the exact remainder
    `variance + 2^17 - raw * 2^18`, with its upper-bound slack
    `2^18 - 1 - remainder`;
  - one-hot fine/middle/coarse selectors with the matching nonnegative slack
    (`FINE - 1 - raw`, `raw - FINE`, `COARSE_END - 1 - raw`,
    `raw - COARSE_END`);
  - the middle quotient `(raw - FINE) >> 8` and byte remainder, and the
    coarse quotient `(raw - COARSE_END) >> 16` and 16-bit remainder;
  - the piecewise index and the bound slack `N - 1 - index`; an index at or
    above the table length is rejected;
  - the frozen table value at the index.
  The runtime index function now shares the same named constants.
- `RsqrtRow::check` re-derives the row and compares the index with the
  runtime segmentation.
- The rounding layout reserves the new digit blocks (raw, raw remainder and
  slack, fine slack, middle quotient and byte remainder, middle high slack,
  coarse quotient and 16-bit remainder, index digits, index slack) and the
  value slots (selectors, index value, table value). The existing byte lookup
  key map covers every new digit slot. The rounding widths derive the raw and
  quotient block widths from the variance width, capped so the rounding
  product stays below half the modulus.
- `values_with_rsqrt` writes the new slots and fails closed when a derived
  digit row is wider than its plan block.
- `norm_no_wrap` adds the rsqrt rounding and segment/index expression bounds.

## Evidence

- `RsqrtRow` test: index equals the runtime segmentation at
  `FINE - 1`, `FINE`, `COARSE_END - 1`, `COARSE_END` and beyond; the
  one-hot selector, the segment slacks and the quotient remainders hold; the
  round-half-up boundary is exact on both sides of `2^17`; negative variance,
  zero table length, an index at or past the table length and a tampered index
  are rejected.
- Layout test: the rsqrt slots carry the row values; a raw index wider than a
  narrowed plan block fails closed.
- Focused: `cargo test -p zkie-ops --lib proof::hidden_norm` 26 passed.
- Workspace: 48 suites ok, exit 0 (see `workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (see `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 492) | Change | Evidence | Boundary / open |
| --- | --- | --- | --- |
| A: prove the variance nonnegative and in the real range before the raw rounding | `RsqrtRow::from_variance` rejects negative variance; the raw rounding uses the exact `(variance + 2^17) / 2^18` identity | row test rejects negative variance; the rounding boundary test is exact | the relation that ties the committed variance digits to `RsqrtRow` is the next step |
| A: raw rounding quotient/remainder and no-wrap without the negative clamp | raw digits, exact remainder, remainder slack; width derivation and `norm_no_wrap` rsqrt bound | rounding test around the half step; no-wrap bound test in the layout suite | the relation rows (rounding, remainder bound) are not wired yet |
| A: three one-hot selectors with strict and non-strict segment bounds | selectors and slacks derived with exact boundary conditions; value slots reserved | boundary test at FINE and COARSE_END; selector one-hot check | Booleanity and one-hot relation rows are the next step |
| A: segment index formulas and one committed raw/position | index formula derived; index value slot and index bound slack reserved; table length is the frozen table real length | index equals the runtime formula at all tested points; index >= N rejected | the index formula relation and the rstd lookup key are not wired yet |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused test wall time: 0.02 s (26 tests).
- Workspace: 48 suites ok, 0 failed.
- No wall time or memory claim is made in this round.

## Open items

- Relation indices 16..25: raw rounding, remainder bound, selector
  Booleanity/one-hot, fine/middle/coarse conditional slacks, coarse split,
  index formula, index bound, with the hidden row binding.
- Per-position `RsqrtRow` in `NormWitness` and the fixture wiring.
- The rstd lookup key plan over the committed index and table value slots and
  the negative matrix through the real hidden verifier.
- No push in this round.
