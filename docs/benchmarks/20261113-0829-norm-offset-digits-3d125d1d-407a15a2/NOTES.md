# Norm offset digit rows

## Scheme and baseline

- Scheme: `norm-offset-digits`.
- Code: `3d125d1d`; previous commit `407a15a2`.
- This is partial progress on the norm range representation. The rounding witness
  now uses the reference's fixed-offset digit form instead of a sign bit: the
  committed digits of the sum and mean are the digits of `value + offset`, with
  public plan offsets; the sign slots and the sign/magnitude product are
  removed. The reconstruction, source and lookup relations are not part of this
  step.

## Reference reading

- Digit rows: a cell holds the digit minus 128; the digit is `cell + 128` and is
  range-checked by a byte lookup with a per-digit multiplier. Carries are
  six-digit blocks reconstructed with a fixed 2^47 offset; the carry equation is
  `W_s + c_s - v_s - 256 c_(s+1) = 0`.
- The four-limb value uses a single eight-digit reconstruction with no carry
  chain; only the five-limb value needs the carry blocks. The digit-sum forms
  and the ROW key form use `(cell + 128)` with the lookup multiplier.
- The first value in this batch fits below 2^56, so the four-limb-style single
  reconstruction applies; multi-limb carries are not introduced.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Fixed-offset signed values | digit sum of `value + offset`; offset public in the plan | `NormRoundLayout::sum_offset`/`mean_offset`, `shifted_digits` | Offsets fixed at 2^25 for the sum and 2^16 for the mean |
| No sign slot, no product | signed values enter as offset quantities | sign slots removed; six fixed slots and eight digit blocks; layout has 35 slots padded to 64 | Product and sign are gone |
| No-wrap | reconstruction and each equation stay below `p/2` | `norm_no_wrap(input_bound, valid, widths, offsets)` | Still plan validation only; the relations are next |
| Zero and negatives | digit sum `K` for zero, `K + value` for negatives | tests cover zero, a negative sum and the fixed offset | Single canonical byte representation; no dual encoding without a carry chain |
| Version | plan fixes the new layout and offsets | `NORM_BIND_VERSION = 4`; offsets enter the plan digest | Round message stays generic |

## Results

- `cargo test --workspace`: 48 suites, 848 passed, 0 failed, 17 ignored.
- Focused `norm`: 24 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## No-wrap with the real bounds

- Widths: sum 5, squares 7, variance 6, mean 3, offset 2; offsets 2^25 and
  2^16. `Bv = 2^40 - 1 + 2^25`, `Bq = 2^24 - 1 + 2^16`.
- Bounds with `X = 22,855,985`, `d = 768`: sum 1,117,098,578,687; squares
  1,282,417,637,556,158,463 (margin 7.19); mean 1,112,480,415,743; variance
  288,230,376,151,711,743; reconstruction 2^56; offsets 131,838; remainder
  2,304. All below `p/2`.
- `256^L + offset < p` for both signed rows, so the digit sum plus the offset
  stays canonical.

## Zero, negatives and wrap

- Zero is the digit row of the offset `K`; with a fixed width and base-256
  digits the representation is unique. No carry chain exists in this batch, so
  the two-encoding zero case of a multi-limb carry row does not arise; this is
  recorded explicitly rather than left implicit.
- A negative sum or mean uses `K + value >= 0`; the test covers a negative sum
  and the fixed offset.
- A digit row cannot represent `K + p` or any wrapped value because
  `256^L + offset < p`; a value outside the width fails the prover before
  emission, and the verifier's digit bound comes from the committed row.

## Scope and next

- No reconstruction relation, no u+s/r=lo+u/v=q*d+r relation, no lookup forest
  and no BAT/FOLD binding.
- The source of the sum/mean digits is the same integer core, checked slot by
  slot before the first message; the proof-side binding is next.
- No claim that integer rounding is proven.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
