# Norm round plan layout

## Scheme and baseline

- Scheme: `norm-round-plan-layout`.
- Code: `fd880c51`; previous commit `badd1a1a`.
- This is partial progress on the norm range representation. It fixes the
  rounding witness widths, the slot map, the zero padding and the integer
  no-wrap bounds in the norm plan, and raises the plan version. The committed
  message, the source relations, the lookup forest and the BAT/FOLD binding are
  not part of this step; the first message rejects a plan that carries the new
  group until that message exists.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Fixed digit widths and slot map | plan-fixed committed layout; zero padded rows | `NormRoundWidths`, `NormRoundLayout`, `NormRoundLayout::values` | Message commitment is next |
| Signed value source | sign slot plus magnitude digits for sum, mean, squares, variance | slots 6 and 7 and the four magnitude digit blocks | Source relations are next |
| Remainder decomposition | `u`, `s`, `r` slots with `u+s=d-1` and `r=lo+u` | six fixed slots; offset width covers the valid range | Relation proof is next |
| No-wrap bound | every integer equation stays below `p/2` | `norm_no_wrap` checks the seven bounds; plan validation rejects a failing width set | Relations must enforce the same bounds |
| Version and fail closed | plan fixes the new layout and version | `NORM_BIND_VERSION = 3`; `NormPlan::round`; first message returns `UnsupportedPrimitive` while the round message is absent | Round first message is next |

## Config source for the widths and bounds

- Fixture: gpt2-seq16 on the test machine, `public/model-commitment.bin` and the
  `proof.bin` trace, verified with the model trace plan. The final norm input is
  the output of step 4442, consumed by step 4443.
- Valid tensor: 16 positions x 768 valid columns (1024 padded, padding all
  zero); `d = 768`; fixed-point scale `2^16`.
- Measured: max |x| = 22,855,985; min x = -13,999,866; max |sum| = 25,379,944;
  max |mean| = 33,047; max squares = 1,363,617,134,835,326; max variance =
  1,775,543,144,317.
- Fixed widths: sum 5, squares 7, variance 6, mean 3, offset 2 bytes. All
  `256^L < p`; the largest digit sum is 2^56.
- No-wrap with `p = 2^64 - 2^32 + 1` and `p/2 = 9,223,372,034,707,292,160`:
  sum 1,117,065,024,255; squares 1,278,424,730,478,247,935 (margin 7.21);
  mean 1,112,396,529,663; variance 288,230,376,151,711,743 (margin 32);
  reconstruction 72,057,594,037,927,936 (margin 128); offsets 131,838;
  remainder 2,304. A four-byte mean width fails the bound.

## Trust boundary

- `X` is the external trusted input contract of the fixture, not a value proven
  in this chain: **production dependency A, the input range is not verified
  upstream**, and it stays recorded as such.
- The ranges of `v`, `q`, `u`, `s` and their digits are to be verified in the
  chain by the digit rows, the reconstruction relations, the sign slots and the
  no-wrap equations; the plan fields alone are not the proof.

## Results

- `cargo test --workspace`: 48 suites, 847 passed, 0 failed, 17 ignored.
- Focused `hidden_norm`: 17 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run. The bounds for other models and for seq512 are not part of this
  batch and need their own measured configuration source.

## Positive and negative coverage

- The layout test checks the slot map, the padded zero digits and the zero tail
  of the message row.
- The width tests reject an eight-byte row, an offset width too small for the
  valid range, a digit row that does not fit its width and a message row shorter
  than the layout.
- The no-wrap test checks the measured bounds and rejects a four-byte mean
  width.
- The plan test checks that a round group enters the plan digest, rejects an
  oracle collision with the existing groups and rejects a failing width set; the
  first message rejects a plan carrying the group until the message is added.

## Scope and next

- No committed round message, no first-message round field, no openings.
- No source/reconstruction relation, no lookup forest and no BAT/FOLD binding.
- No claim that integer rounding is proven.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
