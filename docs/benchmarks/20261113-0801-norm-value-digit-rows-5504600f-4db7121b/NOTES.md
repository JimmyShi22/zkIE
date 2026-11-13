# Norm value digit rows

## Scheme and baseline

- Scheme: `norm-value-digit-rows`.
- Code: `5504600f`; previous commit `4db7121b`.
- This is partial progress on the norm core. It adds the mean sign and magnitude
  digits and checks every magnitude digit row and sign of sum, mean, squares and
  variance against its integer value. The committed layout, the source
  relations, the range proof and the lookup are not part of this step.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Signed value representation | sign bit plus canonical base-256 magnitude digits | `sum_negative`/`sum_digits`, new `mean_negative`/`mean_digits`; `squares_digits`, `variance_digits` | Witness level only |
| Digit reconstruction | each digit row rebuilt to its integer value | `digits_value` in the witness check for all four rows | No committed row or relation yet |
| Sign consistency | the sign bit matches the signed integer | checks `sum_negative` and `mean_negative` against their values | Zero has a false sign |
| Source of v/q | sum and squares from the input row, mean and variance from the rounded quotients | `NormCoreRow::from_values` | The proof-side source relations are next |

## Results

- `cargo test --workspace`: 48 suites, 843 passed, 0 failed, 17 ignored.
- Focused `hidden_norm`: 13 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: witness-level block, no end-to-end path change; no seq512
  case was run because the block does not touch the seq512 path.

## Positive and negative coverage

- The fixture and boundary tests rebuild the new mean digit row; a negative sum
  gives a true mean sign and the magnitude digit row.
- The tamper test now rejects a changed mean digit and a changed mean sign in
  addition to the existing digit, offset and remainder tampers.
- Digit reconstruction rejects empty rows and leading-zero rows.

## Scope and next

- No committed v/q digit or sign message; no source relation tying the rows to
  the committed sum/squares/mean/variance.
- No range/no-wrap proof; the trusted input bound and the real valid width still
  have to be measured from the fixture and recorded.
- No lookup forest, BAT/FOLD binding or rounding claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
