# Exact norm rounding decomposition

## Scheme and baseline

- Scheme: `norm-rounding-decomposition`.
- Code: `bdb18eaf`; previous commit `1a605c91`.
- This is partial progress on the norm core. It replaces the loose remainder
  bounds with the exact round-half-up interval, computes the offset and
  complement of the mean and variance remainders, derives canonical byte-digit
  rows of both, and reconstructs every digit row in the witness check. The
  committed messages and the proof relations for these values are not part of
  this step.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Exact rounding interval | the exact integer remainder of round-half-up division, `lo = -floor(d/2)`, `hi = d - 1 - floor(d/2)` | `NormCoreRow::from_values` computes the remainder; `check` enforces the interval | Witness level only; the proof interval relation is next |
| Offset and complement | `u = r - lo`, `s = d - 1 - u`, both nonnegative | new `mean_offset`, `mean_complement`, `variance_offset`, `variance_complement` fields | Not yet committed or bound |
| Byte digit rows | canonical base-256 digits, least significant first | `unsigned_digits` for the offsets and complements; `digits_value` reconstructs and rejects empty or leading-zero rows | The byte lookup over these digits is next |
| Boundary semantics | even divisor tie falls to `lo`; odd divisor and unit divisor behave exactly | boundary test covers tie, negative sum, odd divisor and `d = 1` | No proof coverage yet |

## Results

- `cargo test --workspace`: 48 suites, 843 passed, 0 failed, 17 ignored.
- Focused `hidden_norm`: 13 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: this is a witness-level block with no end-to-end path
  change; the norm-free byte gate is the required byte gate for the step. No
  seq512 case was run because the block does not touch the seq512 path.

## Positive and negative coverage

- The boundary test covers an even-divisor tie (`r = lo`, `u = 0`), a negative
  sum, an odd divisor and the unit divisor.
- The existing tamper test still rejects a changed mean, a changed remainder, a
  changed sum digit and a changed squares digit; it now also rejects a changed
  offset and a changed complement digit.
- The digit reconstruction rejects empty rows, leading-zero rows and values
  that do not fit the row.

## Scope and next

- No committed digit or remainder messages, no proof relations and no
  BAT/FOLD binding.
- No byte lookup over the digit rows; no rsqrt, weight product or output
  consumer.
- No rounding, soundness or complete-norm claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
