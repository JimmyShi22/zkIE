# Norm digit reconstruction binding

## Scheme and baseline

- Scheme: `norm-digit-reconstruction-bind`.
- Code: `5e8b525a`; previous commit `02a28f5c`.
- This is partial progress on the norm rounding relation. The eight digit
  reconstruction relations now enter the real hidden path. The prover maps
  build the fixed-offset sum and mean rows, the squares, the variance and the
  four offset/complement rows; the BAT rows bind every digit slot with a
  `256^i` equality weight, the value member or the round u/s slot with the
  negative equality weight, and fold the fixed offset into the target. The
  fixture runs all sixteen relations with the derived mask rows.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Sum/mean reconstruction | digits of value+offset with the offset folded into target | relations 8/11 maps and BAT rows | Real hidden verifier |
| Squares/variance | committed magnitude digits | relations 9/10 | Real hidden verifier |
| Offset/complement rows | u/s digit rows | relations 12-15 | Real hidden verifier |
| t2Lin binding | digit `256^i·eq`, value `−eq`, constant to target | `norm_plain_rows` relations 8-15 | Verifier rebuilds the same rows |
| Derived mask size | rows follow the relation count | fixture: mask rows 16, own mask 32, 16 relations | Plan validation |
| Negative coverage | endpoint/point/message tampering per instance | negative matrix now 16 instances per position | Digit-row tampering still to add |

## Diagnosis and fixes

- The first attempt returned `InvalidProof` from the proximity stage. Stage tags
  showed the norm second stage, samples, plain rows, fold and IOPP were all
  fine; the failure was in the proximity proof.
- First row error: the digit weight used `power.add(&power)` (1, 2, 4, ...)
  instead of `power.mul(256)` (1, 256, 65536, ...).
- Second row error: the right-row constant counted the value coordinate twice
  for relations 8-11. Both are fixed here.
- The earlier inconclusive run was caused by a stale build; the tags were only
  trusted after a forced recompilation.

## Results

- `cargo test --workspace`: 48 suites, 850 passed, 0 failed, 17 ignored.
- Focused `norm`: 26 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## Scope and next

- The lookup forest over the digit slots, the BAT/FOLD binding of its claims and
  the binding-level negatives (d=1, zero, negative, extreme, wrap, digit row)
  are next.
- No claim that integer rounding is proven.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
