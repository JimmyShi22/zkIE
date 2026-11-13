# Norm digit reconstruction maps

## Scheme and baseline

- Scheme: `norm-digit-reconstruction-maps`.
- Code: `dd577c00`; previous commit `e4cd238d`.
- This is partial progress on the digit reconstruction relations. The prover
  maps for relations 8-15 now cover the fixed-offset sum and mean digits, the
  squares, the variance and the four offset/complement rows. Every map places
  the digit row at coordinates 0..L with weight 256^i, the fixed offset at
  coordinate L, and the value at coordinate `columns`; the right map is one on
  the used coordinates. A digit row shorter than the fixed width is zero padded.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Sum reconstruction | digits of value+offset | relation 8 map; offset constant coordinate | BAT row next |
| Squares/variance | committed magnitude digits | relations 9/10 maps | BAT row next |
| Mean reconstruction | digits of mean+offset | relation 11 map; offset constant | BAT row next |
| Offset/complement rows | u/s digit rows | relations 12-15 maps | BAT row next |
| Unit check | every map sums to zero on toy rows | `digit_reconstruction_maps_sum_to_zero` | Isolates map correctness from BAT |

## Results

- `cargo test --workspace`: 48 suites, 850 passed, 0 failed, 17 ignored.
- Focused `norm`: 26 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## Scope and next

- The BAT rows for relations 8-15, their fixture wiring (0-15, mask rows 16,
  own mask 32, positions x 16) and the lookup forest are next.
- The map unit test is not a hidden-verifier proof; no rounding claim is made.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
