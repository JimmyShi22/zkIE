# Byte lookup driver

## Scheme and baseline

- Scheme: `byte-lookup-driver`.
- Code: `1e3be78f`; previous commit `a1c96798`.
- This is partial progress on the lookup module. It adds the public byte
  table, the plan and message version, the challenge source, the mask-slot
  cursor, and the prover/verifier driver for the tie, the key and table product
  trees, the leaf and bit sumchecks and the multiplicity sum row. It does not
  yet add the key or multiplicity committed messages or bind the returned
  claims to BAT/FOLD.

## Four-column mapping

| Requirement | Reference anchor (read) | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Public byte table and its multilinear extension | public range table use in the lookup; table evaluation in the byte case | `ByteTable` and `TableAt` over eight coordinates | Table is public and fixed; the plan fixes eight table coordinates |
| Key side product tree | key product tree and its level sumchecks | `key_and_table_levels`, key-level loop with `WeightedProductInner` and `line` | Key vector is a witness vector in this step; the committed key message and BAT binding are next |
| Table side product tree and tie | table product tree, plane roots, tie steps | table leaf values `1 + b(alpha - 1 - tau)`, table-level loop, tie loop with `TieFirst`/`TieNext` and `ScalarInner` | Tie combines the roots; mask slots and claim masks follow the line chain |
| Leaves, bits and multiplicity sum | leaf summand, bit summand, multiplicity row | `LeavesInner`, `WeightedBitsInner`, multiplicity coefficient vector | Claims and rows are returned; no row binding yet |
| Version and encoding | versioned plan and messages | `LOOKUP_VERSION`, `LookupPlan::digest`, `LookupProof::encode`/`decode` with round/end counts and trailing checks | The plan also fixes the table size and the relation/plane counts |
| Mask slots | one entry for each sent scalar | `MaskSlots`, `ProverMasks`, per-step hiding and shown slots | The mask message commitment and row binding are next |

## Results

- `cargo test --workspace`: 48 suites, 842 passed, 0 failed, 17 ignored.
- Focused `lookup`: 4 passed, 0 failed; focused `psc_general`: 8 passed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.

## Positive and negative coverage

- An honest four-key vector with multiplicities proves and verifies; the
  prover and verifier outcomes (key claim, multiplicity row and rows) agree.
- A key outside the table cannot be proved.
- Malformed messages (truncated tie round, cleared table round, missing bit
  end value, missing key level) are rejected; wrong version, trailing bytes
  and truncation are rejected by the decoder.
- These are structural negatives. The binding negatives (wrong committed key
  or multiplicity, wrong row, wrong mask) belong to the next step, where the
  claims enter BAT/FOLD.

## Scope and next

- No committed key or multiplicity message, no mask message, no BAT/FOLD/CS/
  BASE binding and no norm digit row.
- No soundness, round-half-up or complete-norm claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
