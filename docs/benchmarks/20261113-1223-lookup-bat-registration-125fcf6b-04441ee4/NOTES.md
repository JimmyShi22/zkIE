# Lookup claims on the hidden BAT and FOLD

## Scheme and baseline

- Scheme: `lookup-bat-registration`.
- Code: `125fcf6b`; scheme code commit `c0f07094`; previous scheme `04441ee4`.
- Purpose: unify the public table's coordinate order with the leaf inner
  sumcheck, verify the kept-map row identity on committed messages, and bind
  every lookup declaration into the hidden second BAT/FOLD.
- This is partial progress. The lookup commit plan is still assembled by the
  caller, and no rounding claim is made.

## Validation basis

- The seq16 x3 rule is not applicable here: no production plan carries the
  lookup, no seq512 logic and no all-case structural path changes. The
  workspace suite, the focused tests and the byte-identical norm-free proof
  are the acceptance evidence.

## Changes

- The byte table uses the sumcheck index order: the first point coordinate is
  the least significant row bit, so the Boolean basis `e_j` takes `2^j` and a
  general point equals the adjacent-pair folding of the committed table. The
  old high-bit-first fold is asserted different at a general point.
- The kept-map `(U)` row identity
  `<hp_form, w> + eps * prod(psi) * <claim masks> + sum eps * A_i * kept_i = last - eps * A_0`
  holds with `S = 1 + b (alpha - 1 - tau)`, `A_0 = 1`, `A_b = alpha - 1 - tau(point)`;
  it is checked on the committed multiplicity bits at two plane counts, with
  duplicate keys and nonzero claim masks. Changed-bit and changed-mask-entry
  variants break the row.
- The hidden second stage adds the multiplicity-bit and mask members to the
  own group: bound identifiers and trees, sample rows, retained trees and the
  fold mask rows. The lookup rows registered in the same BAT/FOLD batch as the
  norm rows are: the key claim on its committed rounding cells, the
  multiplicity row, the leaf kept-map row, the bit summand hiding row and the
  bit summand shown-end row. The verifier rebuilds the same rows from its
  recomputed claims; the lookup outcome carries the allocated shown and hiding
  mask slots so both sides agree on every unit.
- Binding-level negatives: changed key value, leaf row target, leaf point,
  multiplicity count, bit end value and transcript round are rejected through
  the real hidden verifier.

## Four-column mapping

| Requirement | Mechanism | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| One coordinate order | Plan-fixed index order shared by the table, leaf inner sumcheck and equality weights | `ByteTable::at` adjacent folding; test covers `e_j -> 2^j`, a general point and the differing old fold | Table and leaf inner agree; equality weights unchanged |
| Kept-map identity | `hp_form + eps*prod(psi)*claim masks + sum eps*A_i*kept_i = last - eps*A_0` | `leaves_kept_row_matches_the_committed_bits` at two plane counts | Same random point and committed `b`; kept endpoints are not sent and no shown mask is added |
| Real-message check | Asymmetric table, random non-Boolean point, different `mu`, asymmetric `b`, nonzero claim mask, many variables | Row test on the committed bits; integrated positive and negative paths | Values come from committed messages and the plan; no self-reported source or table range |
| Other call sites | Audit of byte-table and coordinate conventions | Byte table used only by the lookup prover/verifier/tests; the general-sumcheck test tables are independent | No reverse interpretation left in the lookup path |
| BAT/FOLD binding | Every declaration in the same batch | Membership, sample rows, five row kinds, verifier rebuild and claim comparison | Key cells, multiplicity bits and mask units all on physical members |

## Results

- `cargo test --workspace`: 48 suites, 860 passed, 0 failed, 17 ignored.
- Focused `proof::lookup`: 10 passed; `proof::hidden_norm_psc`: 17 passed;
  `norm_lookup_carrier_proves_and_verifies`: 1 passed (with six negatives).
- Hard gate: the norm-free toy output-head proof is 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  byte-identical to the baseline dump.

## Explicitly deferred

- Sizing the lookup commit plan (bit and mask shapes, trees, openings) from the
  parent plan instead of caller-supplied values.
- The rounding claim itself; the digit byte lookup is not yet claimed proven
  as exact integer rounding.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
